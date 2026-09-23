// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "Control.h"
#include "ExtendedGridOrbitals.h"
#include "LocGridOrbitals.h"
#include "MGmol.h"
#include "MGmol_MPI.h"
#include "MPIdata.h"
#include "mgmol_run.h"

#ifdef MGMOL_HAS_LIBROM
#include "librom.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <time.h>
#include <vector>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

int main(int argc, char** argv)
{
    int mpirc = MPI_Init(&argc, &argv);
    if (mpirc != MPI_SUCCESS)
    {
        std::cerr << "MPI Initialization failed!!!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 0);
    }

    MPI_Comm comm = MPI_COMM_WORLD;

    mgmol_init(comm);

    std::string input_filename("");
    std::string lrs_filename;
    std::string constraints_filename("");

    float total_spin = 0.;
    bool with_spin = false;

    po::variables_map vm;

    if (MPIdata::onpe0)
    {
        read_config(argc, argv, vm, input_filename, lrs_filename,
            constraints_filename, total_spin, with_spin);
    }

    MGmol_MPI::setup(comm, std::cout, with_spin);
    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    MPI_Comm global_comm = mmpi.commGlobal();

    Control::setup(global_comm, with_spin, total_spin);
    Control& ct = *(Control::instance());

    ct.setOptions(vm);

    int ret = ct.checkOptions();
    if (ret < 0) return ret;

    mmpi.bcastGlobal(input_filename);
    mmpi.bcastGlobal(lrs_filename);

    {
        if (MPIdata::onpe0)
        {
            std::cout << "-------------------------" << std::endl;
            std::cout << "Construct MGmol object..." << std::endl;
            std::cout << "-------------------------" << std::endl;
        }

        MGmolInterface* mgmol =
            new MGmol<ExtendedGridOrbitals<ORBDTYPE>>(
                global_comm, *MPIdata::sout, input_filename, lrs_filename,
                constraints_filename);

        if (MPIdata::onpe0)
        {
            std::cout << "-------------------------" << std::endl;
            std::cout << "MGmol setup..." << std::endl;
            std::cout << "-------------------------" << std::endl;
        }

        mgmol->setup();

        if (MPIdata::onpe0)
        {
            std::cout << "-------------------------" << std::endl;
            std::cout << "Setup done..." << std::endl;
            std::cout << "-------------------------" << std::endl;
        }

        std::vector<double> positions;
        mgmol->getAtomicPositions(positions);

        std::vector<short> anumbers;
        mgmol->getAtomicNumbers(anumbers);

        if (MPIdata::onpe0)
        {
            std::cout << "Positions:" << std::endl;
            std::vector<short>::iterator ita = anumbers.begin();

            for (std::vector<double>::iterator it = positions.begin();
                 it != positions.end(); it += 3)
            {
                std::cout << *ita;

                for (int i = 0; i < 3; i++)
                    std::cout << "    " << *(it + i);

                std::cout << std::endl;
                ita++;
            }
        }

        Mesh* mymesh = Mesh::instance();
        const pb::Grid& mygrid = mymesh->grid();
        const pb::PEenv& myPEenv = mymesh->peenv();

        const int rdim = ct.getROMOptions().num_orbbasis;

        if (rdim != ct.numst)
        {
            std::cerr << "The number of functions in the ROM basis file, "
                      << rdim << " is not equal to ct.numst, " << ct.numst
                      << std::endl;
            MPI_Abort(mmpi.commSameSpin(), 0);
        }

        std::shared_ptr<ProjectedMatricesInterface> projmatrices =
            mgmol->getProjectedMatrices();

        ExtendedGridOrbitals<ORBDTYPE> orbitals("new_orbitals", mygrid,
            mymesh->subdivx(), ct.numst, ct.bcWF, projmatrices.get(), nullptr,
            nullptr, nullptr, nullptr);

        HDFrestart h5file(ct.restart_file, myPEenv, ct.restart_file_type);
        orbitals.read_hdf5(h5file);

        //
        // evaluate energy and forces again, with wavefunctions
        // frozen to solution of previous problem
        //

        // reset initial DM to test iterative solve for it
        projmatrices->setDMuniform(ct.getNelSpin());
        ct.dm_inner_steps = 50;
        std::vector<double> forces;

        double eks = mgmol->evaluateDMandEnergyAndForces(
            &orbitals, positions, anumbers, forces);

        // print out results
        if (MPIdata::onpe0)
        {
            std::cout << "Eks1 : " << eks << std::endl;
            std::cout << "Forces1 :" << std::endl;
            for (std::vector<double>::iterator it = forces.begin();
                 it != forces.end(); it += 3)
            {
                for (int i = 0; i < 3; i++)
                    std::cout << "    " << *(it + i);
                std::cout << std::endl;
            }
        }

        orbitals.set(ct.getROMOptions().basis_file, ct.numst);
        orbitals.orthonormalizeLoewdin();
        orbitals.setDataWithGhosts(true);

        orbitals.setIterativeIndex(10);

        projmatrices->setDMuniform(ct.getNelSpin());
        projmatrices->printDM(std::cout);

        eks = mgmol->evaluateDMandEnergyAndForces(
            &orbitals, positions, anumbers, forces);

        if (MPIdata::onpe0)
        {
            std::cout << "Eks2: " << eks << std::endl;
            std::cout << "Forces2 :" << std::endl;
            for (std::vector<double>::iterator it = forces.begin();
                 it != forces.end(); it += 3)
            {
                for (int i = 0; i < 3; i++)
                    std::cout << "    " << *(it + i);
                std::cout << std::endl;
            }
        }

        delete mgmol;

    }

    mgmol_finalize();

    mpirc = MPI_Finalize();

    if (mpirc != MPI_SUCCESS)
    {
        std::cerr << "MPI Finalize failed!!!" << std::endl;
    }

    time_t tt;
    time(&tt);

    if (onpe0)
        std::cout << " Run ended at " << ctime(&tt) << std::endl;

    return 0;
}
#endif

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

#include <iostream>
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
    bool with_spin    = false;

    po::variables_map vm;
    if (MPIdata::onpe0)
    {
        read_config(argc, argv, vm, input_filename, lrs_filename,
            constraints_filename, total_spin, with_spin);
    }

    MGmol_MPI::setup(comm, std::cout, with_spin);
    MGmol_MPI& mmpi      = *(MGmol_MPI::instance());
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

        MGmolInterface* mgmol;
        if (ct.isLocMode())
            mgmol = new MGmol<LocGridOrbitals<ORBDTYPE>>(global_comm, *MPIdata::sout,
                input_filename, lrs_filename, constraints_filename);
        else
            mgmol = new MGmol<ExtendedGridOrbitals<ORBDTYPE>>(global_comm, *MPIdata::sout,
                input_filename, lrs_filename, constraints_filename);

        if (MPIdata::onpe0)
        {
            std::cout << "-------------------------" << std::endl;
            std::cout << "MGmol setup..." << std::endl;
            std::cout << "-------------------------" << std::endl;
        }
        mgmol->setup();

        std::vector<double> positions;
        mgmol->getAtomicPositions(positions);
        std::vector<short> anumbers;
        mgmol->getAtomicNumbers(anumbers);
        if (MPIdata::onpe0)
        {
            std::cout << "Positions:" << std::endl;
            auto ita = anumbers.begin();
            for (auto it = positions.begin(); it != positions.end(); it += 3)
            {
                std::cout << *ita;
                for (int i = 0; i < 3; i++) std::cout << "    " << *(it + i);
                std::cout << std::endl;
                ita++;
            }
        }

        // Quench: compute ground-state energy, forces, and density matrix
        std::vector<double> forces;
        double eks = mgmol->evaluateEnergyAndForces(positions, anumbers, forces);

        // Dump converged orbitals/DM to WFinit for the TDDFT driver to read
        mgmol->dumpRestart();

        if (MPIdata::onpe0)
        {
            std::cout << "Ground state Energy, Eks : " << eks << std::endl;
            std::cout << "Ground state Forces :" << std::endl;
            for (auto it = forces.begin(); it != forces.end(); it += 3)
            {
                for (int i = 0; i < 3; i++) std::cout << "    " << *(it + i);
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
    if (onpe0) std::cout << " Run ended at " << ctime(&tt) << std::endl;

    return 0;
}
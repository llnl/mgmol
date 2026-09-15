// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

//
//                  main.cc
//
//    Description:
//        Real grid, finite difference, molecular dynamics program
//        for with nonorthogonal localized orbitals.
//
//        Uses Mehrstellen operators, multigrid accelerations, and
//        non-local pseudopotentials.
//
//     Includes LDA and PBE exchange and correlation functionals.
//
// Units:
//   Potentials, eigenvalues and operators in Rydberg
//   Energies in Hartree
//
#include "rom_workflows.h"

// A helper function
template <class T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
    return os;
}

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

    // read runtime parameters
    std::string input_filename("");
    std::string lrs_filename;
    std::string constraints_filename("");

    float total_spin = 0.;
    bool with_spin   = false;

    po::variables_map vm;

    // use configure file if it can be found
    // std::string config_filename("mgmol.cfg");

    // read options from PE0 only
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

    unsigned ngpts[3]    = { ct.ngpts_[0], ct.ngpts_[1], ct.ngpts_[2] };
    double origin[3]     = { ct.ox_, ct.oy_, ct.oz_ };
    const double cell[3] = { ct.lx_, ct.ly_, ct.lz_ };
    Mesh::setup(mmpi.commSpin(), ngpts, origin, cell, ct.lap_type);

    mmpi.bcastGlobal(input_filename);
    mmpi.bcastGlobal(lrs_filename);

    /* Get ROM driver mode */
    ROMPrivateOptions rom_options = ct.getROMOptions();
    ROMStage rom_stage            = rom_options.rom_stage;

    // Enter main scope
    {
        MGmolInterface* mgmol;
        if (ct.isLocMode())
            mgmol = new MGmol<LocGridOrbitals<ORBDTYPE>>(global_comm,
                *MPIdata::sout, input_filename, lrs_filename,
                constraints_filename);
        else
            mgmol = new MGmol<ExtendedGridOrbitals<ORBDTYPE>>(global_comm,
                *MPIdata::sout, input_filename, lrs_filename,
                constraints_filename);

        mgmol->setup();

        switch (rom_stage)
        {
            case (ROMStage::OFFLINE):
                if (ct.isLocMode())
                    readRestartFiles<LocGridOrbitals<ORBDTYPE>>(mgmol);
                else
                    readRestartFiles<ExtendedGridOrbitals<ORBDTYPE>>(mgmol);
                break;

            case (ROMStage::BUILD):
                if (ct.isLocMode())
                    buildROMPoissonOperator<LocGridOrbitals<ORBDTYPE>>(mgmol);
                else
                    buildROMPoissonOperator<ExtendedGridOrbitals<ORBDTYPE>>(
                        mgmol);
                break;

            case (ROMStage::ONLINE_POISSON):
                if (ct.isLocMode())
                    runPoissonROM<LocGridOrbitals<ORBDTYPE>>(mgmol);
                else
                    runPoissonROM<ExtendedGridOrbitals<ORBDTYPE>>(mgmol);
                break;

            case (ROMStage::TEST_POISSON):
                if (ct.isLocMode())
                    testROMPoissonOperator<LocGridOrbitals<ORBDTYPE>>(mgmol);
                else
                    testROMPoissonOperator<ExtendedGridOrbitals<ORBDTYPE>>(
                        mgmol);
                break;

            case (ROMStage::TEST_RHO):
                if (ct.isLocMode())
                    testROMRhoOperator<LocGridOrbitals<ORBDTYPE>>(mgmol);
                else
                    testROMRhoOperator<ExtendedGridOrbitals<ORBDTYPE>>(mgmol);

            case (ROMStage::TEST_ION):
                if (ct.isLocMode())
                    testROMIonDensity<LocGridOrbitals<ORBDTYPE>>(mgmol);
                else
                    testROMIonDensity<ExtendedGridOrbitals<ORBDTYPE>>(mgmol);

                break;

            default:
                std::cerr << "rom_main error: Unknown ROM stage" << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 0);
                break;
        }

        delete mgmol;

    } // close main scope

    mgmol_finalize();

    mpirc = MPI_Finalize();
    if (mpirc != MPI_SUCCESS)
    {
        std::cerr << "MPI Finalize failed!!!" << std::endl;
    }

    time_t tt;
    time(&tt);
    if (onpe0) std::cout << " Run ended at " << ctime(&tt) << std::endl;

    // MemTrack::TrackDumpBlocks();

    //    MemTrack::TrackListMemoryUsage();

    return 0;
}

// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE
#include "Control.h"
#include "LocGridOrbitals.h"
#include "MGmol_MPI.h"
#include "MPIdata.h"
#include "MatricesBlacsContext.h"
#include "Mesh.h"
#include "PackedCommunicationBuffer.h"
#include "ReplicatedWorkSpace.h"
// #include "SparseDistMatrix.h"
#include "tools.h"

#include <cassert>
#include <mpi.h>

#ifdef MGMOL_USE_BLIS
#include <blis/blis.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_CNR
#include <mkl.h>
#endif

int mgmol_init(MPI_Comm comm)
{
    // change handling of memory allocation errors
    std::set_new_handler(noMoreMemory);

    std::cout.sync_with_stdio();

    MPI_Comm_rank(comm, &MPIdata::mype);
    assert(mype > -1);
    MPIdata::onpe0 = (MPIdata::mype == 0);

    if (mype == 0)
    {
#if defined(__GNUC__) && !defined(__clang__)
        std::cout << "GCC " << __GNUC__ << "." << __GNUC_MINOR__ << "."
                  << __GNUC_PATCHLEVEL__ << std::endl;
#elif defined(__clang__)
        std::cout << "Clang " << __clang_major__ << "." << __clang_minor__
                  << "." << __clang_patchlevel__ << std::endl;
#else
        std::cout << "Unknown Compiler" << std::endl;
#endif

        int version, subversion;
        MPI_Get_version(&version, &subversion);
        std::cout << "MPI Version: " << version << "." << subversion
                  << std::endl;
    }

#ifdef MGMOL_USE_BLIS
    bli_init();
#endif

#ifdef HAVE_MAGMA
    magma_int_t magmalog = magma_init();
    if (magmalog == MAGMA_SUCCESS)
    {
        std::cout << "MAGMA Initialization: success" << std::endl;
    }
    else
    {
        if (magmalog == MAGMA_ERR_UNKNOWN)
            std::cout << "MAGMA Initialization: unknown error" << std::endl;
        if (magmalog == MAGMA_ERR_HOST_ALLOC)
            std::cout << "MAGMA Initialization: fails host alloc" << std::endl;
        return 1;
    }
#endif

#ifdef _OPENMP
    if (MPIdata::onpe0)
    {
        std::cout << " " << omp_get_max_threads() << " thread"
                  << (omp_get_max_threads() > 1 ? "s " : " ");
        std::cout << "active" << std::endl << std::endl;
    }
    omp_set_max_active_levels(1); // 1 disables nesting
    if (omp_get_max_active_levels() > 1)
    {
        std::cerr << "Nested parallelism not allowed" << std::endl;
        return 1;
    }
#endif
    // setup standard output and error
    MPIdata::sout = &std::cout;
    MPIdata::serr = &std::cerr;

#ifdef USE_CNR
    /* use conditional numerical reproducibility for MKL */
    int my_cbwr_branch = mkl_cbwr_get_auto_branch();
    if (mkl_cbwr_set(my_cbwr_branch) != MKL_CBWR_SUCCESS)
    {
        printf("Error in setting MKL_CBWR_BRANCH! Aborting…\n");
        return (-1);
    }
#endif

    return 0;
}

int mgmol_check()
{
    Control& ct              = *(Control::instance());
    Mesh* mymesh             = Mesh::instance();
    const pb::PEenv& myPEenv = mymesh->peenv();

    if (myPEenv.color() > 0)
    {
        std::cerr << "Code should be called with " << myPEenv.n_mpi_tasks()
                  << " MPI tasks only" << std::endl;
        ct.global_exit();
    }

    assert(ct.getMGlevels() >= -1);
    if (ct.withPreconditioner())
    {
        const pb::Grid& mygrid = mymesh->grid();

        if ((mygrid.dim(0) % (1 << (ct.getMGlevels() - 1))) != 0)
        {
            std::cerr << "main: mygrid.dim(0)=" << mygrid.dim(0)
                      << " not evenly divisible by "
                      << "2^(ct.getMGlevels()-1)="
                      << (1 << (ct.getMGlevels() - 1)) << std::endl;
            return -1;
        }
        if ((mygrid.dim(1) % (1 << (ct.getMGlevels() - 1))) != 0)
        {
            std::cerr << "main: mygrid.dim(1)=" << mygrid.dim(1)
                      << " not evenly divisible by "
                      << "2^(ct.getMGlevels()-1)="
                      << (1 << (ct.getMGlevels() - 1)) << std::endl;
            return -1;
        }
        if ((mygrid.dim(2) % (1 << (ct.getMGlevels() - 1))) != 0)
        {
            std::cerr << "main: mygrid.dim(2)=" << mygrid.dim(2)
                      << " not evenly divisible by "
                      << "2^(ct.getMGlevels()-1)="
                      << (1 << (ct.getMGlevels() - 1)) << std::endl;
            return -1;
        }
    }

    return 0;
}

void mgmol_finalize()
{
    Control& ct = *(Control::instance());

    if (!ct.short_sighted)
    {
#ifdef MGMOL_USE_SCALAPACK
        // need to destroy any MPI based object befor calling MPI_Finalize
        MatricesBlacsContext::instance().clear();
#endif
    }

    // release memory for static arrays
    PackedCommunicationBuffer::deleteStorage();
    Mesh::deleteInstance();
    Control::deleteInstance();
    MGmol_MPI::deleteInstance();

#ifdef HAVE_MAGMA
    // Delete the data in the singleton before finalizing magma
    auto& magma_singleton = MagmaSingleton::get_magma_singleton();
    magma_singleton.free();

    magma_int_t magmalog = magma_finalize();

    if (magmalog == MAGMA_SUCCESS)
    {
        std::cout << "MAGMA Finalize: success" << std::endl;
    }
    else
    {
        if (magmalog == MAGMA_ERR_UNKNOWN)
            std::cout << "MAGMA Finalize: unknown error" << std::endl;
        if (magmalog == MAGMA_ERR_HOST_ALLOC)
            std::cout << "MAGMA FINALIZE: fails host alloc" << std::endl;
    }

#endif
}

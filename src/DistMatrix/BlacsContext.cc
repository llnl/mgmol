// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE
#include "BlacsContext.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

#ifdef MGMOL_USE_SCALAPACK
#include "blacs.h"
#endif

#include "MGmol_abort.h"

#ifndef MGMOL_USE_SCALAPACK

void Cblacs_pinfo(int* mypnum, int* nprocs)
{
    *mypnum = 0;
    *nprocs = 1;

    return;
}

void Cblacs_get(int icontxt, int what, int* val)
{
    *val = 0;
    return;
}

void Cblacs_gridinit(int* icontxt, char order[], int nprow, int npcol)
{
    *icontxt = 0;
    return;
}

void Cblacs_gridmap(int* icontxt, int* pmap, int ldpmap, int nprow, int npcol)
{
    pmap[0] = 0;

    return;
}

void Cblacs_abort(int icontxt, int errornum) { return; }

void Cblacs_gridexit(int icontxt) { return; }

void Cblacs_exit(int doneflag) { return; }

void Cblacs_gridinfo(
    int icontxt, int* nprow, int* npcol, int* myprow, int* mypcol)
{
    *nprow  = 1;
    *npcol  = 1;
    *myprow = 0;
    *mypcol = 0;
    return;
}

int Cblacs_pnum(int icontxt, int prow, int pcol) { return 0; }

void Cigesd2d(int icontxt, int m, int n, int* A, int lda, int rdest, int cdest)
{
    return;
}

void Cigerv2d(int icontxt, int m, int n, int* A, int lda, int rdest, int cdest)
{
    return;
}

void Cdgerv2d(
    int icontxt, int m, int n, double* A, int lda, int rdest, int cdest)
{
    return;
}

void Cdgesd2d(
    int icontxt, int m, int n, double* A, int lda, int rdest, int cdest)
{
    return;
}

void Cigamn2d(int icontxt, char scope[], char top[], int m, int n, int* A,
    int lda, int* ra, int* ca, int rcflag, int rdest, int cdest)
{
    return;
}

void Cdgsum2d(int icontxt, int m, int n, double* A, int lda) { return; }

#endif

namespace dist_matrix
{

////////////////////////////////////////////////////////////////////////////////
void BlacsContext::buildCommunicator()
{
    int* ranks = new int[size_];
    for (int i = 0; i < size_; i++)
        ranks[i] = i;
    MPI_Group group_world, subgroup;
    int status = MPI_Comm_group(comm_global_, &group_world);
    if (status != MPI_SUCCESS)
    {
        std::cerr
            << "BlacsContext::buildCommunicator() --- Error in MPI_Comm_group"
            << std::endl;
        std::cerr << "size_=" << size_ << std::endl;
        std::cerr << "comm_global_=" << comm_global_ << std::endl;
        mgmol_abort("BlacsContext::buildCommunicator");
    }
    status = MPI_Group_incl(group_world, size_, ranks, &subgroup);
    if (status != MPI_SUCCESS)
    {
        mgmol_abort(
            "BlacsContext::buildCommunicator() --- Error in MPI_Group_incl");
    }
    MPI_Comm_create(comm_global_, subgroup, &comm_active_);
    delete[] ranks;
}

////////////////////////////////////////////////////////////////////////////////
BlacsContext::BlacsContext(MPI_Comm comm_global)
    : ictxt_(-1), myrow_(-1), mycol_(-1), comm_global_(comm_global)
{
    assert(comm_global != MPI_COMM_NULL);

    // construct a single row global context
    char order = 'R';
    MPI_Comm_size(comm_global, &nprocs_);
    MPI_Comm_rank(comm_global, &mype_);

    size_  = nprocs_;
    nprow_ = 1;
    npcol_ = nprocs_;

    ictxt_ = Csys2blacs_handle(comm_global_);
    Cblacs_gridinit(&ictxt_, &order, nprow_, npcol_);

    setup();

    // build MPI communicator for this context
    buildCommunicator();
}

////////////////////////////////////////////////////////////////////////////////
BlacsContext::BlacsContext(
    MPI_Comm comm_global, const char type, const int max_cpus)
    : ictxt_(-1), myrow_(-1), mycol_(-1), comm_global_(comm_global)
{
    // specialized global contexts
    // single row if type == 'r'
    // single column if type == 'c'
    // largest square if type == 's'

    char order = 'R';
    MPI_Comm_size(comm_global, &nprocs_);
    MPI_Comm_rank(comm_global, &mype_);

    size_ = std::min(nprocs_, max_cpus);

    if (type == 'r')
    {
        // single row
        nprow_ = 1;
        npcol_ = size_;
    }
    else if (type == 'c')
    {
        // single column
        nprow_ = size_;
        npcol_ = 1;
    }
    else if (type == 's')
    {
        // largest possible square context
        int sqrt_nprocs = (int)sqrt((double)size_);
        assert(nprocs_ >= sqrt_nprocs * sqrt_nprocs);
        nprow_ = npcol_ = sqrt_nprocs;
    }
    else
    {
        std::cerr << " BlacsContext::BlacsContext: type = " << type
                  << " is an incorrect parameter" << std::endl;
        MPI_Abort(comm_global, EXIT_FAILURE);
    }

    size_ = nprow_ * npcol_;

    ictxt_ = Csys2blacs_handle(comm_global);
    Cblacs_gridinit(&ictxt_, &order, nprow_, npcol_);

    setup();

    // build MPI communicator for this context
    buildCommunicator();
}

////////////////////////////////////////////////////////////////////////////////
BlacsContext::BlacsContext(
    MPI_Comm comm_global, const int nprow, const int npcol)
    : ictxt_(-1),
      myrow_(-1),
      mycol_(-1),
      nprow_(nprow),
      npcol_(npcol),
      comm_global_(comm_global)
{
    char order = 'R';
    MPI_Comm_size(comm_global_, &nprocs_);
    MPI_Comm_rank(comm_global_, &mype_);
    if (nprocs_ < nprow * npcol)
    {
        std::cerr << " nprocs_=" << nprocs_ << std::endl;
        std::cerr << " BlacsContext nprow*npcol > nprocs_" << std::endl;
        MPI_Abort(comm_global, EXIT_FAILURE);
    }

    ictxt_ = Csys2blacs_handle(comm_global_);
    Cblacs_gridinit(&ictxt_, &order, nprow, npcol);

    setup();
    size_ = nprow_ * npcol_;

    // build MPI communicator for this context
    buildCommunicator();
}

////////////////////////////////////////////////////////////////////////////////
BlacsContext::BlacsContext(
    MPI_Comm comm_global, const int ipe, const int nprow, const int npcol)
    : ictxt_(-1),
      myrow_(-1),
      mycol_(-1),
      nprow_(nprow),
      npcol_(npcol),
      comm_global_(comm_global)
{
    MPI_Comm_size(comm_global, &nprocs_);
    MPI_Comm_rank(comm_global, &mype_);

    if (ipe < 0 || nprow <= 0 || npcol <= 0 || ipe + nprow * npcol > nprocs_)
    {
        std::cerr << " BlacsContext::BlacsContext: invalid parameters"
                  << " in " << __FILE__ << ":" << __LINE__ << std::endl;
        MPI_Abort(comm_global, EXIT_FAILURE);
    }
    int* pmap = new int[nprow * npcol];
    // build pmap
    for (int i = 0; i < nprow * npcol; i++)
        pmap[i] = ipe + i;

    ictxt_ = Csys2blacs_handle(comm_global_);
    Cblacs_gridmap(&ictxt_, pmap, nprow, nprow, npcol);
    delete[] pmap;

    setup();
    size_ = nprow_ * npcol_;

    // build MPI communicator for this context
    int* ranks = new int[size_];
    for (int i = 0; i < size_; i++)
        ranks[i] = ipe + i;
    MPI_Group group_world, subgroup;
    MPI_Comm_group(comm_global, &group_world);
    MPI_Group_incl(group_world, size_, ranks, &subgroup);
    MPI_Comm_create(comm_global, subgroup, &comm_active_);
    delete[] ranks;
}

////////////////////////////////////////////////////////////////////////////////
BlacsContext::BlacsContext(BlacsContext& bc, const int irow, const int icol,
    const int nr, const int nc, const int nprow, const int npcol)
    : ictxt_(-1),
      myrow_(-1),
      mycol_(-1),
      nprow_(nprow),
      npcol_(npcol),
      comm_global_(bc.comm_global_)
{
    mype_ = bc.mype_;

    // construct a (nprow*npcol) context using the processes in bc
    // located in the rectangle (nr*nc) anchored at (irow,icol)
    if (irow < 0 || icol < 0 || nr <= 0 || nc <= 0 || irow + nr > bc.nprow()
        || icol + nc > bc.npcol() || nr * nc != nprow * npcol)
    {
        std::cerr << " BlacsContext::BlacsContext: invalid parameters"
                  << std::endl;
        MPI_Abort(comm_global_, EXIT_FAILURE);
    }
    int* pmap = new int[nprow * npcol];
    // build pmap
    int i = 0;
    for (int ir = 0; ir < nr; ir++)
        for (int ic = 0; ic < nc; ic++)
            pmap[i++] = Cblacs_pnum(bc.ictxt(), irow + ir, icol + ic);

    ictxt_ = Csys2blacs_handle(comm_global_);
    Cblacs_gridmap(&ictxt_, pmap, nprow, nprow, npcol);

    setup();
    size_ = nprow_ * npcol_;

    MPI_Group bc_group, subgroup;
    MPI_Comm_group(bc.comm_active(), &bc_group);
    MPI_Group_incl(bc_group, size_, pmap, &subgroup);
    MPI_Comm_create(bc.comm_active(), subgroup, &comm_active_);
    delete[] pmap;
}

////////////////////////////////////////////////////////////////////////////////
BlacsContext::BlacsContext(const BlacsContext& bc, const char type)
    : ictxt_(-1), myrow_(-1), mycol_(-1), comm_global_(bc.comm_global_)
{
    mype_     = bc.mype_;
    int* pmap = nullptr;
    if (type == 'c')
    {
        // make context consisting of my column in context bc
        nprow_   = bc.nprow();
        npcol_   = 1;
        int icol = bc.mycol();
        pmap     = new int[nprow_];
        for (int ir = 0; ir < nprow_; ir++)
            pmap[ir] = Cblacs_pnum(bc.ictxt(), ir, icol);
    }
    else if (type == 'r')
    {
        // make context consisting of my row in context bc
        nprow_   = 1;
        npcol_   = bc.npcol();
        int irow = bc.myrow();
        pmap     = new int[npcol_];
        // build pmap
        for (int ic = 0; ic < npcol_; ic++)
            pmap[ic] = Cblacs_pnum(bc.ictxt(), irow, ic);
    }
    else
    {
        std::cerr
            << " BlacsContext::BlacsContext: row/col incorrect parameter: "
            << type << std::endl;
        MPI_Abort(comm_global_, EXIT_FAILURE);
    }

    ictxt_ = Csys2blacs_handle(comm_global_);
    Cblacs_gridmap(&ictxt_, pmap, nprow_, nprow_, npcol_);

    setup();
    size_ = nprow_ * npcol_;

    MPI_Group bc_group, subgroup;
    MPI_Comm_group(bc.comm_active(), &bc_group);
    MPI_Group_incl(bc_group, size_, pmap, &subgroup);
    MPI_Comm_create(bc.comm_active(), subgroup, &comm_active_);
    delete[] pmap;
}

////////////////////////////////////////////////////////////////////////////////
BlacsContext::~BlacsContext()
{
    if (myrow_ != -1) Cblacs_gridexit(ictxt_);
    if (active_)
    {
        assert(comm_active_ != MPI_COMM_NULL);
        MPI_Comm_free(&comm_active_);
    }
}

////////////////////////////////////////////////////////////////////////////////
void BlacsContext::setup()
{
    // get values of nprow_, npcol_, myrow_ and mycol_ in the new context
    if (ictxt_ >= 0)
        Cblacs_gridinfo(ictxt_, &nprow_, &npcol_, &myrow_, &mycol_);

    myproc_ = myrow_ < 0 ? -1 : mycol_ + npcol_ * myrow_;
    onpe0_  = (mype_ == 0);
    active_ = (ictxt_ >= 0);
}

} // namespace

// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "GridFuncVector.h"
#include "MGkernels.h"
#include "MGmol_blas1.h"
#include "MPIdata.h"
#include "mputils.h"

#include <cassert>
#include <set>

namespace pb
{
template <typename ScalarType, typename MemorySpaceType>
void GridFuncVector<ScalarType, MemorySpaceType>::allocate(const int n)
{
    functions_.resize(n);

    // allocate memory on host
    int alloc_size = grid_.sizeg();
    memory_.reset(new ScalarType[n * alloc_size]);

    memset(memory_.get(), 0, n * alloc_size * sizeof(ScalarType));

    // pin GridFunc to host memory
    for (int i = 0; i < n; i++)
    {
        ScalarType* alloc = memory_.get() + i * alloc_size;
        functions_[i]
            = new GridFunc<ScalarType>(grid_, bc_[0], bc_[1], bc_[2], alloc);
    }

    // jlf, 8/6/2020: one should be able to set this flag to true
    // but we may need to fix a few things for that to work
    updated_boundaries_ = false;
}

template <typename ScalarType, typename MemorySpaceType>
void GridFuncVector<ScalarType, MemorySpaceType>::setup()
{
    const int mytask = grid_.mype_env().mytask();
    north_ = (bc_[1] == 1 || (grid_.mype_env().mpi_neighbors(NORTH) > mytask));
    south_ = (bc_[1] == 1 || (grid_.mype_env().mpi_neighbors(SOUTH) < mytask));
    up_    = (bc_[2] == 1 || (grid_.mype_env().mpi_neighbors(UP) > mytask));
    down_  = (bc_[2] == 1 || (grid_.mype_env().mpi_neighbors(DOWN) < mytask));
    east_  = (bc_[0] == 1 || (grid_.mype_env().mpi_neighbors(EAST) > mytask));
    west_  = (bc_[0] == 1 || (grid_.mype_env().mpi_neighbors(WEST) < mytask));

    nfunc_ = (int)gid_[0].size();
    MPI_Allreduce(&nfunc_, &nfunc_max_global_, 1, MPI_INT, MPI_MAX, comm_);
    nsubdivx_ = (short)gid_.size();

    remote_gids_.resize(6);

    for (short iloc = 0; iloc < nsubdivx_; iloc++)
        for (short k = 0; k < nfunc_; k++)
            if (gid_[iloc][k] >= 0)
                gid2lid_.insert(std::pair<int, short>(gid_[iloc][k], k));

    nghosts_ = grid_.ghost_pt();

    dimx_ = grid_.dim(0);
    dimy_ = grid_.dim(1);
    dimz_ = grid_.dim(2);

    incx_ = grid_.inc(0);
    incy_ = grid_.inc(1);

    dimxy_ = (dimy_ + 2 * nghosts_) * dimx_;
    incxy_ = dimxy_ / nsubdivx_;

    east_west_size_   = nghosts_ * incx_;
    south_north_size_ = nghosts_ * dimz_ * dimx_ + nsubdivx_;
    up_down_size_     = nghosts_ * dimxy_ + nsubdivx_;

    assert(dimx_ >= nghosts_);
    assert(dimy_ >= nghosts_);
    assert(dimz_ >= nghosts_);
}

template <typename ScalarType, typename MemorySpaceType>
template <typename ScalarType2>
void GridFuncVector<ScalarType, MemorySpaceType>::pointwiseProduct(
    GridFuncVector<ScalarType, MemorySpaceType>& A,
    const GridFunc<ScalarType2>& B)
{
    assert(A.grid_.sizeg() == grid_.sizeg());
    assert(B.grid().sizeg() == grid_.sizeg());
    assert(A.size() == size());

    prod_tm_.start();

    const int ngpts = grid_.sizeg();

    ScalarType2* b_view = B.uu();

    const int bsize = 64;
    const int nb    = ngpts / bsize;
    const int nf    = (int)size();

    // loop over blocks (subdomains) to reuse B data
    // for several columns of this and A
    for (int ib = 0; ib <= nb; ib++)
    {
        const int ibstart                  = ib * bsize;
        const ScalarType2* __restrict__ v2 = b_view + ibstart;
        const int npt = (ib < nb) ? bsize : ngpts - ibstart;
        assert(npt >= 0);

        if (npt > 0)
            for (int j = 0; j < nf; j++)
            {
                ScalarType* __restrict__ pu       = getDataPtr(j, ibstart);
                const ScalarType* __restrict__ v1 = A.getDataPtr(j, ibstart);
                for (int i = 0; i < npt; i++)
                {
                    pu[i] = (ScalarType)(v1[i] * v2[i]);
                }
            }
    }

    updated_boundaries_ = A.updated_boundaries_ && B.updated_boundaries();
    for (int j = 0; j < nf; j++)
    {
        functions_[j]->set_updated_boundaries(
            A.functions_[j]->updated_boundaries() && B.updated_boundaries());
    }
    prod_tm_.stop();
}

template <typename ScalarType, typename MemorySpaceType>
void GridFuncVector<ScalarType, MemorySpaceType>::wait_north_south()
{
    if (grid_.mype_env().n_mpi_task(1) == 1) return;

    wait_north_south_tm_.start();

    if (south_)
    {
        MPI_Wait(req_north_south_ + 2, MPI_STATUS_IGNORE);
        MPI_Wait(req_north_south_ + 1, MPI_STATUS_IGNORE);
    }
    if (north_)
    {
        MPI_Wait(req_north_south_, MPI_STATUS_IGNORE);
        MPI_Wait(req_north_south_ + 3, MPI_STATUS_IGNORE);
    }

    wait_north_south_tm_.stop();
}
template <typename ScalarType, typename MemorySpaceType>
void GridFuncVector<ScalarType, MemorySpaceType>::wait_east_west()
{
    if (grid_.mype_env().n_mpi_task(0) == 1) return;

    wait_east_west_tm_.start();

    if (west_)
    {
        MPI_Wait(req_east_west_ + 1, MPI_STATUS_IGNORE);
        MPI_Wait(req_east_west_ + 2, MPI_STATUS_IGNORE);
    }
    if (east_)
    {
        MPI_Wait(req_east_west_, MPI_STATUS_IGNORE);
        MPI_Wait(req_east_west_ + 3, MPI_STATUS_IGNORE);
    }

    wait_east_west_tm_.stop();
}
template <typename ScalarType, typename MemorySpaceType>
void GridFuncVector<ScalarType, MemorySpaceType>::wait_up_down()
{
    if (grid_.mype_env().n_mpi_task(2) == 1) return;

    wait_up_down_tm_.start();

    if (up_)
    {
        MPI_Wait(req_up_down_, MPI_STATUS_IGNORE);
        MPI_Wait(req_up_down_ + 3, MPI_STATUS_IGNORE);
    }
    if (down_)
    {
        MPI_Wait(req_up_down_ + 2, MPI_STATUS_IGNORE);
        MPI_Wait(req_up_down_ + 1, MPI_STATUS_IGNORE);
    }

    wait_up_down_tm_.stop();
}

template <typename ScalarType, typename MemorySpaceType>
void GridFuncVector<ScalarType, MemorySpaceType>::allocate_buffers(
    const int nfunc)
{
    if (nfunc <= nfunc4buffers_) return;

    if (grid_.mype_env().n_mpi_tasks() > 1)
    {
        int size_max = nfunc * nghosts_ * incx_;
        size_max     = std::max(size_max, nfunc * nghosts_ * dimxy_);
        size_max     = std::max(size_max, nfunc * nghosts_ * dimx_ * incy_);
        size_max += nfunc * nsubdivx_; // to pack gids
        size_max += 1; // to pack number of functions (data) in buffer

        comm_buf1_.resize(3);
        comm_buf2_.resize(3);
        comm_buf3_.resize(3);
        comm_buf4_.resize(3);

        for (short i = 0; i < 3; i++)
        {
            comm_buf1_[i].resize(size_max);
            comm_buf2_[i].resize(size_max);
            comm_buf3_[i].resize(size_max);
            comm_buf4_[i].resize(size_max);
        }

        nfunc4buffers_ = nfunc;
    }
}
template <typename ScalarType, typename MemorySpaceType>
template <typename MST,
    typename std::enable_if<std::is_same<MemorySpace::Host, MST>::value>::type*>
void GridFuncVector<ScalarType, MemorySpaceType>::initiateNorthSouthComm(
    const int begin_color, const int end_color)
{
    std::vector<ScalarType>& comm_buf1(comm_buf1_[1]);
    std::vector<ScalarType>& comm_buf2(comm_buf2_[1]);
    std::vector<ScalarType>& comm_buf3(comm_buf3_[1]);
    std::vector<ScalarType>& comm_buf4(comm_buf4_[1]);

    const int ncolors = end_color - begin_color;

    const size_t sizebuffer = 1 + nfunc_max_global_ * south_north_size_;
    const size_t sdimz      = dimz_ * sizeof(ScalarType);
    if (north_)
        grid_.mype_env().Irecv(
            &comm_buf4[0], sizebuffer, NORTH, &req_north_south_[3]);
    if (south_)
        grid_.mype_env().Irecv(
            &comm_buf3[0], sizebuffer, SOUTH, &req_north_south_[1]);

    const int imin = nghosts_ * incx_;
    const int iinc = incx_ * dimx_ / nsubdivx_;

    const int jmax = nghosts_ * incy_;
    const int ymax = dimy_ * grid_.inc(1);

    if (south_)
    {
        ScalarType* buf2_ptr = &comm_buf2[0];
        // first element will tell how many functions (data) are in buffer
        *buf2_ptr = (ScalarType)ncolors;
        buf2_ptr++;

        // pack data
        for (int color = begin_color; color < end_color; color++)
        {
            for (short iloc = 0; iloc < nsubdivx_; iloc++)
            {
                const ScalarType* uus
                    = getDataPtr(color, nghosts_ * (incy_ + 1));
                *buf2_ptr = (ScalarType)gid_[iloc][color];
                buf2_ptr++;
                for (int j = 0; j < jmax; j += incy_)
                    for (int i = imin + iloc * iinc;
                         i < imin + (iloc + 1) * iinc; i += incx_)
                    {
                        memcpy(buf2_ptr, &uus[i + j], sdimz);
                        buf2_ptr += dimz_;
                    }
            }
        }

        grid_.mype_env().Isend(&comm_buf2[0], 1 + ncolors * south_north_size_,
            SOUTH, &req_north_south_[2]);
    }

    if (north_)
    {
        ScalarType* buf1_ptr = &comm_buf1[0];
        // first element will tell how many functions (data) are in buffer
        *buf1_ptr = (ScalarType)ncolors;
        buf1_ptr++;

        // pack data
        for (int color = begin_color; color < end_color; color++)
        {
            for (short iloc = 0; iloc < nsubdivx_; iloc++)
            {
                const ScalarType* uus = getDataPtr(color, nghosts_ + ymax);
                *buf1_ptr             = (ScalarType)gid_[iloc][color];
                buf1_ptr++;
                for (int j = 0; j < jmax; j += incy_)
                    for (int i = imin + iloc * iinc;
                         i < imin + (iloc + 1) * iinc; i += incx_)
                    {
                        memcpy(buf1_ptr, &uus[i + j], sdimz);
                        buf1_ptr += dimz_;
                    }
            }
        }

        grid_.mype_env().Isend(&comm_buf1[0], 1 + ncolors * south_north_size_,
            NORTH, &req_north_south_[0]);
    }
}
template <typename ScalarType, typename MemorySpaceType>
template <typename MST, typename std::enable_if<std::is_same<
                            MemorySpace::Device, MST>::value>::type*>
void GridFuncVector<ScalarType, MemorySpaceType>::initiateNorthSouthComm(
    const int begin_color, const int end_color)
{
    std::vector<ScalarType>& comm_buf1(comm_buf1_[1]);
    std::vector<ScalarType>& comm_buf2(comm_buf2_[1]);
    std::vector<ScalarType>& comm_buf3(comm_buf3_[1]);
    std::vector<ScalarType>& comm_buf4(comm_buf4_[1]);

    const int ncolors = end_color - begin_color;

    const size_t sizebuffer = 1 + nfunc_max_global_ * south_north_size_;
    if (north_)
        grid_.mype_env().Irecv(
            &comm_buf4[0], sizebuffer, NORTH, &req_north_south_[3]);
    if (south_)
        grid_.mype_env().Irecv(
            &comm_buf3[0], sizebuffer, SOUTH, &req_north_south_[1]);

    const int imin = nghosts_ * incx_;
    const int iinc = incx_ * dimx_ / nsubdivx_;

    const int jmax = nghosts_ * incy_;
    const int ymax = dimy_ * grid_.inc(1);

    auto nghosts           = nghosts_;
    auto incx              = incx_;
    auto incy              = incy_;
    auto dimz              = dimz_;
    auto size_per_function = grid_.sizeg();
    auto south_north_size  = south_north_size_;

    ScalarType* functions_alias = memory_dev_.get();

    if (south_)
    {
        ScalarType* buf2_ptr = &comm_buf2[0];
        // first element will tell how many functions (data) are in buffer
        *buf2_ptr = (ScalarType)ncolors;
        buf2_ptr++;

        std::unique_ptr<ScalarType, void (*)(ScalarType*)> buf2_ptr_dev(
            MemoryST::allocate(sizebuffer - 1), MemoryST::free);

        ScalarType* buf2_alias = buf2_ptr_dev.get();

        MemorySpace::assert_is_dev_ptr(buf2_alias);

        MGMOL_PARALLEL_FOR_COLLAPSE(2, buf2_alias, functions_alias)
        for (int color = begin_color; color < end_color; color++)
        {
            for (int j = 0; j < jmax; j += incy)
            {
                for (int i = imin; i < imin + iinc; i += incx)
                {
                    for (int k = 0; k < dimz; k++)
                    {
                        const ScalarType* __restrict__ uus
                            = functions_alias + color * size_per_function
                              + nghosts * (incy + 1);
                        size_t index_buf2 = color * south_north_size + 1
                                            + j / incy * iinc / incx * dimz
                                            + (i - imin) / incx * dimz + k;

                        buf2_alias[index_buf2] = uus[i + j + k];
                    }
                }
            }
        }

        MemorySpace::copy_to_host(buf2_alias, sizebuffer - 1, buf2_ptr);

        grid_.mype_env().Isend(&comm_buf2[0], 1 + ncolors * south_north_size_,
            SOUTH, &req_north_south_[2]);
    }

    if (north_)
    {
        ScalarType* buf1_ptr = &comm_buf1[0];
        // first element will tell how many functions (data) are in buffer
        *buf1_ptr = (ScalarType)ncolors;
        buf1_ptr++;

        std::unique_ptr<ScalarType, void (*)(ScalarType*)> buf1_ptr_dev(
            MemoryST::allocate(sizebuffer - 1), MemoryST::free);

        ScalarType* buf1_alias = buf1_ptr_dev.get();

        MemorySpace::assert_is_dev_ptr(buf1_alias);

        MGMOL_PARALLEL_FOR_COLLAPSE(2, buf1_alias, functions_alias)
        for (int color = begin_color; color < end_color; color++)
        {
            for (int j = 0; j < jmax; j += incy)
            {
                for (int i = imin; i < imin + iinc; i += incx)
                {
                    for (int k = 0; k < dimz; k++)
                    {
                        const ScalarType* __restrict__ uus
                            = functions_alias + color * size_per_function
                              + nghosts + ymax;
                        size_t index_buf1 = color * south_north_size + 1
                                            + j / incy * iinc / incx * dimz
                                            + (i - imin) / incx * dimz + k;
                        buf1_alias[index_buf1] = uus[i + j + k];
                    }
                }
            }
        }

        MemorySpace::copy_to_host(buf1_alias, sizebuffer - 1, buf1_ptr);

        grid_.mype_env().Isend(&comm_buf1[0], 1 + ncolors * south_north_size_,
            NORTH, &req_north_south_[0]);
    }
}

template <typename ScalarType, typename MemorySpaceType>
template <typename MST,
    typename std::enable_if<std::is_same<MemorySpace::Host, MST>::value>::type*>
void GridFuncVector<ScalarType, MemorySpaceType>::finishNorthSouthComm()
{
    finishExchangeNorthSouth_tm_.start();

    std::vector<ScalarType>& comm_buf3(comm_buf3_[1]);
    std::vector<ScalarType>& comm_buf4(comm_buf4_[1]);

    const int ymax     = dimy_ * grid_.inc(1);
    const size_t sdimz = dimz_ * sizeof(ScalarType);

    if (grid_.mype_env().n_mpi_task(1) > 1)
    {
        const int imin = nghosts_ * incx_;
        const int iinc = incx_ * dimx_ / nsubdivx_;
        const int jmax = nghosts_ * incy_;

        if (north_)
        {
            ScalarType* buf4_ptr   = &comm_buf4[0];
            const int nremote_func = (int)(*buf4_ptr);
            buf4_ptr++;

            for (int k = 0; k < nremote_func; k++)
            {
                for (short iloc = 0; iloc < nsubdivx_; iloc++)
                {
                    const int gid = (int)(*buf4_ptr);
                    buf4_ptr++;

                    if (gid >= 0)
                    {
                        std::map<int, short>::const_iterator ilid
                            = gid2lid_.find(gid);
                        if (ilid != gid2lid_.end())
                        {
                            const short lid = ilid->second;
                            if (gid == gid_[iloc][lid])
                            {
                                ScalarType* uus = getDataPtr(
                                    lid, nghosts_ * (incy_ + 1) + ymax);
                                for (int j = 0; j < jmax; j += incy_)
                                    for (int i = imin + iloc * iinc;
                                         i < imin + (iloc + 1) * iinc;
                                         i += incx_)
                                    {
                                        memcpy(&uus[i + j], buf4_ptr, sdimz);
                                        buf4_ptr += dimz_;
                                    }
                            }
                            else
                            {
                                for (int j = 0; j < jmax; j += incy_)
                                    for (int i = imin + iloc * iinc;
                                         i < imin + (iloc + 1) * iinc;
                                         i += incx_)
                                        buf4_ptr += dimz_;
                            }
                        }
                        else
                        {
                            for (int j = 0; j < jmax; j += incy_)
                                for (int i = imin + iloc * iinc;
                                     i < imin + (iloc + 1) * iinc; i += incx_)
                                    buf4_ptr += dimz_;
                        }
                    }
                    else
                    {
                        for (int j = 0; j < jmax; j += incy_)
                            for (int i = imin + iloc * iinc;
                                 i < imin + (iloc + 1) * iinc; i += incx_)
                            {
                                buf4_ptr += dimz_;
                            }
                    }
                }
            }

            assert((buf4_ptr - &comm_buf4[0])
                   <= static_cast<int>(comm_buf4.size()));
        }

        if (south_)
        {
            ScalarType* buf3_ptr   = &comm_buf3[0];
            const int nremote_func = (int)(*buf3_ptr);
            buf3_ptr++;

            for (int k = 0; k < nremote_func; k++)
            {
                for (short iloc = 0; iloc < nsubdivx_; iloc++)
                {
                    const int gid = (int)(*buf3_ptr);
                    buf3_ptr++;
                    if (gid >= 0)
                    {
                        std::map<int, short>::const_iterator ilid
                            = gid2lid_.find(gid);
                        if (ilid != gid2lid_.end())
                        {
                            const short lid = ilid->second;
                            if (gid == gid_[iloc][lid])
                            {
                                ScalarType* uus = getDataPtr(lid, nghosts_);
                                for (int j = 0; j < jmax; j += incy_)
                                    for (int i = imin + iloc * iinc;
                                         i < imin + (iloc + 1) * iinc;
                                         i += incx_)
                                    {
                                        memcpy(&uus[i + j], buf3_ptr, sdimz);
                                        buf3_ptr += dimz_;
                                    }
                            }
                            else
                            {
                                for (int j = 0; j < jmax; j += incy_)
                                    for (int i = imin + iloc * iinc;
                                         i < imin + (iloc + 1) * iinc;
                                         i += incx_)
                                        buf3_ptr += dimz_;
                            }
                        }
                        else
                        {
                            for (int j = 0; j < jmax; j += incy_)
                                for (int i = imin + iloc * iinc;
                                     i < imin + (iloc + 1) * iinc; i += incx_)
                                    buf3_ptr += dimz_;
                        }
                    }
                    else
                    {
                        for (int j = 0; j < jmax; j += incy_)
                            for (int i = imin + iloc * iinc;
                                 i < imin + (iloc + 1) * iinc; i += incx_)
                            {
                                buf3_ptr += dimz_;
                            }
                    }
                }
            }

            assert((buf3_ptr - &comm_buf3[0])
                   <= static_cast<int>(comm_buf3.size()));
        }
    }
    else
    { // grid_.mype_env().n_mpi_task(1)==1
        if (bc_[1] == 1)
        {
            // only for i already initialized
            for (int k = 0; k < nfunc_; k++)
            {
                ScalarType* pu = getDataPtr(k);
                for (int j = 0; j < nghosts_ * incy_; j += incy_)
                    for (int i = nghosts_ * incx_;
                         i < (dimx_ + nghosts_) * incx_; i += incx_)
                    {
                        memcpy(&pu[i + nghosts_ + j],
                            &pu[i + nghosts_ + j + ymax], sdimz);
                        memcpy(&pu[i + nghosts_ * (incy_ + 1) + j + ymax],
                            &pu[i + nghosts_ * (incy_ + 1) + j], sdimz);
                    }
            }
        }
    }

    finishExchangeNorthSouth_tm_.stop();
}

template <typename ScalarType, typename MemorySpaceType>
template <typename MST, typename std::enable_if<std::is_same<
                            MemorySpace::Device, MST>::value>::type*>
void GridFuncVector<ScalarType, MemorySpaceType>::finishNorthSouthComm()
{
    finishExchangeNorthSouth_tm_.start();

    std::vector<ScalarType>& comm_buf3(comm_buf3_[1]);
    std::vector<ScalarType>& comm_buf4(comm_buf4_[1]);

    auto sizebuffer = comm_buf3.size();

    auto nfunc             = nfunc_;
    auto incy              = incy_;
    auto incx              = incx_;
    auto dimx              = dimx_;
    auto dimz              = dimz_;
    auto nghosts           = nghosts_;
    auto size_per_function = grid_.sizeg();
    auto south_north_size  = south_north_size_;

    ScalarType* functions_alias = memory_dev_.get();

    const int ymax = dimy_ * grid_.inc(1);

    if (grid_.mype_env().n_mpi_task(1) > 1)
    {
        const int imin = nghosts_ * incx_;
        const int iinc = incx_ * dimx_ / nsubdivx_;
        const int jmax = nghosts_ * incy_;

        if (north_)
        {
            ScalarType* buf4_ptr = &comm_buf4[0];
            buf4_ptr++;

            std::unique_ptr<ScalarType, void (*)(ScalarType*)> buf4_ptr_dev(
                MemoryST::allocate(sizebuffer - 1), MemoryST::free);

            ScalarType* buf4_alias = buf4_ptr_dev.get();

            MemorySpace::assert_is_dev_ptr(buf4_alias);

            MemorySpace::copy_to_dev(buf4_ptr, sizebuffer - 1, buf4_alias);

            MGMOL_PARALLEL_FOR_COLLAPSE(2, buf4_alias, functions_alias)
            for (int color = 0; color < nfunc; color++)
            {
                for (int j = 0; j < jmax; j += incy)
                {
                    for (int i = imin; i < imin + iinc; i += incx)
                    {
                        for (int k = 0; k < dimz; k++)
                        {
                            ScalarType* uus = functions_alias
                                              + color * size_per_function
                                              + nghosts * (incy + 1) + ymax;
                            size_t index_buf4 = color * south_north_size + 1
                                                + j / incy * iinc / incx * dimz
                                                + (i - imin) / incx * dimz + k;
                            uus[i + j + k] = buf4_alias[index_buf4];
                        }
                    }
                }
            }
            assert((buf4_ptr - &comm_buf4[0])
                   <= static_cast<int>(comm_buf4.size()));
        }

        if (south_)
        {
            ScalarType* buf3_ptr = &comm_buf3[0];
            buf3_ptr++;

            std::unique_ptr<ScalarType, void (*)(ScalarType*)> buf3_ptr_dev(
                MemoryST::allocate(sizebuffer - 1), MemoryST::free);

            ScalarType* buf3_alias = buf3_ptr_dev.get();

            MemorySpace::assert_is_dev_ptr(buf3_alias);

            MemorySpace::copy_to_dev(buf3_ptr, sizebuffer - 1, buf3_alias);

            MGMOL_PARALLEL_FOR_COLLAPSE(2, buf3_alias, functions_alias)
            for (int color = 0; color < nfunc; color++)
            {
                for (int j = 0; j < jmax; j += incy)
                {
                    for (int i = imin; i < imin + iinc; i += incx)
                    {
                        for (int k = 0; k < dimz; k++)
                        {
                            ScalarType* uus = functions_alias
                                              + color * size_per_function
                                              + nghosts;
                            size_t index_buf3 = color * south_north_size + 1
                                                + j / incy * iinc / incx * dimz
                                                + (i - imin) / incx * dimz + k;
                            uus[i + j + k] = buf3_alias[index_buf3];
                        }
                    }
                }
            }
            assert((buf3_ptr - &comm_buf3[0])
                   <= static_cast<int>(comm_buf3.size()));
        }
    }
    else
    { // grid_.mype_env().n_mpi_task(1)==1
        if (bc_[1] == 1)
        {
            MGMOL_PARALLEL_FOR_COLLAPSE(2, functions_alias)
            for (int color = 0; color < nfunc; color++)
            {
                for (int j = 0; j < nghosts * incy; j += incy)
                {
                    for (int i = nghosts * incx; i < (dimx + nghosts) * incx;
                         i += incx)
                    {
                        for (int k = 0; k < dimz; k++)
                        {
                            ScalarType* pu
                                = functions_alias + color * size_per_function;
                            size_t index1 = i + nghosts + j + k;
                            size_t index2 = i + nghosts * (incy + 1) + j + k;
                            pu[index1]    = pu[index1 + ymax];
                            pu[index2 + ymax] = pu[index2];
                        }
                    }
                }
            }
        }
    }

    finishExchangeNorthSouth_tm_.stop();
}
template <typename ScalarType, typename MemorySpaceType>
template <typename MST,
    typename std::enable_if<std::is_same<MemorySpace::Host, MST>::value>::type*>
void GridFuncVector<ScalarType, MemorySpaceType>::initiateUpDownComm(
    const int begin_color, const int end_color)
{
    std::vector<ScalarType>& comm_buf1(comm_buf1_[2]);
    std::vector<ScalarType>& comm_buf2(comm_buf2_[2]);
    std::vector<ScalarType>& comm_buf3(comm_buf3_[2]);
    std::vector<ScalarType>& comm_buf4(comm_buf4_[2]);

    const int ncolors = end_color - begin_color;

    const int sizebuffer = 1 + nfunc_max_global_ * up_down_size_;
    int iinit            = (dimy_ + 2 * nghosts_) * nghosts_;
    const int zmax       = dimz_;
    const int ione       = 1;

    if (down_)
    {
        grid_.mype_env().Irecv(
            &comm_buf3[0], sizebuffer, DOWN, &req_up_down_[1]);
    }
    if (up_)
    {
        grid_.mype_env().Irecv(&comm_buf4[0], sizebuffer, UP, &req_up_down_[3]);
    }

    if (up_)
    {
        ScalarType* buf1_ptr = &comm_buf1[0];
        // first element will tell how many functions (data) are in buffer
        *buf1_ptr = (ScalarType)ncolors;
        buf1_ptr++;

        for (int color = begin_color; color < end_color; color++)
        {
            for (short iloc = 0; iloc < nsubdivx_; iloc++)
            {
                const ScalarType* const uus
                    = getDataPtr(color, nghosts_ + incy_ * iinit);
                *buf1_ptr = (ScalarType)gid_[iloc][color];
                buf1_ptr++;
                for (int j = 0; j < nghosts_; j++)
                {
                    Tcopy(&incxy_, &uus[zmax - 1 - j + iloc * incxy_ * incy_],
                        &incy_, buf1_ptr, &ione);
                    buf1_ptr += incxy_;
                }
            }
        }

        grid_.mype_env().Isend(
            &comm_buf1[0], 1 + ncolors * up_down_size_, UP, &req_up_down_[0]);
    }
    if (down_)
    {
        ScalarType* buf2_ptr = &comm_buf2[0];
        // first element will tell how many functions (data) are in buffer
        *buf2_ptr = (ScalarType)ncolors;
        buf2_ptr++;

        for (int color = begin_color; color < end_color; color++)
        {
            for (short iloc = 0; iloc < nsubdivx_; iloc++)
            {
                const ScalarType* const uus
                    = getDataPtr(color, nghosts_ + incy_ * iinit);
                *buf2_ptr = (ScalarType)gid_[iloc][color];
                buf2_ptr++;
                for (int j = 0; j < nghosts_; j++)
                {
                    Tcopy(&incxy_, &uus[j + iloc * incxy_ * incy_], &incy_,
                        buf2_ptr, &ione);
                    buf2_ptr += incxy_;
                }
            }
        }

        grid_.mype_env().Isend(
            &comm_buf2[0], 1 + ncolors * up_down_size_, DOWN, &req_up_down_[2]);
    }
}
template <typename ScalarType, typename MemorySpaceType>
template <typename MST, typename std::enable_if<std::is_same<
                            MemorySpace::Device, MST>::value>::type*>
void GridFuncVector<ScalarType, MemorySpaceType>::initiateUpDownComm(
    const int begin_color, const int end_color)
{
    std::vector<ScalarType>& comm_buf1(comm_buf1_[2]);
    std::vector<ScalarType>& comm_buf2(comm_buf2_[2]);
    std::vector<ScalarType>& comm_buf3(comm_buf3_[2]);
    std::vector<ScalarType>& comm_buf4(comm_buf4_[2]);

    const int ncolors = end_color - begin_color;

    const int sizebuffer = 1 + nfunc_max_global_ * up_down_size_;
    int iinit            = (dimy_ + 2 * nghosts_) * nghosts_;
    const int zmax       = dimz_;

    if (down_)
    {
        grid_.mype_env().Irecv(
            &comm_buf3[0], sizebuffer, DOWN, &req_up_down_[1]);
    }
    if (up_)
    {
        grid_.mype_env().Irecv(&comm_buf4[0], sizebuffer, UP, &req_up_down_[3]);
    }

    auto nghosts           = nghosts_;
    auto incxy             = incxy_;
    auto incy              = incy_;
    auto size_per_function = grid_.sizeg();
    auto up_down_size      = up_down_size_;

    ScalarType* functions_alias = memory_dev_.get();

    if (up_)
    {
        ScalarType* buf1_ptr = &comm_buf1[0];
        // first element will tell how many functions (data) are in buffer
        *buf1_ptr = (ScalarType)ncolors;
        buf1_ptr++;

        std::unique_ptr<ScalarType, void (*)(ScalarType*)> buf1_ptr_dev(
            MemoryST::allocate(sizebuffer - 1), MemoryST::free);

        ScalarType* buf1_alias = buf1_ptr_dev.get();

        MemorySpace::assert_is_dev_ptr(buf1_alias);

        MGMOL_PARALLEL_FOR_COLLAPSE(2, buf1_alias, functions_alias)
        for (int color = begin_color; color < end_color; color++)
        {
            for (int k = 0; k < incxy; k++)
            {
                for (int j = 0; j < nghosts; j++)
                {
                    const ScalarType* __restrict__ uus
                        = functions_alias + color * size_per_function + nghosts
                          + incy * iinit;
                    size_t index_buf1
                        = color * up_down_size + 1 + j * incxy + k;
                    buf1_alias[index_buf1] = uus[zmax - 1 - j + k * incy];
                }
            }
        }

        MemorySpace::copy_to_host(buf1_alias, sizebuffer - 1, buf1_ptr);

        grid_.mype_env().Isend(
            &comm_buf1[0], 1 + ncolors * up_down_size_, UP, &req_up_down_[0]);
    }
    if (down_)
    {
        ScalarType* buf2_ptr = &comm_buf2[0];
        // first element will tell how many functions (data) are in buffer
        *buf2_ptr = (ScalarType)ncolors;
        buf2_ptr++;

        std::unique_ptr<ScalarType, void (*)(ScalarType*)> buf2_ptr_dev(
            MemoryST::allocate(sizebuffer - 1), MemoryST::free);

        ScalarType* buf2_alias = buf2_ptr_dev.get();

        MemorySpace::assert_is_dev_ptr(buf2_alias);

        MGMOL_PARALLEL_FOR_COLLAPSE(2, buf2_alias, functions_alias)
        for (int color = begin_color; color < end_color; color++)
        {
            for (int k = 0; k < incxy; k++)
            {
                for (int j = 0; j < nghosts; j++)
                {
                    const ScalarType* __restrict__ uus
                        = functions_alias + color * size_per_function + nghosts
                          + incy * iinit;
                    size_t index_buf2
                        = color * up_down_size + 1 + j * incxy + k;
                    buf2_alias[index_buf2] = uus[j + k * incy];
                }
            }
        }

        MemorySpace::copy_to_host(buf2_alias, sizebuffer - 1, buf2_ptr);

        grid_.mype_env().Isend(
            &comm_buf2[0], 1 + ncolors * up_down_size_, DOWN, &req_up_down_[2]);
    }
}
template <typename ScalarType, typename MemorySpaceType>
template <typename MST,
    typename std::enable_if<std::is_same<MemorySpace::Host, MST>::value>::type*>
void GridFuncVector<ScalarType, MemorySpaceType>::finishUpDownComm()
{
    finishExchangeUpDown_tm_.start();

    std::vector<ScalarType>& comm_buf3(comm_buf3_[2]);
    std::vector<ScalarType>& comm_buf4(comm_buf4_[2]);

    const int zmax = dimz_;

    if (grid_.mype_env().n_mpi_task(2) > 1)
    {
        int iinit      = (dimy_ + 2 * nghosts_) * nghosts_;
        const int ione = 1;
        if (down_)
        {
            ScalarType* buf3_ptr   = &comm_buf3[0];
            const int nremote_func = (int)(*buf3_ptr);
            buf3_ptr++;

            for (int k = 0; k < nremote_func; k++)
            {
                for (short iloc = 0; iloc < nsubdivx_; iloc++)
                {
                    const int gid = (int)(*buf3_ptr);
                    buf3_ptr++;
                    if (gid >= 0)
                    {
                        std::map<int, short>::const_iterator ilid
                            = gid2lid_.find(gid);
                        if (ilid != gid2lid_.end())
                        {
                            short lid = ilid->second;
                            if (gid == gid_[iloc][lid])
                            {
                                for (int j = 0; j < nghosts_; j++)
                                {
                                    ScalarType* const uus
                                        = getDataPtr(lid, nghosts_ - 1 - j);
                                    Tcopy(&incxy_, buf3_ptr, &ione,
                                        &uus[incy_ * iinit
                                             + iloc * incxy_ * incy_],
                                        &incy_);
                                    buf3_ptr += incxy_;
                                }
                            }
                            else
                            {
                                buf3_ptr += incxy_ * nghosts_;
                            }
                        }
                        else
                        {
                            buf3_ptr += incxy_ * nghosts_;
                        }
                    }
                    else
                    {
                        buf3_ptr += incxy_ * nghosts_;
                    }
                }
            }
        }
        if (up_)
        {
            ScalarType* buf4_ptr   = &comm_buf4[0];
            const int nremote_func = (int)(*buf4_ptr);
            buf4_ptr++;

            for (int k = 0; k < nremote_func; k++)
            {
                for (short iloc = 0; iloc < nsubdivx_; iloc++)
                {
                    const int gid = (int)(*buf4_ptr);
                    buf4_ptr++;
                    if (gid >= 0)
                    {
                        std::map<int, short>::const_iterator ilid
                            = gid2lid_.find(gid);
                        if (ilid != gid2lid_.end())
                        {
                            short lid = ilid->second;
                            if (gid == gid_[iloc][lid])
                            {
                                for (int j = 0; j < nghosts_; j++)
                                {
                                    ScalarType* const uus
                                        = getDataPtr(lid, nghosts_ + zmax + j);
                                    Tcopy(&incxy_, buf4_ptr, &ione,
                                        &uus[incy_ * iinit
                                             + iloc * incxy_ * incy_],
                                        &incy_);
                                    buf4_ptr += incxy_;
                                }
                            }
                            else
                            {
                                buf4_ptr += incxy_ * nghosts_;
                            }
                        }
                        else
                        {
                            buf4_ptr += incxy_ * nghosts_;
                        }
                    }
                    else
                    {
                        buf4_ptr += incxy_ * nghosts_;
                    }
                }
            }
        }
    }
    else
    {
        if (bc_[2] == 1) /* grid_.mype_env().n_mpi_task(2)==1 */
        {
            int iinit = (dimy_ + 2 * nghosts_) * nghosts_;
            for (int k = 0; k < nfunc_; k++)
            {
                ScalarType* pu = getDataPtr(k);
                for (int j = 0; j < nghosts_; j++)
                {
                    Tcopy(&dimxy_, &pu[nghosts_ - j + iinit * incy_ + zmax - 1],
                        &incy_, &pu[nghosts_ - j + iinit * incy_ - 1], &incy_);
                }
                for (int j = 0; j < nghosts_; j++)
                {
                    Tcopy(&dimxy_, &pu[nghosts_ + j + iinit * incy_], &incy_,
                        &pu[nghosts_ + j + iinit * incy_ + zmax], &incy_);
                }
            }
        }
    }

    finishExchangeUpDown_tm_.stop();
}
template <typename ScalarType, typename MemorySpaceType>
template <typename MST, typename std::enable_if<std::is_same<
                            MemorySpace::Device, MST>::value>::type*>
void GridFuncVector<ScalarType, MemorySpaceType>::finishUpDownComm()
{
    finishExchangeUpDown_tm_.start();

    std::vector<ScalarType>& comm_buf3(comm_buf3_[2]);
    std::vector<ScalarType>& comm_buf4(comm_buf4_[2]);

    auto sizebuffer = comm_buf3.size();

    int nfunc              = nfunc_;
    auto incy              = incy_;
    auto incxy             = incxy_;
    auto nghosts           = nghosts_;
    auto size_per_function = grid_.sizeg();
    auto dimxy             = dimxy_;
    auto up_down_size      = up_down_size_;

    ScalarType* functions_alias = memory_dev_.get();

    const int zmax = dimz_;

    if (grid_.mype_env().n_mpi_task(2) > 1)
    {
        int iinit = (dimy_ + 2 * nghosts_) * nghosts_;
        if (down_)
        {
            ScalarType* buf3_ptr = &comm_buf3[0];
            buf3_ptr++;

            std::unique_ptr<ScalarType, void (*)(ScalarType*)> buf3_ptr_dev(
                MemoryST::allocate(sizebuffer - 1), MemoryST::free);

            ScalarType* buf3_alias = buf3_ptr_dev.get();

            MemorySpace::assert_is_dev_ptr(buf3_alias);

            MemorySpace::copy_to_dev(buf3_ptr, sizebuffer - 1, buf3_alias);

            MGMOL_PARALLEL_FOR_COLLAPSE(2, buf3_alias, functions_alias)
            for (int color = 0; color < nfunc; color++)
            {
                for (int k = 0; k < incxy; k++)
                {
                    for (int j = 0; j < nghosts; j++)
                    {
                        ScalarType* uus = functions_alias
                                          + color * size_per_function + nghosts
                                          - 1 - j;
                        size_t index_buf3
                            = color * up_down_size + 1 + j * incxy + k;
                        uus[incy * iinit + k * incy] = buf3_alias[index_buf3];
                    }
                }
            }
        }
        if (up_)
        {
            ScalarType* buf4_ptr = &comm_buf4[0];
            buf4_ptr++;

            std::unique_ptr<ScalarType, void (*)(ScalarType*)> buf4_ptr_dev(
                MemoryST::allocate(sizebuffer - 1), MemoryST::free);

            ScalarType* buf4_alias = buf4_ptr_dev.get();

            MemorySpace::assert_is_dev_ptr(buf4_alias);

            MemorySpace::copy_to_dev(buf4_ptr, sizebuffer - 1, buf4_alias);

            MGMOL_PARALLEL_FOR_COLLAPSE(2, buf4_alias, functions_alias)
            for (int color = 0; color < nfunc; color++)
            {
                for (int k = 0; k < incxy; k++)
                {
                    for (int j = 0; j < nghosts; j++)
                    {
                        ScalarType* uus = functions_alias
                                          + color * size_per_function + nghosts
                                          + zmax + j;
                        size_t index_buf4
                            = color * up_down_size + 1 + j * incxy + k;
                        uus[incy * iinit + k * incy] = buf4_alias[index_buf4];
                    }
                }
            }
        }
    }
    else
    {
        if (bc_[2] == 1) /* grid_.mype_env().n_mpi_task(2)==1 */
        {
            int iinit = (dimy_ + 2 * nghosts_) * nghosts_;
            MGMOL_PARALLEL_FOR_COLLAPSE(2, functions_alias)
            for (int color = 0; color < nfunc; color++)
            {
                for (int j = 0; j < nghosts; j++)
                {
                    for (int k = 0; k < dimxy; k++)
                    {
                        ScalarType* pu
                            = functions_alias + color * size_per_function;
                        size_t index1
                            = nghosts - j + iinit * incy - 1 + k * incy;
                        size_t index2 = nghosts + j + iinit * incy + k * incy;
                        pu[index1]    = pu[index1 + zmax];
                        pu[index2 + zmax] = pu[index2];
                    }
                }
            }
        }
    }

    finishExchangeUpDown_tm_.stop();
}
template <typename ScalarType, typename MemorySpaceType>
template <typename MST,
    typename std::enable_if<std::is_same<MemorySpace::Host, MST>::value>::type*>
void GridFuncVector<ScalarType, MemorySpaceType>::initiateEastWestComm(
    const int begin_color, const int end_color)
{
    std::vector<ScalarType>& comm_buf1(comm_buf1_[0]);
    std::vector<ScalarType>& comm_buf2(comm_buf2_[0]);
    std::vector<ScalarType>& comm_buf3(comm_buf3_[0]);
    std::vector<ScalarType>& comm_buf4(comm_buf4_[0]);

    const int ncolors = end_color - begin_color;

    const int sizebuffer = 1 + nfunc_max_global_ * (east_west_size_ + 1);
    const int xmax       = dimx_ * grid_.inc(0);
    const size_t east_west_size_data = east_west_size_ * sizeof(ScalarType);

    /* Non-blocking MPI */
    if (east_)
        grid_.mype_env().Irecv(
            &comm_buf3[0], sizebuffer, EAST, &req_east_west_[3]);
    if (west_)
        grid_.mype_env().Irecv(
            &comm_buf4[0], sizebuffer, WEST, &req_east_west_[2]);

    if (west_)
    {
        ScalarType* buf1_ptr = &comm_buf1[0];
        // first element will tell how many functions (data) are in buffer
        *buf1_ptr = (ScalarType)ncolors;
        buf1_ptr++;

        const int initu = east_west_size_;
        for (int color = begin_color; color < end_color; color++)
        {
            *buf1_ptr = (ScalarType)gid_[0][color];
            buf1_ptr++;
            const ScalarType* const pu = getDataPtr(color, initu);
            memcpy(buf1_ptr, pu, east_west_size_data);
            buf1_ptr += east_west_size_;
        }

        grid_.mype_env().Isend(
            &comm_buf1[0], sizebuffer, WEST, &req_east_west_[1]);
    }
    if (east_)
    {
        ScalarType* buf2_ptr = &comm_buf2[0];
        // first element will tell how many functions (data) are in buffer
        *buf2_ptr = (ScalarType)ncolors;
        buf2_ptr++;

        const int initu = xmax;
        for (int color = begin_color; color < end_color; color++)
        {
            *buf2_ptr = (ScalarType)gid_[nsubdivx_ - 1][color];
            buf2_ptr++;
            const ScalarType* const pu = getDataPtr(color, initu);
            memcpy(buf2_ptr, pu, east_west_size_data);
            buf2_ptr += east_west_size_;
        }
        grid_.mype_env().Isend(
            &comm_buf2[0], sizebuffer, EAST, &req_east_west_[0]);
    }
}
template <typename ScalarType, typename MemorySpaceType>
template <typename MST, typename std::enable_if<std::is_same<
                            MemorySpace::Device, MST>::value>::type*>
void GridFuncVector<ScalarType, MemorySpaceType>::initiateEastWestComm(
    const int begin_color, const int end_color)
{
    std::vector<ScalarType>& comm_buf1(comm_buf1_[0]);
    std::vector<ScalarType>& comm_buf2(comm_buf2_[0]);
    std::vector<ScalarType>& comm_buf3(comm_buf3_[0]);
    std::vector<ScalarType>& comm_buf4(comm_buf4_[0]);

    const int ncolors = end_color - begin_color;

    const int sizebuffer = 1 + nfunc_max_global_ * (east_west_size_ + 1);
    const int xmax       = dimx_ * grid_.inc(0);

    /* Non-blocking MPI */
    if (east_)
        grid_.mype_env().Irecv(
            &comm_buf3[0], sizebuffer, EAST, &req_east_west_[3]);
    if (west_)
        grid_.mype_env().Irecv(
            &comm_buf4[0], sizebuffer, WEST, &req_east_west_[2]);

    auto size_per_function = grid_.sizeg();
    auto east_west_size    = east_west_size_;

    ScalarType* functions_alias = memory_dev_.get();

    if (west_)
    {
        ScalarType* buf1_ptr = &comm_buf1[0];
        // first element will tell how many functions (data) are in buffer
        *buf1_ptr = (ScalarType)ncolors;
        buf1_ptr++;

        std::unique_ptr<ScalarType, void (*)(ScalarType*)> buf1_ptr_dev(
            MemoryST::allocate(sizebuffer - 1), MemoryST::free);

        ScalarType* buf1_alias = buf1_ptr_dev.get();

        MemorySpace::assert_is_dev_ptr(buf1_alias);

        MGMOL_PARALLEL_FOR_COLLAPSE(2, buf1_alias, functions_alias)
        for (int color = begin_color; color < end_color; color++)
        {
            for (size_t k = 0; k < east_west_size; k++)
            {
                const ScalarType* __restrict__ uus = functions_alias
                                                     + color * size_per_function
                                                     + east_west_size;
                size_t index_buf1      = color * (east_west_size + 1) + 1 + k;
                buf1_alias[index_buf1] = uus[k];
            }
        }

        MemorySpace::copy_to_host(buf1_alias, sizebuffer - 1, buf1_ptr);

        grid_.mype_env().Isend(
            &comm_buf1[0], sizebuffer, WEST, &req_east_west_[1]);
    }
    if (east_)
    {
        ScalarType* buf2_ptr = &comm_buf2[0];
        // first element will tell how many functions (data) are in buffer
        *buf2_ptr = (ScalarType)ncolors;
        buf2_ptr++;

        std::unique_ptr<ScalarType, void (*)(ScalarType*)> buf2_ptr_dev(
            MemoryST::allocate(sizebuffer - 1), MemoryST::free);

        ScalarType* buf2_alias = buf2_ptr_dev.get();

        MemorySpace::assert_is_dev_ptr(buf2_alias);

        MGMOL_PARALLEL_FOR_COLLAPSE(2, buf2_alias, functions_alias)
        for (int color = begin_color; color < end_color; color++)
        {
            for (size_t k = 0; k < east_west_size; k++)
            {
                const ScalarType* __restrict__ uus
                    = functions_alias + color * size_per_function + xmax;
                size_t index_buf2      = color * (east_west_size + 1) + 1 + k;
                buf2_alias[index_buf2] = uus[k];
            }
        }

        MemorySpace::copy_to_host(buf2_alias, sizebuffer - 1, buf2_ptr);

        grid_.mype_env().Isend(
            &comm_buf2[0], sizebuffer, EAST, &req_east_west_[0]);
    }
}
template <typename ScalarType, typename MemorySpaceType>
template <typename MST,
    typename std::enable_if<std::is_same<MemorySpace::Host, MST>::value>::type*>
void GridFuncVector<ScalarType, MemorySpaceType>::finishEastWestComm()
{
    finishExchangeEastWest_tm_.start();

    std::vector<ScalarType>& comm_buf3(comm_buf3_[0]);
    std::vector<ScalarType>& comm_buf4(comm_buf4_[0]);

    const int xmax = dimx_ * grid_.inc(0);

    if (grid_.mype_env().n_mpi_task(0) > 1)
    {

        const size_t east_west_size_data = east_west_size_ * sizeof(ScalarType);
        if (east_)
        {
            ScalarType* buf3_ptr   = &comm_buf3[0];
            const int nremote_func = (int)(*buf3_ptr);
            buf3_ptr++;

            const int initu = xmax + east_west_size_;

            for (int k = 0; k < nremote_func; k++)
            {
                int gid = (int)(*buf3_ptr);
                buf3_ptr++;
                if (gid >= 0)
                {
                    std::map<int, short>::const_iterator ilid
                        = gid2lid_.find(gid);
                    if (ilid != gid2lid_.end())
                    {
                        short lid = ilid->second;
                        if (gid == gid_[nsubdivx_ - 1][lid])
                        {
                            ScalarType* const pu = getDataPtr(lid, initu);
                            memcpy(pu, buf3_ptr, east_west_size_data);
                        }
                    }
                }
                buf3_ptr += east_west_size_;
            }
        }
        if (west_)
        {
            ScalarType* buf4_ptr   = &comm_buf4[0];
            const int nremote_func = (int)(*buf4_ptr);
            buf4_ptr++;

            const int initu = 0;

            for (int k = 0; k < nremote_func; k++)
            {
                int gid = (int)(*buf4_ptr);
                buf4_ptr++;
                if (gid >= 0)
                {
                    std::map<int, short>::const_iterator ilid
                        = gid2lid_.find(gid);
                    if (ilid != gid2lid_.end())
                    {
                        short lid = ilid->second;
                        if (gid == gid_[0][lid])
                        {
                            ScalarType* const pu = getDataPtr(lid, initu);
                            memcpy(pu, buf4_ptr, east_west_size_data);
                        }
                    }
                }
                buf4_ptr += east_west_size_;
            }
        }
    }
    else
    { /* grid_.mype_env().n_mpi_task(0)==1 */

        if (bc_[0] == 1)
        {
            const size_t east_west_size_data
                = east_west_size_ * sizeof(ScalarType);

            for (int k = 0; k < nfunc_; k++)
            {
                ScalarType* pu = getDataPtr(k);
                memcpy(&pu[0], &pu[xmax], east_west_size_data);
                memcpy(&pu[east_west_size_ + xmax], &pu[east_west_size_],
                    east_west_size_data);
            }
        }
    }
    finishExchangeEastWest_tm_.stop();
}
template <typename ScalarType, typename MemorySpaceType>
template <typename MST, typename std::enable_if<std::is_same<
                            MemorySpace::Device, MST>::value>::type*>
void GridFuncVector<ScalarType, MemorySpaceType>::finishEastWestComm()
{
    finishExchangeEastWest_tm_.start();

    std::vector<ScalarType>& comm_buf3(comm_buf3_[0]);
    std::vector<ScalarType>& comm_buf4(comm_buf4_[0]);

    auto sizebuffer = comm_buf3.size();

    auto nfunc             = nfunc_;
    auto east_west_size    = east_west_size_;
    auto size_per_function = grid_.sizeg();

    ScalarType* functions_alias = memory_dev_.get();

    const int xmax = dimx_ * grid_.inc(0);

    if (grid_.mype_env().n_mpi_task(0) > 1)
    {
        if (east_)
        {
            ScalarType* buf3_ptr = &comm_buf3[0];
            buf3_ptr++;

            const int initu = xmax + east_west_size_;

            std::unique_ptr<ScalarType, void (*)(ScalarType*)> buf3_ptr_dev(
                MemoryST::allocate(sizebuffer - 1), MemoryST::free);

            ScalarType* buf3_alias = buf3_ptr_dev.get();

            MemorySpace::assert_is_dev_ptr(buf3_alias);

            MemorySpace::copy_to_dev(buf3_ptr, sizebuffer - 1, buf3_alias);

            MGMOL_PARALLEL_FOR_COLLAPSE(2, buf3_alias, functions_alias)
            for (int color = 0; color < nfunc; color++)
            {
                for (size_t k = 0; k < east_west_size; k++)
                {
                    ScalarType* uus
                        = functions_alias + color * size_per_function + initu;
                    size_t index_buf3 = color * (east_west_size + 1) + 1 + k;
                    uus[k]            = buf3_alias[index_buf3];
                }
            }
        }
        if (west_)
        {
            ScalarType* buf4_ptr = &comm_buf4[0];
            buf4_ptr++;

            const int initu = 0;

            std::unique_ptr<ScalarType, void (*)(ScalarType*)> buf4_ptr_dev(
                MemoryST::allocate(sizebuffer - 1), MemoryST::free);

            ScalarType* buf4_alias = buf4_ptr_dev.get();

            MemorySpace::assert_is_dev_ptr(buf4_alias);

            MemorySpace::copy_to_dev(buf4_ptr, sizebuffer - 1, buf4_alias);

            MGMOL_PARALLEL_FOR_COLLAPSE(2, buf4_alias, functions_alias)
            for (int color = 0; color < nfunc; color++)
            {
                for (size_t k = 0; k < east_west_size; k++)
                {
                    ScalarType* uus
                        = functions_alias + color * size_per_function + initu;
                    size_t index_buf4 = color * (east_west_size + 1) + 1 + k;
                    uus[k]            = buf4_alias[index_buf4];
                }
            }
        }
    }
    else
    { /* grid_.mype_env().n_mpi_task(0)==1 */

        if (bc_[0] == 1)
        {
            MGMOL_PARALLEL_FOR_COLLAPSE(2, functions_alias)
            for (int color = 0; color < nfunc; color++)
            {
                for (size_t k = 0; k < east_west_size; k++)
                {
                    ScalarType* pu
                        = functions_alias + color * size_per_function;
                    pu[k]                         = pu[xmax + k];
                    pu[east_west_size + xmax + k] = pu[east_west_size + k];
                }
            }
        }
    }
    finishExchangeEastWest_tm_.stop();
}
template <typename ScalarType, typename MemorySpaceType>
void GridFuncVector<ScalarType, MemorySpaceType>::trade_boundaries()
{
    if (updated_boundaries_) return;

    assert(dimx_ >= nghosts_);
    assert(dimy_ >= nghosts_);
    assert(dimz_ >= nghosts_);

    trade_bc_tm_.start();

    bool direction[3] = { !bc_[0], !bc_[1], !bc_[2] };
    if (bc_[0] != 1 || bc_[1] != 1 || bc_[2] != 1)
        for (int i = 0; i < nfunc_; i++)
            functions_[i]->setBoundaryValues(0., direction);

// all other functions now work on host only, so host
// data is reference data
#ifdef HAVE_OPENMP_OFFLOAD
    copyHtoD(nfunc_ * grid_.sizeg());
#endif

    // Create buffers
    allocate_buffers(nfunc_max_global_);

    if (grid_.mype_env().n_mpi_task(1) > 1)
    {
        initiateNorthSouthComm(0, nfunc_);
    }
    if (!skinny_stencil_)
    {
        wait_north_south();
        finishNorthSouthComm();
    }

    if (grid_.mype_env().n_mpi_task(2) > 1)
    {
        initiateUpDownComm(0, nfunc_);
    }
    if (!skinny_stencil_)
    {
        wait_up_down();
        finishUpDownComm();
    }

    if (grid_.mype_env().n_mpi_task(0) > 1)
    {
        initiateEastWestComm(0, nfunc_);
    }

    if (!skinny_stencil_)
    {
        wait_east_west();
        finishEastWestComm();
    }

    if (skinny_stencil_)
    {
        wait_north_south();
        finishNorthSouthComm();

        wait_up_down();
        finishUpDownComm();

        wait_east_west();
        finishEastWestComm();
    }

    updated_boundaries_ = true;

    for (int k = 0; k < nfunc_; k++)
        functions_[k]->set_updated_boundaries(true);

    trade_bc_tm_.stop();

// copy to host since some functions are defined on host only
#ifdef HAVE_OPENMP_OFFLOAD
    copyDtoH(nfunc_ * grid_.sizeg());
#endif
}
template <typename ScalarType, typename MemorySpaceType>
void GridFuncVector<ScalarType, MemorySpaceType>::restrict3D(
    GridFuncVector& ucoarse)
{
    if (!updated_boundaries_) trade_boundaries();

    MGkernelRestrict3D(data(), grid_, ucoarse.data(), ucoarse.grid(), nfunc_);
}

template <typename ScalarType, typename MemorySpaceType>
void GridFuncVector<ScalarType, MemorySpaceType>::extend3D(
    GridFuncVector& ucoarse)
{
    if (!ucoarse.updated_boundaries_) ucoarse.trade_boundaries();

    MGkernelExtend3D(ucoarse.data(), ucoarse.grid(), data(), grid_, nfunc_);

    updated_boundaries_ = false;
}

template <typename ScalarType, typename MemorySpaceType>
GridFuncVector<ScalarType, MemorySpaceType>&
GridFuncVector<ScalarType, MemorySpaceType>::operator-=(
    const GridFuncVector<ScalarType, MemorySpaceType>& func)
{
    assert(func.grid_.sizeg() == grid_.sizeg());
    assert(func.grid_.ghost_pt() == grid_.ghost_pt());
    assert(this != &func);

    LinearAlgebraUtils<MemorySpace::Host>::MPaxpy(nfunc_ * grid_.sizeg(),
        (ScalarType)(-1.), func.memory_.get(), memory_.get());

    updated_boundaries_ = (func.updated_boundaries_ && updated_boundaries_);

    return *this;
}

template <typename ScalarType, typename MemorySpaceType>
template <typename ScalarType2>
void GridFuncVector<ScalarType, MemorySpaceType>::copyFrom(
    const GridFuncVector<ScalarType2, MemorySpaceType>& src)
{
    copy_tm_.start();

    MPcpy(memory_.get(), src.getDataPtr(0), nfunc_ * grid_.sizeg());

    updated_boundaries_ = src.getUpdatedBoundariesFlag();

    copy_tm_.stop();
}

template <typename ScalarType, typename MemorySpaceType>
template <typename ScalarType2>
void GridFuncVector<ScalarType, MemorySpaceType>::axpy(const ScalarType2 alpha,
    const GridFuncVector<ScalarType, MemorySpaceType>& func)
{
    LinearAlgebraUtils<MemorySpace::Host>::MPaxpy(
        nfunc_ * grid_.sizeg(), alpha, func.memory_.get(), memory_.get());

    updated_boundaries_ = (func.updated_boundaries_ && updated_boundaries_);
}

template <typename ScalarType, typename MemorySpaceType>
template <typename InputScalarType>
void GridFuncVector<ScalarType, MemorySpaceType>::getValues(
    const int k, InputScalarType* vv) const
{
    assert(k < static_cast<int>(functions_.size()));
    functions_[k]->template getValues<InputScalarType, MemorySpaceType>(vv);
}

// build list of local gids I need ghost values for
template <typename ScalarType, typename MemorySpaceType>
void GridFuncVector<ScalarType, MemorySpaceType>::communicateRemoteGids(
    const int begin_color, const int end_color)
{
    const int ncolors = end_color - begin_color;

    MPI_Request gid_req[6];

    // build list of local gids I need ghost values for
    std::vector<int> local_gids;
    const short ndata = ncolors * nsubdivx_;
    local_gids.reserve(ndata);
    for (short color = begin_color; color < end_color; color++)
    {
        for (short iloc = 0; iloc < nsubdivx_; iloc++)
        {
            if (color < static_cast<int>(gid_[iloc].size()))
                local_gids.push_back(gid_[iloc][color]);
            else
                local_gids.push_back(-1);
        }
    }

    for (short i = 0; i < 6; i++)
        remote_gids_[i].resize(ndata);

    if (north_)
        grid_.mype_env().Irecv(
            &remote_gids_[NORTH][0], ndata, NORTH, &gid_req[NORTH]);
    if (south_)
        grid_.mype_env().Irecv(
            &remote_gids_[SOUTH][0], ndata, SOUTH, &gid_req[SOUTH]);
    if (east_)
        grid_.mype_env().Irecv(
            &remote_gids_[EAST][0], ndata, EAST, &gid_req[EAST]);
    if (west_)
        grid_.mype_env().Irecv(
            &remote_gids_[WEST][0], ndata, WEST, &gid_req[WEST]);
    if (up_)
        grid_.mype_env().Irecv(&remote_gids_[UP][0], ndata, UP, &gid_req[UP]);
    if (down_)
        grid_.mype_env().Irecv(
            &remote_gids_[DOWN][0], ndata, DOWN, &gid_req[DOWN]);

    if (north_)
        grid_.mype_env().Isend(&local_gids[0], ndata, NORTH, &gid_req[NORTH]);
    if (south_)
        grid_.mype_env().Isend(&local_gids[0], ndata, SOUTH, &gid_req[SOUTH]);
    if (east_)
        grid_.mype_env().Isend(&local_gids[0], ndata, EAST, &gid_req[EAST]);
    if (west_)
        grid_.mype_env().Isend(&local_gids[0], ndata, WEST, &gid_req[WEST]);
    if (up_) grid_.mype_env().Isend(&local_gids[0], ndata, UP, &gid_req[UP]);
    if (down_)
        grid_.mype_env().Isend(&local_gids[0], ndata, DOWN, &gid_req[DOWN]);

    if (south_) MPI_Wait(gid_req + SOUTH, MPI_STATUS_IGNORE);
    if (north_) MPI_Wait(gid_req + NORTH, MPI_STATUS_IGNORE);
    if (west_) MPI_Wait(gid_req + WEST, MPI_STATUS_IGNORE);
    if (east_) MPI_Wait(gid_req + EAST, MPI_STATUS_IGNORE);
    if (down_) MPI_Wait(gid_req + DOWN, MPI_STATUS_IGNORE);
    if (up_) MPI_Wait(gid_req + UP, MPI_STATUS_IGNORE);
}

// assumes all processors call with same arguments
template <typename ScalarType, typename MemorySpaceType>
void GridFuncVector<ScalarType, MemorySpaceType>::trade_boundaries_colors(
    const short first_color, const short last_color)
{
    assert(first_color >= 0);
    assert(last_color >= 0);
    assert(comm_buf3_.size() > 0);

    grid_.mype_env().barrier();
    // if (onpe0)
    //    std::cout << "Color " << first_color << " to " << last_color - 1
    //              << std::endl;

    if (updated_boundaries_) return;

    std::vector<ScalarType>& comm_buf1(comm_buf1_[0]);
    std::vector<ScalarType>& comm_buf2(comm_buf2_[0]);
    std::vector<ScalarType>& comm_buf3(comm_buf3_[0]);
    std::vector<ScalarType>& comm_buf4(comm_buf4_[0]);

    const short ncolors   = last_color - first_color;
    const short end_color = std::min(last_color, (short)functions_.size());

    for (short color = first_color + 1; color < end_color; color++)
        assert(functions_[color]->updated_boundaries()
               == functions_[first_color]->updated_boundaries());

    trade_bc_colors_tm_.start();

    const int xmax = dimx_ * grid_.inc(0);
    const int ymax = dimy_ * grid_.inc(1);
    const int zmax = dimz_;

    const size_t sdimz = dimz_ * sizeof(ScalarType);

    communicateRemoteGids(first_color, last_color);

    const int ione = 1;

    // Create buffers
    allocate_buffers(ncolors);

    if (grid_.mype_env().n_mpi_task(1) > 1)
    {

        const size_t sizebuffer = 1 + ncolors * south_north_size_;
        if (north_)
            grid_.mype_env().Irecv(
                &comm_buf4[0], sizebuffer, NORTH, &req_north_south_[3]);
        if (south_)
            grid_.mype_env().Irecv(
                &comm_buf3[0], sizebuffer, SOUTH, &req_north_south_[1]);

        const int imin = nghosts_ * incx_;
        const int iinc = incx_ * dimx_ / nsubdivx_;

        const int jmax = nghosts_ * incy_;

        if (south_)
        {
            ScalarType* buf2_ptr = &comm_buf2[0];
            // first element will tell how many functions (data) are in buffer
            *buf2_ptr = (ScalarType)ncolors;
            buf2_ptr++;
            // pack data
            for (short color = first_color; color < last_color; color++)
            {
                for (short iloc = 0; iloc < nsubdivx_; iloc++)
                {
                    const short rgid
                        = remote_gids_[SOUTH][nsubdivx_ * color + iloc];
                    const short k = gid2lid_[rgid];
                    assert(k < static_cast<int>(functions_.size()));
                    const ScalarType* uus
                        = functions_[k]->uu(nghosts_ * (incy_ + 1));
                    *buf2_ptr = (ScalarType)gid_[iloc][k];
                    buf2_ptr++;
                    for (int j = 0; j < jmax; j += incy_)
                        for (int i = imin + iloc * iinc;
                             i < imin + (iloc + 1) * iinc; i += incx_)
                        {
                            memcpy(buf2_ptr, &uus[i + j], sdimz);
                            buf2_ptr += dimz_;
                        }
                }
            }
            grid_.mype_env().Isend(&comm_buf2[0],
                1 + ncolors * south_north_size_, SOUTH, &req_north_south_[2]);
        }

        if (north_)
        {
            ScalarType* buf1_ptr = &comm_buf1[0];
            // first element will tell how many functions (data) are in buffer
            *buf1_ptr = (ScalarType)ncolors;
            buf1_ptr++;
            // pack data
            for (short color = first_color; color < last_color; color++)
            {
                for (short iloc = 0; iloc < nsubdivx_; iloc++)
                {
                    const short rgid
                        = remote_gids_[NORTH][nsubdivx_ * color + iloc];
                    const short k         = gid2lid_[rgid];
                    const ScalarType* uus = functions_[k]->uu(nghosts_ + ymax);
                    *buf1_ptr             = (ScalarType)gid_[iloc][k];
                    buf1_ptr++;
                    for (int j = 0; j < jmax; j += incy_)
                        for (int i = imin + iloc * iinc;
                             i < imin + (iloc + 1) * iinc; i += incx_)
                        {
                            memcpy(buf1_ptr, &uus[i + j], sdimz);
                            buf1_ptr += dimz_;
                        }
                }
            }
            grid_.mype_env().Isend(&comm_buf1[0],
                1 + ncolors * south_north_size_, NORTH, &req_north_south_[0]);
        }

        wait_north_south();

        if (north_)
        {
            ScalarType* buf4_ptr   = &comm_buf4[0];
            const int nremote_func = (int)(*buf4_ptr);
            buf4_ptr++;

            for (int k = 0; k < nremote_func; k++)
            {
                for (short iloc = 0; iloc < nsubdivx_; iloc++)
                {
                    const int gid = (int)(*buf4_ptr);
                    buf4_ptr++;

                    if (gid >= 0)
                    {
                        std::map<int, short>::const_iterator ilid
                            = gid2lid_.find(gid);
                        if (ilid != gid2lid_.end())
                        {
                            const short lid = ilid->second;
                            if (gid == gid_[iloc][lid])
                            {
                                ScalarType* uus = functions_[lid]->uu(
                                    nghosts_ * (incy_ + 1) + ymax);
                                for (int j = 0; j < jmax; j += incy_)
                                    for (int i = imin + iloc * iinc;
                                         i < imin + (iloc + 1) * iinc;
                                         i += incx_)
                                    {
                                        memcpy(&uus[i + j], buf4_ptr, sdimz);
                                        buf4_ptr += dimz_;
                                    }
                            }
                            else
                            {
                                for (int j = 0; j < jmax; j += incy_)
                                    for (int i = imin + iloc * iinc;
                                         i < imin + (iloc + 1) * iinc;
                                         i += incx_)
                                        buf4_ptr += dimz_;
                            }
                        }
                        else
                        {
                            for (int j = 0; j < jmax; j += incy_)
                                for (int i = imin + iloc * iinc;
                                     i < imin + (iloc + 1) * iinc; i += incx_)
                                    buf4_ptr += dimz_;
                        }
                    }
                    else
                    {
                        for (int j = 0; j < jmax; j += incy_)
                            for (int i = imin + iloc * iinc;
                                 i < imin + (iloc + 1) * iinc; i += incx_)
                            {
                                buf4_ptr += dimz_;
                            }
                    }
                }
            }
            assert((buf4_ptr - &comm_buf4[0]) <= static_cast<int>(sizebuffer));
        }

        if (south_)
        {
            ScalarType* buf3_ptr   = &comm_buf3[0];
            const int nremote_func = (int)(*buf3_ptr);
            buf3_ptr++;

            for (int k = 0; k < nremote_func; k++)
            {
                for (short iloc = 0; iloc < nsubdivx_; iloc++)
                {
                    const int gid = (int)(*buf3_ptr);
                    buf3_ptr++;
                    if (gid >= 0)
                    {
                        std::map<int, short>::const_iterator ilid
                            = gid2lid_.find(gid);
                        if (ilid != gid2lid_.end())
                        {
                            const short lid = ilid->second;
                            if (gid == gid_[iloc][lid])
                            {
                                ScalarType* uus = functions_[lid]->uu(nghosts_);
                                for (int j = 0; j < jmax; j += incy_)
                                    for (int i = imin + iloc * iinc;
                                         i < imin + (iloc + 1) * iinc;
                                         i += incx_)
                                    {
                                        memcpy(&uus[i + j], buf3_ptr, sdimz);
                                        buf3_ptr += dimz_;
                                    }
                            }
                            else
                            {
                                for (int j = 0; j < jmax; j += incy_)
                                    for (int i = imin + iloc * iinc;
                                         i < imin + (iloc + 1) * iinc;
                                         i += incx_)
                                        buf3_ptr += dimz_;
                            }
                        }
                        else
                        {
                            for (int j = 0; j < jmax; j += incy_)
                                for (int i = imin + iloc * iinc;
                                     i < imin + (iloc + 1) * iinc; i += incx_)
                                    buf3_ptr += dimz_;
                        }
                    }
                    else
                    {
                        for (int j = 0; j < jmax; j += incy_)
                            for (int i = imin + iloc * iinc;
                                 i < imin + (iloc + 1) * iinc; i += incx_)
                            {
                                buf3_ptr += dimz_;
                            }
                    }
                }
            }
            assert((buf3_ptr - &comm_buf3[0]) <= static_cast<int>(sizebuffer));
        }
    }
    else if (bc_[1] == 1)
    { // grid_.mype_env().n_mpi_task(1)==1

        // only for i already initialized
        for (short color = first_color; color < end_color; color++)
        {
            ScalarType* pu = functions_[color]->uu();
            for (int j = 0; j < nghosts_ * incy_; j += incy_)
                for (int i = nghosts_ * incx_; i < (dimx_ + nghosts_) * incx_;
                     i += incx_)
                {

                    memcpy(&pu[i + nghosts_ + j], &pu[i + nghosts_ + j + ymax],
                        sdimz);
                    memcpy(&pu[i + nghosts_ * (incy_ + 1) + j + ymax],
                        &pu[i + nghosts_ * (incy_ + 1) + j], sdimz);
                }
        }
    }

    int iinit = (dimy_ + 2 * nghosts_) * nghosts_;

    if (grid_.mype_env().n_mpi_task(2) > 1)
    {

        const int sizebuffer = 1 + ncolors * up_down_size_;
        if (down_)
        {
            assert(sizebuffer <= static_cast<int>(comm_buf3.size()));
            grid_.mype_env().Irecv(
                &comm_buf3[0], sizebuffer, DOWN, &req_up_down_[1]);
        }
        if (up_)
        {
            grid_.mype_env().Irecv(
                &comm_buf4[0], sizebuffer, UP, &req_up_down_[3]);
        }

        if (up_)
        {
            ScalarType* buf1_ptr = &comm_buf1[0];
            // first element will tell how many functions (data) are in buffer
            *buf1_ptr = (ScalarType)ncolors;
            buf1_ptr++;
            for (short color = first_color; color < last_color; color++)
            {
                for (short iloc = 0; iloc < nsubdivx_; iloc++)
                {
                    const short rgid
                        = remote_gids_[UP][nsubdivx_ * color + iloc];
                    const short k = gid2lid_[rgid];
                    const ScalarType* const uus
                        = functions_[k]->uu(nghosts_ + incy_ * iinit);
                    *buf1_ptr = (ScalarType)gid_[iloc][k];
                    buf1_ptr++;
                    for (int j = 0; j < nghosts_; j++)
                    {
                        Tcopy(&incxy_,
                            &uus[zmax - 1 - j + iloc * incxy_ * incy_], &incy_,
                            buf1_ptr, &ione);
                        buf1_ptr += incxy_;
                    }
                }
            }
            grid_.mype_env().Isend(&comm_buf1[0], 1 + ncolors * up_down_size_,
                UP, &req_up_down_[0]);
        }
        if (down_)
        {
            ScalarType* buf2_ptr = &comm_buf2[0];
            // first element will tell how many functions (data) are in buffer
            *buf2_ptr = (ScalarType)ncolors;
            buf2_ptr++;
            for (short color = first_color; color < last_color; color++)
            {
                for (short iloc = 0; iloc < nsubdivx_; iloc++)
                {
                    const short rgid
                        = remote_gids_[DOWN][nsubdivx_ * color + iloc];
                    const short k = gid2lid_[rgid];
                    const ScalarType* const uus
                        = functions_[k]->uu(nghosts_ + incy_ * iinit);
                    *buf2_ptr = (ScalarType)gid_[iloc][k];
                    buf2_ptr++;
                    for (int j = 0; j < nghosts_; j++)
                    {
                        Tcopy(&incxy_, &uus[j + iloc * incxy_ * incy_], &incy_,
                            buf2_ptr, &ione);
                        buf2_ptr += incxy_;
                    }
                }
            }
            grid_.mype_env().Isend(&comm_buf2[0], 1 + ncolors * up_down_size_,
                DOWN, &req_up_down_[2]);
        }

        wait_up_down();

        if (down_)
        {
            ScalarType* buf3_ptr   = &comm_buf3[0];
            const int nremote_func = (int)(*buf3_ptr);
            buf3_ptr++;

            for (int k = 0; k < nremote_func; k++)
            {
                for (short iloc = 0; iloc < nsubdivx_; iloc++)
                {
                    const int gid = (int)(*buf3_ptr);
                    buf3_ptr++;
                    if (gid >= 0)
                    {
                        std::map<int, short>::const_iterator ilid
                            = gid2lid_.find(gid);
                        if (ilid != gid2lid_.end())
                        {
                            short lid = ilid->second;
                            if (gid == gid_[iloc][lid])
                            {
                                for (int j = 0; j < nghosts_; j++)
                                {
                                    ScalarType* const uus
                                        = functions_[lid]->uu(nghosts_ - 1 - j);
                                    Tcopy(&incxy_, buf3_ptr, &ione,
                                        &uus[incy_ * iinit
                                             + iloc * incxy_ * incy_],
                                        &incy_);
                                    buf3_ptr += incxy_;
                                }
                            }
                            else
                            {
                                buf3_ptr += incxy_ * nghosts_;
                            }
                        }
                        else
                        {
                            buf3_ptr += incxy_ * nghosts_;
                        }
                    }
                    else
                    {
                        buf3_ptr += incxy_ * nghosts_;
                    }
                }
            }
        }
        if (up_)
        {
            ScalarType* buf4_ptr   = &comm_buf4[0];
            const int nremote_func = (int)(*buf4_ptr);
            buf4_ptr++;

            for (int k = 0; k < nremote_func; k++)
            {
                for (short iloc = 0; iloc < nsubdivx_; iloc++)
                {
                    const int gid = (int)(*buf4_ptr);
                    buf4_ptr++;
                    if (gid >= 0)
                    {
                        std::map<int, short>::const_iterator ilid
                            = gid2lid_.find(gid);
                        if (ilid != gid2lid_.end())
                        {
                            short lid = ilid->second;
                            if (gid == gid_[iloc][lid])
                            {
                                for (int j = 0; j < nghosts_; j++)
                                {
                                    ScalarType* const uus = functions_[lid]->uu(
                                        nghosts_ + zmax + j);
                                    Tcopy(&incxy_, buf4_ptr, &ione,
                                        &uus[incy_ * iinit
                                             + iloc * incxy_ * incy_],
                                        &incy_);
                                    buf4_ptr += incxy_;
                                }
                            }
                            else
                            {
                                buf4_ptr += incxy_ * nghosts_;
                            }
                        }
                        else
                        {
                            buf4_ptr += incxy_ * nghosts_;
                        }
                    }
                    else
                    {
                        buf4_ptr += incxy_ * nghosts_;
                    }
                }
            }
        }
    }
    else if (bc_[2] == 1)
    { /* grid_.mype_env().n_mpi_task(2)==1 */

        for (short color = first_color; color < end_color; color++)
        {
            ScalarType* pu = functions_[color]->uu();
            for (int j = 0; j < nghosts_; j++)
            {
                Tcopy(&dimxy_, &pu[nghosts_ - j + iinit * incy_ + zmax - 1],
                    &incy_, &pu[nghosts_ - j + iinit * incy_ - 1], &incy_);
            }
            for (int j = 0; j < nghosts_; j++)
            {
                Tcopy(&dimxy_, &pu[nghosts_ + j + iinit * incy_], &incy_,
                    &pu[nghosts_ + j + iinit * incy_ + zmax], &incy_);
            }
        }
    }

    const int east_west_size_        = nghosts_ * incx_;
    const size_t east_west_size_data = east_west_size_ * sizeof(ScalarType);

    if (grid_.mype_env().n_mpi_task(0) > 1)
    {
        const int sizebuffer = 1 + ncolors * (east_west_size_ + 1);

        /* Non-blocking MPI */
        if (east_)
            grid_.mype_env().Irecv(
                &comm_buf3[0], sizebuffer, EAST, &req_east_west_[3]);
        if (west_)
            grid_.mype_env().Irecv(
                &comm_buf4[0], sizebuffer, WEST, &req_east_west_[2]);

        if (west_)
        {
            ScalarType* buf1_ptr = &comm_buf1[0];
            // first element will tell how many functions (data) are in buffer
            *buf1_ptr = (ScalarType)ncolors;
            buf1_ptr++;

            const int initu = east_west_size_;
            for (short color = first_color; color < last_color; color++)
            {
                const short rgid
                    = remote_gids_[WEST][nsubdivx_ * color + nsubdivx_ - 1];
                if (rgid >= 0)
                {
                    const short k = gid2lid_[rgid];
                    *buf1_ptr     = (ScalarType)gid_[0][k];
                    buf1_ptr++;
                    if (k < static_cast<int>(functions_.size()))
                    {
                        const ScalarType* const pu = functions_[k]->uu(initu);
                        memcpy(buf1_ptr, pu, east_west_size_data);
                    }
                }
                else
                {
                    *buf1_ptr = -1.;
                    buf1_ptr++;
                }
                buf1_ptr += east_west_size_;
            }
            grid_.mype_env().Isend(
                &comm_buf1[0], sizebuffer, WEST, &req_east_west_[1]);
        }
        if (east_)
        {
            ScalarType* buf2_ptr = &comm_buf2[0];
            // first element will tell how many functions (data) are in buffer
            *buf2_ptr = (ScalarType)ncolors;
            buf2_ptr++;

            const int initu = xmax;
            for (short color = first_color; color < last_color; color++)
            {
                const short rgid = remote_gids_[EAST][nsubdivx_ * color + 0];
                if (rgid >= 0)
                {
                    const short k = gid2lid_[rgid];
                    *buf2_ptr     = (ScalarType)gid_[nsubdivx_ - 1][k];
                    buf2_ptr++;
                    if (k < static_cast<int>(functions_.size()))
                    {
                        const ScalarType* const pu = functions_[k]->uu(initu);
                        memcpy(buf2_ptr, pu, east_west_size_data);
                    }
                }
                else
                {
                    *buf2_ptr = -1.;
                    buf2_ptr++;
                }
                buf2_ptr += east_west_size_;
            }
            grid_.mype_env().Isend(
                &comm_buf2[0], sizebuffer, EAST, &req_east_west_[0]);
        }

        wait_east_west();

        if (east_)
        {
            ScalarType* buf3_ptr   = &comm_buf3[0];
            const int nremote_func = (int)(*buf3_ptr);
            buf3_ptr++;

            const int initu = xmax + east_west_size_;
            for (int k = 0; k < nremote_func; k++)
            {
                int gid = (int)(*buf3_ptr);
                buf3_ptr++;
                if (gid >= 0)
                {
                    std::map<int, short>::const_iterator ilid
                        = gid2lid_.find(gid);
                    if (ilid != gid2lid_.end())
                    {
                        short lid = ilid->second;
                        if (gid == gid_[nsubdivx_ - 1][lid])
                        {
                            ScalarType* const pu = functions_[lid]->uu(initu);
                            memcpy(pu, buf3_ptr, east_west_size_data);
                        }
                    }
                }
                buf3_ptr += east_west_size_;
            }
        }
        if (west_)
        {
            ScalarType* buf4_ptr   = &comm_buf4[0];
            const int nremote_func = (int)(*buf4_ptr);
            buf4_ptr++;

            const int initu = 0;
            for (int k = 0; k < nremote_func; k++)
            {
                int gid = (int)(*buf4_ptr);
                buf4_ptr++;
                if (gid >= 0)
                {
                    std::map<int, short>::const_iterator ilid
                        = gid2lid_.find(gid);
                    if (ilid != gid2lid_.end())
                    {
                        short lid = ilid->second;
                        if (gid == gid_[0][lid])
                        {
                            ScalarType* const pu = functions_[lid]->uu(initu);
                            memcpy(pu, buf4_ptr, east_west_size_data);
                        }
                    }
                }
                buf4_ptr += east_west_size_;
            }
        }
    }
    else if (bc_[0] == 1)
    { /* grid_.mype_env().n_mpi_task(0)==1 */

        for (short color = first_color; color < end_color; color++)
        {
            ScalarType* pu = functions_[color]->uu();
            memcpy(&pu[0], &pu[xmax], east_west_size_data);
            memcpy(&pu[east_west_size_ + xmax], &pu[east_west_size_],
                east_west_size_data);
        }
    }

    updated_boundaries_ = true;

    for (short color = first_color; color < end_color; color++)
        functions_[color]->set_updated_boundaries(true);

    trade_bc_colors_tm_.stop();
}

template <typename ScalarType, typename MemorySpaceType>
void GridFuncVector<ScalarType, MemorySpaceType>::applyLap(
    const int type, GridFuncVector<ScalarType, MemorySpaceType>& rhs)
{
    switch (type)
    {
        // Laph4M
        case 0:
            this->del2_4th_Mehr(rhs);
            break;
        // Laph2
        case 1:
            this->del2_2nd(rhs);
            break;
        case 2:
            this->del2_4th(rhs);
            break;
        case 3:
            this->del2_6th(rhs);
            break;
        case 4:
            this->del2_8th(rhs);
            break;
        default:
            std::cerr << "LapFactory::createLap() --- option invalid:" << type
                      << std::endl;
            abort();
    }
}

template <typename ScalarType, typename MemorySpaceType>
void GridFuncVector<ScalarType, MemorySpaceType>::applyRHS(
    const int type, GridFuncVector<ScalarType, MemorySpaceType>& rhs)
{
    // Laph4M
    if (type == 0)
    {
        this->rhs_4th_Mehr1(rhs);
    }
    else
    {
        memcpy(memory_.get(), rhs.memory_.get(),
            functions_.size() * grid_.sizeg() * sizeof(ScalarType));
    }
}

template <typename ScalarType, typename MemorySpaceType>
void GridFuncVector<ScalarType, MemorySpaceType>::jacobi(const int type,
    const GridFuncVector<ScalarType, MemorySpaceType>& B,
    GridFuncVector<ScalarType, MemorySpaceType>& w, const double jacobiFactor)
{
    applyLap(type, w);
    w -= B;
    axpy((ScalarType)(-1. * jacobiFactor), w);

    set_updated_boundaries(false);
}

template <typename ScalarType, typename MemorySpaceType>
void GridFuncVector<ScalarType, MemorySpaceType>::app_mask(const short level)
{
    if (map2masks_ == nullptr) return;

    const int nfunc = (int)gid_[0].size();
#pragma omp parallel for
    for (int color = 0; color < nfunc; color++)
    {
        map2masks_->apply(getGridFunc(color), gid_, level, color);
    }
}

template class GridFuncVector<double, MemorySpace::Host>;
template class GridFuncVector<float, MemorySpace::Host>;
template void GridFuncVector<float, MemorySpace::Host>::assign(
    const int i, const float* const v, const char dis);
template void GridFuncVector<float, MemorySpace::Host>::assign(
    const int i, const double* const v, const char dis);
template void GridFuncVector<double, MemorySpace::Host>::assign(
    const int i, const float* const v, const char dis);
template void GridFuncVector<double, MemorySpace::Host>::assign(
    const int i, const double* const v, const char dis);
template void GridFuncVector<float, MemorySpace::Host>::getValues<float>(
    const int, float*) const;
template void GridFuncVector<float, MemorySpace::Host>::getValues<double>(
    const int, double*) const;
template void GridFuncVector<double, MemorySpace::Host>::getValues<float>(
    const int, float*) const;
template void GridFuncVector<double, MemorySpace::Host>::getValues<double>(
    const int, double*) const;
template void GridFuncVector<double, MemorySpace::Host>::pointwiseProduct(
    GridFuncVector<double, MemorySpace::Host>& A, const GridFunc<double>& B);
template void GridFuncVector<float, MemorySpace::Host>::pointwiseProduct(
    GridFuncVector<float, MemorySpace::Host>& A, const GridFunc<double>& B);

template void GridFuncVector<float, MemorySpace::Host>::axpy(
    const float alpha, const GridFuncVector<float, MemorySpace::Host>& func);
template void GridFuncVector<double, MemorySpace::Host>::axpy(
    const double alpha, const GridFuncVector<double, MemorySpace::Host>& func);
template void GridFuncVector<float, MemorySpace::Host>::copyFrom(
    const GridFuncVector<double, MemorySpace::Host>& src);
template void GridFuncVector<double, MemorySpace::Host>::copyFrom(
    const GridFuncVector<float, MemorySpace::Host>& src);
template void GridFuncVector<ORBDTYPE, MemorySpace::Host>::copyFrom(
    const GridFuncVector<ORBDTYPE, MemorySpace::Host>& src);

#ifdef HAVE_MAGMA
template class GridFuncVector<double, MemorySpace::Device>;
template class GridFuncVector<float, MemorySpace::Device>;
template void GridFuncVector<float, MemorySpace::Device>::getValues<float>(
    const int, float*) const;
template void GridFuncVector<float, MemorySpace::Device>::getValues<double>(
    const int, double*) const;
template void GridFuncVector<double, MemorySpace::Device>::getValues<float>(
    const int, float*) const;
template void GridFuncVector<double, MemorySpace::Device>::getValues<double>(
    const int, double*) const;
template void GridFuncVector<double, MemorySpace::Device>::pointwiseProduct(
    GridFuncVector<double, MemorySpace::Device>& A, const GridFunc<double>& B);
#endif

} // namespace pb

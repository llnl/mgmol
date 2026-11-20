// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "BlockVector.h"
#include "Control.h"
#include "MGmol_MPI.h"
#include "MPIdata.h"
#include "global.h"
#include "memory_space.h"

#include "MGmol_abort.h"
#include <cassert>

namespace
{
template <typename ScalarType, typename MemorySpaceType>
struct BV
{
    static void subtract(int const numel, ScalarType const* s, ScalarType* v);
};

// MemorySpace::Host
template <typename ScalarType>
struct BV<ScalarType, MemorySpace::Host>
{
    static void subtract(int const numel, ScalarType const* s, ScalarType* v)
    {
#pragma omp parallel for
        for (int j = 0; j < numel; j++)
            v[j] -= s[j];
    }
};

// MemorySpace::Device
#ifdef HAVE_MAGMA
template <>
struct BV<float, MemorySpace::Device>
{
    static void subtract(int const numel, float const* s, float* v)
    {
        auto& magma_singleton = MagmaSingleton::get_magma_singleton();
        int const increment   = 1;
        magma_saxpy(
            numel, -1., s, increment, v, increment, magma_singleton.queue_);
    }
};

template <>
struct BV<double, MemorySpace::Device>
{
    static void subtract(int const numel, double const* s, double* v)
    {
        auto& magma_singleton = MagmaSingleton::get_magma_singleton();
        int const increment   = 1;
        magma_daxpy(
            numel, -1., s, increment, v, increment, magma_singleton.queue_);
    }
};
#endif
}

template <typename ScalarType, typename MemorySpaceType>
BlockVector<ScalarType, MemorySpaceType>::BlockVector(
    const pb::Grid& my_grid, const short subdivx, const short bc[3])
    : mygrid_(my_grid)
{
    for (short i = 0; i < 3; i++)
        bc_[i] = bc[i];
    if (n_instances_ == 0)
    {
        subdivx_  = subdivx;
        numel_    = (int)my_grid.size();
        locnumel_ = numel_ / subdivx_;
        ld_       = numel_ + PAD;
    }

    ld_instance_ = ld_;

    storage_       = nullptr;
    my_allocation_ = -1;

    n_instances_++;
}

template <typename ScalarType, typename MemorySpaceType>
BlockVector<ScalarType, MemorySpaceType>::~BlockVector()
{
    deallocate_storage();
    if (n_instances_ == 1)
    {
        delete data_wghosts_;
        data_wghosts_ = nullptr;
        allocated_.clear();
        for (typename std::vector<ScalarType*>::iterator it
             = class_storage_.begin();
             it != class_storage_.end(); ++it)
            MemorySpace::Memory<ScalarType, MemorySpaceType>::free(*it);
    }
    n_instances_--;
}

template <typename ScalarType, typename MemorySpaceType>
void BlockVector<ScalarType, MemorySpaceType>::allocate1NewBlock()
{
    assert(size_storage_ > 0);
    assert(overallocate_factor_ >= 1.);

    // size of allocation is based on first call to this function
    if (allocated_size_storage_ == 0)
        allocated_size_storage_ = static_cast<int>(
            static_cast<ScalarType>(size_storage_) * overallocate_factor_);

    // check if storage required is not bigger than initial allocation
    if (size_storage_ > allocated_size_storage_)
    {
        (*MPIdata::sout)
            << "BlockVector::allocate1NewBlock(): allocated_size_storage_ = "
            << allocated_size_storage_ << std::endl;
        (*MPIdata::sout)
            << "BlockVector::allocate1NewBlock(): size_storage_           = "
            << size_storage_ << std::endl;
        (*MPIdata::sout)
            << "BlockVector::allocate1NewBlock(): size_storage_ too large!!!"
            << std::endl;
        mgmol_abort("BlockVector::allocate1NewBlock()");
    }

    Control& ct = *(Control::instance());
    if (onpe0 && ct.verbose > 1)
        (*MPIdata::sout) << "BlockVector, allocate memory size "
                         << allocated_size_storage_ << std::endl;

    class_storage_.push_back(
        MemorySpace::Memory<ScalarType, MemorySpaceType>::allocate(
            allocated_size_storage_));

    MemorySpace::Memory<ScalarType, MemorySpaceType>::set(
        class_storage_.back(), allocated_size_storage_, 0);
}

template <typename ScalarType, typename MemorySpaceType>
void BlockVector<ScalarType, MemorySpaceType>::allocate_storage()
{
    assert(size_storage_ > 0);
    assert(storage_ == nullptr);

    Control& ct                  = *(Control::instance());
    static short high_water_mark = 0;

    // firt time, allocate memory for max_alloc_instances_ objects
    if (allocated_.size() == 0)
    {
        if (onpe0 && ct.verbose > 1)
            (*MPIdata::sout) << "BlockVector, allocate " << max_alloc_instances_
                             << " instances" << std::endl;
        for (short i = 0; i < max_alloc_instances_; i++)
            allocate1NewBlock();

        allocated_.clear();
        allocated_.push_back(1);
        my_allocation_ = 0;
    }
    else
    {
        std::vector<short>::iterator ia = allocated_.begin();
        short ii                        = 0;
        while (ia != allocated_.end())
        {
            if (*ia == 0) break;
            ia++;
            ii++;
        }
        if (ia == allocated_.end())
        {
            allocated_.push_back(1);
        }
        else
        {
            *ia = 1;
        }
        my_allocation_ = ii;
        if (my_allocation_ >= static_cast<int>(class_storage_.size()))
        {
            std::cerr << "Number of allocated blocks: " << class_storage_.size()
                      << ", Cannot allocate another BlockVector!!!"
                      << std::endl;
            mgmol_abort("BlockVector::allocate_storage()");
        }
    }

    if (my_allocation_ > high_water_mark) high_water_mark = my_allocation_;
    if (onpe0 && ct.verbose > 2)
    {
        (*MPIdata::sout) << "allocation in slot " << my_allocation_
                         << std::endl;
        (*MPIdata::sout) << "high water mark: " << high_water_mark << std::endl;
    }
    assert(my_allocation_ < static_cast<int>(class_storage_.size()));
    assert(my_allocation_ < static_cast<int>(class_storage_.size()));
    assert(my_allocation_ >= 0);

    if (allocated_size_storage_ < size_storage_)
    {
        std::cerr << "ERROR BlockVector: trying to use allocation "
                  << size_storage_ << " bigger than initialy preallocated "
                  << allocated_size_storage_ << "!!!" << std::endl;
        mgmol_abort();
    }
    storage_ = class_storage_[my_allocation_];
    assert(class_storage_.size() > 0);

    assert(storage_ != nullptr);
}

// free slot of memory allocated to particular object
template <typename ScalarType, typename MemorySpaceType>
void BlockVector<ScalarType, MemorySpaceType>::deallocate_storage()
{
    if (my_allocation_ >= 0)
    {
        assert(storage_ != nullptr);
        assert(my_allocation_ >= 0);
        allocated_[my_allocation_] = 0;
        my_allocation_             = -1;
        storage_                   = nullptr;
    }
}

template <typename ScalarType, typename MemorySpaceType>
void BlockVector<ScalarType, MemorySpaceType>::setup(const BlockVector& bv)
{
    storage_       = nullptr;
    my_allocation_ = -1;
    for (short i = 0; i < 3; i++)
        bc_[i] = bv.bc_[i];
    allocate_storage();

    // set vect_
    vect_.resize(bv.vect_.size());
    for (unsigned int i = 0; i < vect_.size(); i++)
    {
        assert(i * ld_ < static_cast<unsigned int>(size_storage_));
        vect_[i] = storage_ + i * ld_;
    }
}

template <typename ScalarType, typename MemorySpaceType>
BlockVector<ScalarType, MemorySpaceType>::BlockVector(
    const BlockVector<ScalarType, MemorySpaceType>& bv, const bool copy_data)
    : size_storage_instance_(bv.size_storage_instance_),
      ld_instance_(bv.ld_instance_),
      mygrid_(bv.mygrid_)
{
    n_instances_++;

    set_ld_and_size_storage();

    setup(bv);

    if (copy_data) copyDataFrom(bv);
}

template <typename ScalarType, typename MemorySpaceType>
BlockVector<ScalarType, MemorySpaceType>&
BlockVector<ScalarType, MemorySpaceType>::operator=(
    const BlockVector<ScalarType, MemorySpaceType>& bv)
{
    if (this == &bv) return *this;

    n_instances_++;

    setup(bv);

    copyDataFrom(bv);

    return *this;
}

template <typename ScalarType, typename MemorySpaceType>
BlockVector<ScalarType, MemorySpaceType>&
BlockVector<ScalarType, MemorySpaceType>::operator-=(
    const BlockVector<ScalarType, MemorySpaceType>& src)
{
    opminus_tm_.start();

    for (unsigned int i = 0; i < vect_.size(); i++)
    {
        ScalarType* vi             = vect_[i];
        ScalarType const* const si = src.vect_[i];
        BV<ScalarType, MemorySpaceType>::subtract(numel_, si, vi);
    }

    opminus_tm_.stop();

    return *this;
}

template <typename ScalarType, typename MemorySpaceType>
template <typename ScalarType2>
void BlockVector<ScalarType, MemorySpaceType>::assign(
    const pb::GridFuncVector<ScalarType2, MemorySpaceType>& src)
{
    assign_tm_.start();

    for (unsigned int i = 0; i < vect_.size(); i++)
    {
        ScalarType* dest = vect_[i];
        src.template getValues<ScalarType>(i, dest);
    }

    assign_tm_.stop();
}

template <typename ScalarType, typename MemorySpaceType>
template <typename ScalarType2>
void BlockVector<ScalarType, MemorySpaceType>::assignComponent(
    const pb::GridFunc<ScalarType2>& src, const int i)
{
    ScalarType* dest = vect_[i];
    src.template getValues<ScalarType, MemorySpaceType>(dest);
}

template <typename ScalarType, typename MemorySpaceType>
template <typename ScalarType2>
void BlockVector<ScalarType, MemorySpaceType>::assignComponent(
    const pb::GridFuncVector<ScalarType2, MemorySpaceType>& src, const int i)
{
    ScalarType* dest = vect_[i];
    src.template getValues<ScalarType>(i, dest);
}

template <typename ScalarType, typename MemorySpaceType>
void BlockVector<ScalarType, MemorySpaceType>::initialize(
    const std::vector<std::vector<int>>& gid, const bool skinny_stencil)
{
    assert(storage_ == nullptr);

    int nbvect = static_cast<int>(gid[0].size());

    // set minimum number of vectors to take care of case of empty subdomains
    const short nvec       = nbvect > 20 ? nbvect : 20;
    size_storage_          = ld_ * nvec;
    size_storage_instance_ = size_storage_;

    allocate_storage();
    MemorySpace::Memory<ScalarType, MemorySpaceType>::set(
        storage_, size_storage_, 0);

    vect_.resize(nbvect);
    for (int i = 0; i < nbvect; i++)
    {
        vect_[i] = storage_ + i * ld_;
    }

    // rebuild data_wghosts_ with new gid
    if (data_wghosts_ != nullptr)
    {
        delete data_wghosts_;
        data_wghosts_ = nullptr;
    }

    data_wghosts_ = new pb::GridFuncVector<ScalarType, MemorySpaceType>(
        mygrid_, bc_[0], bc_[1], bc_[2], gid, skinny_stencil);

    data_wghosts_->resetData();

    data_wghosts_->set_updated_boundaries(false);
}

template <typename ScalarType, typename MemorySpaceType>
double BlockVector<ScalarType, MemorySpaceType>::dot(
    const int i, const int j, const short iloc) const
{
    assert(i < static_cast<int>(vect_.size()));
    assert(j < static_cast<int>(vect_.size()));
    assert(iloc < subdivx_);
    assert(vect_[i] != nullptr);
    assert(vect_[j] != nullptr);

    const int shift = iloc * locnumel_;
    return LinearAlgebraUtils<MemorySpaceType>::MPdot(
        locnumel_, vect_[i] + shift, vect_[j] + shift);
}

template <typename ScalarType, typename MemorySpaceType>
void BlockVector<ScalarType, MemorySpaceType>::scal(
    const int i, const double alpha)
{
    assert(i < static_cast<int>(vect_.size()));
    assert(vect_[i] != nullptr);

    LinearAlgebraUtils<MemorySpaceType>::MPscal(ld_, alpha, vect_[i]);
}

template <typename ScalarType, typename MemorySpaceType>
void BlockVector<ScalarType, MemorySpaceType>::scal(const double alpha)
{
    assert(storage_ != nullptr);

    scal_tm_.start();

    LinearAlgebraUtils<MemorySpaceType>::MPscal(size_storage_, alpha, storage_);

    scal_tm_.stop();
}

template <typename ScalarType, typename MemorySpaceType>
void BlockVector<ScalarType, MemorySpaceType>::scal(
    const double alpha, const int i, const short iloc)
{
    assert(static_cast<unsigned int>(i) < vect_.size());
    assert(vect_[i] != nullptr);

    const int shift = iloc * locnumel_;
    LinearAlgebraUtils<MemorySpaceType>::MPscal(
        locnumel_, alpha, vect_[i] + shift);
}

template <typename ScalarType, typename MemorySpaceType>
void BlockVector<ScalarType, MemorySpaceType>::set_zero()
{
    MemorySpace::Memory<ScalarType, MemorySpaceType>::set(
        storage_, size_storage_, 0);
}

template <typename ScalarType, typename MemorySpaceType>
void BlockVector<ScalarType, MemorySpaceType>::set_zero(
    const int i, const short iloc)
{
    assert(i < static_cast<int>(vect_.size()));
    assert(iloc < subdivx_);
    MemorySpace::Memory<ScalarType, MemorySpaceType>::set(
        vect_[i] + iloc * locnumel_, locnumel_, 0);
}

template <typename ScalarType, typename MemorySpaceType>
void BlockVector<ScalarType, MemorySpaceType>::set_zero(const int i)
{
    assert(i < static_cast<int>(vect_.size()));

    for (short iloc = 0; iloc < subdivx_; iloc++)
        MemorySpace::Memory<ScalarType, MemorySpaceType>::set(
            vect_[i] + iloc * locnumel_, locnumel_, 0);
}

template <typename ScalarType, typename MemorySpaceType>
void BlockVector<ScalarType, MemorySpaceType>::axpy(
    const double alpha, const int ix, const int iy, const short iloc)
{
    assert(static_cast<unsigned int>(ix) < vect_.size());
    assert(static_cast<unsigned int>(iy) < vect_.size());
    assert(vect_[ix] != nullptr);
    assert(vect_[iy] != nullptr);
    assert(vect_[ix] != vect_[iy]);

    const int shift = iloc * locnumel_;

    LinearAlgebraUtils<MemorySpaceType>::MPaxpy(
        locnumel_, alpha, vect_[ix] + shift, vect_[iy] + shift);
}

template <typename ScalarType, typename MemorySpaceType>
void BlockVector<ScalarType, MemorySpaceType>::axpy(const double alpha,
    BlockVector<ScalarType, MemorySpaceType>& bv, const int ix, const int iy,
    const short iloc)
{
    assert(ix < static_cast<int>(bv.vect_.size()));
    assert(iy < static_cast<int>(vect_.size()));
    assert(bv.vect_[ix] != vect_[iy]);

    const int shift = iloc * locnumel_;

    LinearAlgebraUtils<MemorySpaceType>::MPaxpy(
        locnumel_, alpha, bv.vect_[ix] + shift, vect_[iy] + shift);
}

template <typename ScalarType, typename MemorySpaceType>
void BlockVector<ScalarType, MemorySpaceType>::applyDiagonalOp(
    const std::vector<double>& diag,
    BlockVector<ScalarType, MemorySpaceType>& dst) const
{
    diagop_tm_.start();

    const double* const dd = diag.data();

    for (unsigned int j = 0; j < vect_.size(); j++)
    {
        const ScalarType* __restrict__ srcj = vect_[j];
        ScalarType* __restrict__ dstj       = dst.vect_[j];
        for (int i = 0; i < numel_; i++)
            dstj[i] = (ScalarType)(dd[i] * (double)srcj[i]);
    }

    diagop_tm_.stop();
}

template <typename ScalarType, typename MemorySpaceType>
void BlockVector<ScalarType, MemorySpaceType>::hasnan(const int j) const
{
    for (int i = 0; i < ld_; i++)
    {
        if (std::isnan(vect_[j][i]))
            (*MPIdata::sout) << "BlockVector: Nan in column " << j << ", row "
                             << i << std::endl;
    }
}
template <typename ScalarType, typename MemorySpaceType>
void BlockVector<ScalarType, MemorySpaceType>::setDataWithGhosts()
{
    // set internal data
    setDataWithGhosts(data_wghosts_);
}

template <typename ScalarType, typename MemorySpaceType>
template <typename ScalarType2>
void BlockVector<ScalarType, MemorySpaceType>::setDataWithGhosts(
    pb::GridFuncVector<ScalarType2, MemorySpaceType>* data_wghosts)
{
    assert(data_wghosts != nullptr);

    set_data_tm_.start();

    data_wghosts->set_updated_boundaries(false);

    // get number of mesh points
    const int numpts = mygrid_.size();

    for (unsigned int i = 0; i < vect_.size(); i++)
    {
        ScalarType* ivect_host_view = MemorySpace::Memory<ScalarType,
            MemorySpaceType>::allocate_host_view(numpts);
        MemorySpace::Memory<ScalarType, MemorySpaceType>::copy_view_to_host(
            vect_[i], numpts, ivect_host_view);

        data_wghosts->assign(i, ivect_host_view);

        MemorySpace::Memory<ScalarType, MemorySpaceType>::free_host_view(
            ivect_host_view);
    }

    set_data_tm_.stop();
}

template <typename ScalarType, typename MemorySpaceType>
void BlockVector<ScalarType, MemorySpaceType>::printTimers(std::ostream& os)
{
    set_data_tm_.print(os);
    trade_data_tm_.print(os);
    assign_tm_.print(os);
    scal_tm_.print(os);
    opminus_tm_.print(os);
    copy_tm_.print(os);
    diagop_tm_.print(os);
}

template <typename ScalarType, typename MemorySpaceType>
ScalarType BlockVector<ScalarType, MemorySpaceType>::maxAbsValue() const
{
    int ione = 1;
    // TODO
    MemorySpace::assert_is_host_ptr(storage_);
    int imax        = Tiamax(&size_storage_, storage_, &ione);
    ScalarType maxv = fabs(storage_[imax - 1]);
    MGmol_MPI& mmpi(*(MGmol_MPI::instance()));
    mmpi.allreduce(&maxv, 1, MPI_MAX);
    return maxv;
}

template <typename ScalarType, typename MemorySpaceType>
void BlockVector<ScalarType, MemorySpaceType>::set_ld_and_size_storage()
{
    size_storage_ = size_storage_instance_;
    ld_           = ld_instance_;
}

template class BlockVector<double, MemorySpace::Host>;
template void BlockVector<double, MemorySpace::Host>::assign(
    const pb::GridFuncVector<float, MemorySpace::Host>& src);
template void BlockVector<double, MemorySpace::Host>::assign(
    const pb::GridFuncVector<double, MemorySpace::Host>& src);
template void BlockVector<double, MemorySpace::Host>::assignComponent(
    const pb::GridFunc<float>& src, const int i);
template void BlockVector<double, MemorySpace::Host>::assignComponent(
    const pb::GridFunc<double>& src, const int i);
template void BlockVector<double, MemorySpace::Host>::setDataWithGhosts(
    pb::GridFuncVector<float, MemorySpace::Host>* data_wghosts);
template void BlockVector<double, MemorySpace::Host>::setDataWithGhosts(
    pb::GridFuncVector<double, MemorySpace::Host>* data_wghosts);
#ifdef MGMOL_USE_MIXEDP
template class BlockVector<float, MemorySpace::Host>;
template void BlockVector<float, MemorySpace::Host>::assign(
    const pb::GridFuncVector<float, MemorySpace::Host>& src);
template void BlockVector<float, MemorySpace::Host>::assign(
    const pb::GridFuncVector<double, MemorySpace::Host>& src);
template void BlockVector<float, MemorySpace::Host>::assignComponent(
    const pb::GridFunc<float>& src, const int i);
template void BlockVector<float, MemorySpace::Host>::assignComponent(
    const pb::GridFunc<double>& src, const int i);
template void BlockVector<float, MemorySpace::Host>::setDataWithGhosts(
    pb::GridFuncVector<float, MemorySpace::Host>* data_wghosts);
template void BlockVector<float, MemorySpace::Host>::setDataWithGhosts(
    pb::GridFuncVector<double, MemorySpace::Host>* data_wghosts);
#endif

#ifdef HAVE_MAGMA
template class BlockVector<double, MemorySpace::Device>;
template void BlockVector<double, MemorySpace::Device>::assign(
    const pb::GridFuncVector<float, MemorySpace::Device>& src);
template void BlockVector<double, MemorySpace::Device>::assign(
    const pb::GridFuncVector<double, MemorySpace::Device>& src);
template void BlockVector<double, MemorySpace::Device>::assignComponent(
    const pb::GridFunc<float>& src, const int i);
template void BlockVector<double, MemorySpace::Device>::assignComponent(
    const pb::GridFunc<double>& src, const int i);
template void BlockVector<double, MemorySpace::Device>::setDataWithGhosts(
    pb::GridFuncVector<float, MemorySpace::Device>* data_wghosts);
template void BlockVector<double, MemorySpace::Device>::setDataWithGhosts(
    pb::GridFuncVector<double, MemorySpace::Device>* data_wghosts);
#ifdef MGMOL_USE_MIXEDP
template class BlockVector<float, MemorySpace::Device>;
template void BlockVector<float, MemorySpace::Device>::assign(
    const pb::GridFuncVector<float, MemorySpace::Device>& src);
template void BlockVector<float, MemorySpace::Device>::assign(
    const pb::GridFuncVector<double, MemorySpace::Device>& src);
template void BlockVector<float, MemorySpace::Device>::assignComponent(
    const pb::GridFunc<float>& src, const int i);
template void BlockVector<float, MemorySpace::Device>::assignComponent(
    const pb::GridFunc<double>& src, const int i);
template void BlockVector<float, MemorySpace::Device>::setDataWithGhosts(
    pb::GridFuncVector<float, MemorySpace::Device>* data_wghosts);
template void BlockVector<float, MemorySpace::Device>::setDataWithGhosts(
    pb::GridFuncVector<double, MemorySpace::Device>* data_wghosts);
#endif
#endif

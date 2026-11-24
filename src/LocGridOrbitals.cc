// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "global.h"

#include <mpi.h>

#include "ColoredRegions.h"
#include "Control.h"
#include "DotProductManagerFactory.h"
#include "FunctionsPacking.h"
#include "GridFunc.h"
#include "GridMask.h"
#include "HDFrestart.h"
#include "Laph4M.h"
#include "LocGridOrbitals.h"
#include "LocalizationRegions.h"
#include "MPIdata.h"
#include "Masks4Orbitals.h"
#include "MasksSet.h"
#include "Mesh.h"
#include "ProjectedMatrices.h"
#include "ReplicatedWorkSpace.h"
#include "SquareLocalMatrices.h"
#include "SubCell.h"
#include "VariableSizeMatrix.h"
#include "hdf_tools.h"
#include "lapack_c.h"
#include "memory_space.h"

#ifdef MGMOL_USE_SCALAPACK
#include "DistMatrix.h"
#include "LocalMatrices2DistMatrix.h"
#endif

#include <cmath>
#include <fstream>
#include <utility>
#include <vector>

#define ORBITAL_OCCUPATION 2.
std::string getDatasetName(const std::string& name, const int color);

template <typename ScalarType>
DotProductManager<LocGridOrbitals<ScalarType>>*
    LocGridOrbitals<ScalarType>::dotProductManager_
    = nullptr;

template <typename ScalarType>
Timer LocGridOrbitals<ScalarType>::get_dm_tm_(
    "LocGridOrbitals" + std::to_string(8 * sizeof(ScalarType)) + "::get_dm");
template <typename ScalarType>
Timer LocGridOrbitals<ScalarType>::matB_tm_(
    "LocGridOrbitals" + std::to_string(8 * sizeof(ScalarType)) + "::matB");
template <typename ScalarType>
Timer LocGridOrbitals<ScalarType>::invBmat_tm_(
    "LocGridOrbitals" + std::to_string(8 * sizeof(ScalarType)) + "::invBmat");
template <typename ScalarType>
Timer LocGridOrbitals<ScalarType>::overlap_tm_(
    "LocGridOrbitals" + std::to_string(8 * sizeof(ScalarType)) + "::overlap");
template <typename ScalarType>
Timer LocGridOrbitals<ScalarType>::dot_product_tm_(
    "LocGridOrbitals" + std::to_string(8 * sizeof(ScalarType))
    + "::dot_product");
template <typename ScalarType>
Timer LocGridOrbitals<ScalarType>::addDot_tm_(
    "LocGridOrbitals" + std::to_string(8 * sizeof(ScalarType)) + "::addDot");
template <typename ScalarType>
Timer LocGridOrbitals<ScalarType>::mask_tm_(
    "LocGridOrbitals" + std::to_string(8 * sizeof(ScalarType)) + "::mask");
template <typename ScalarType>
Timer LocGridOrbitals<ScalarType>::prod_matrix_tm_(
    "LocGridOrbitals" + std::to_string(8 * sizeof(ScalarType))
    + "::prod_matrix");
template <typename ScalarType>
Timer LocGridOrbitals<ScalarType>::assign_tm_(
    "LocGridOrbitals" + std::to_string(8 * sizeof(ScalarType)) + "::assign");
template <typename ScalarType>
Timer LocGridOrbitals<ScalarType>::normalize_tm_(
    "LocGridOrbitals" + std::to_string(8 * sizeof(ScalarType)) + "::normalize");
template <typename ScalarType>
Timer LocGridOrbitals<ScalarType>::axpy_tm_(
    "LocGridOrbitals" + std::to_string(8 * sizeof(ScalarType)) + "::axpy");

template <typename ScalarType>
LocGridOrbitals<ScalarType>::LocGridOrbitals(std::string name,
    const pb::Grid& my_grid, const short subdivx, const int numst,
    const short bc[3], ProjectedMatricesInterface* proj_matrices,
    std::shared_ptr<LocalizationRegions> lrs, MasksSet* masks,
    MasksSet* corrmasks, ClusterOrbitals* local_cluster, const bool setup_flag)
    : name_(std::move(name)),
      proj_matrices_(proj_matrices),
      local_cluster_(local_cluster),
      block_vector_(my_grid, subdivx, bc),
      grid_(my_grid),
      lrs_(lrs)
{
    // preconditions
    assert(subdivx > 0);
    assert(proj_matrices != nullptr);
    assert(lrs);

    for (short i = 0; i < 3; i++)
        assert(bc[i] == 0 || bc[i] == 1);
    assert(grid_.size() > 0);

    subdivx_   = subdivx;
    numst_     = numst;
    numpt_     = grid_.size();
    lda_       = block_vector_.getld();
    loc_numpt_ = numpt_ / subdivx_;

    chromatic_number_ = 0;

    assert(numst_ >= 0);

    gidToStorage_ = nullptr;

    overlapping_gids_.clear();

    if (masks && corrmasks)
        masks4orbitals_.reset(
            new Masks4Orbitals(masks, corrmasks, lrs->getOverlapGids()));

    if (setup_flag) setup(lrs);
}

template <typename ScalarType>
LocGridOrbitals<ScalarType>::~LocGridOrbitals()
{
    assert(proj_matrices_ != nullptr);
    assert(pack_);
    assert(gidToStorage_ != nullptr);

    // delete gidToStorage here. This is OK since it is not a shared data
    // else there would be a memory leak.
    delete gidToStorage_;
    gidToStorage_ = nullptr;
}

template <typename ScalarType>
LocGridOrbitals<ScalarType>::LocGridOrbitals(const std::string& name,
    const LocGridOrbitals<ScalarType>& A, const bool copy_data)
    : Orbitals(A, copy_data),
      name_(name),
      proj_matrices_(A.proj_matrices_),
      local_cluster_(A.local_cluster_),
      block_vector_(A.block_vector_, copy_data),
      masks4orbitals_(A.masks4orbitals_),
      grid_(A.grid_),
      lrs_(A.lrs_)
{
    assert(A.chromatic_number_ >= 0);
    assert(A.proj_matrices_ != nullptr);
    assert(A.lrs_);

    copySharedData(A);

    gidToStorage_ = nullptr;

    setGids2Storage();
}

template <typename ScalarType>
LocGridOrbitals<ScalarType>::LocGridOrbitals(const std::string& name,
    const LocGridOrbitals<ScalarType>& A,
    ProjectedMatricesInterface* proj_matrices, MasksSet* masks,
    MasksSet* corrmasks, const bool copy_data)
    : Orbitals(A, copy_data),
      name_(name),
      proj_matrices_(proj_matrices),
      local_cluster_(A.local_cluster_),
      block_vector_(A.block_vector_, copy_data),
      grid_(A.grid_),
      lrs_(A.lrs_)
{
    assert(A.chromatic_number_ >= 0);
    assert(proj_matrices != nullptr);
    assert(masks != nullptr);
    assert(lrs_);

    copySharedData(A);

    gidToStorage_ = nullptr;

    setGids2Storage();

    masks4orbitals_.reset(
        new Masks4Orbitals(masks, corrmasks, A.all_overlapping_gids_));

    // setup new projected_matrices object
    proj_matrices_->setup(overlapping_gids_);
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::copySharedData(
    const LocGridOrbitals<ScalarType>& A)
{
    assert(A.gidToStorage_ != nullptr);
    assert(A.pack_);

    numst_ = A.numst_;

    lrs_iterative_index_ = A.lrs_iterative_index_;

    chromatic_number_ = A.chromatic_number_;

    pack_ = A.pack_;

    overlapping_gids_     = A.overlapping_gids_;
    all_overlapping_gids_ = A.all_overlapping_gids_;
    local_gids_           = A.local_gids_;

    distributor_diagdotprod_ = A.distributor_diagdotprod_;
    distributor_normalize_   = A.distributor_normalize_;
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::copyDataFrom(
    const LocGridOrbitals<ScalarType>& src)
{
    assert(proj_matrices_ != nullptr);

    block_vector_.copyDataFrom(src.block_vector_);

    setIterativeIndex(src);
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::setDotProduct(const short dot_type)
{
    DotProductManagerFactory<LocGridOrbitals> factory;

    dotProductManager_ = factory.create(dot_type);

    assert(dotProductManager_ != nullptr);
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::setGids2Storage()
{
    assert(chromatic_number_ >= 0);
    assert(subdivx_ > 0);

    if (gidToStorage_ != nullptr)
        gidToStorage_->clear();
    else
        gidToStorage_ = new std::vector<std::map<int, ScalarType*>>();
    gidToStorage_->resize(subdivx_);
    for (short iloc = 0; iloc < subdivx_; iloc++)
    {
        std::map<int, ScalarType*>& gid2st((*gidToStorage_)[iloc]);
        for (int color = 0; color < chromatic_number_; color++)
        {
            const int gid = overlapping_gids_[iloc][color];
            if (gid != -1)
            {
                gid2st.insert(
                    std::pair<int, ScalarType*>(gid, getPsi(color, iloc)));
            }
        }
    }
}

// return pointer to const data
template <typename ScalarType>
const ScalarType* LocGridOrbitals<ScalarType>::getGidStorage(
    const int gid, const short iloc) const
{
    assert(numst_ >= 0);
    assert(iloc < subdivx_);
    assert(iloc >= 0);
    assert(gid < numst_);
    assert(iloc < (short)gidToStorage_->size());

    auto p = (*gidToStorage_)[iloc].find(gid);
    if (p != (*gidToStorage_)[iloc].end())
        return p->second;
    else
        return nullptr;
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::setup(MasksSet* masks, MasksSet* corrmasks,
    std::shared_ptr<LocalizationRegions> lrs)
{
    assert(masks != nullptr);
    Control& ct = *(Control::instance());
    if (ct.verbose > 0)
        printWithTimeStamp("LocGridOrbitals::setup(MasksSet*, MasksSet*)...",
            (*MPIdata::sout));

    masks4orbitals_.reset(
        new Masks4Orbitals(masks, corrmasks, lrs->getOverlapGids()));

    setup(lrs);
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::setup(
    std::shared_ptr<LocalizationRegions> lrs)
{
    Control& ct = *(Control::instance());

    // preconditions
    assert(lrs);
    assert(proj_matrices_ != nullptr);

    if (ct.verbose > 0)
        printWithTimeStamp("LocGridOrbitals::setup()...", (*MPIdata::sout));

    lrs_iterative_index_ = lrs->getIterativeIndex();

    overlapping_gids_.clear();

    chromatic_number_ = packStates(lrs);

    computeGlobalIndexes(lrs);

    bool skinny_stencil = !ct.Mehrstellen();

    block_vector_.initialize(overlapping_gids_, skinny_stencil);

    setGids2Storage();

    proj_matrices_->setup(overlapping_gids_);

    Mesh* mymesh             = Mesh::instance();
    const pb::Grid& mygrid   = mymesh->grid();
    const pb::PEenv& myPEenv = mymesh->peenv();
    double domain[3]         = { mygrid.ll(0), mygrid.ll(1), mygrid.ll(2) };

    double maxr = lrs->max_radii();
    distributor_diagdotprod_.reset(
        new DataDistribution("dot", maxr, myPEenv, domain));
    distributor_normalize_.reset(
        new DataDistribution("norm", 2. * maxr, myPEenv, domain));

    if (ct.verbose > 0)
        printWithTimeStamp(
            "LocGridOrbitals::setup() done...", (*MPIdata::sout));
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::reset(MasksSet* masks, MasksSet* corrmasks,
    std::shared_ptr<LocalizationRegions> lrs)
{
    // free some old data
    block_vector_.clear();
    setIterativeIndex(-10);

    // reset
    setup(masks, corrmasks, lrs);
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::assign(
    const LocGridOrbitals<ScalarType>& orbitals)
{
    assign_tm_.start();

#ifdef DEBUG
    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    mmpi.barrier();
#endif
    assert(chromatic_number_ >= 0);
    assert(proj_matrices_ != nullptr);

    setIterativeIndex(orbitals);

    if (pack_ == orbitals.pack_)
    {

        block_vector_.copyDataFrom(orbitals.block_vector_);
    }
    else
    {
        Control& ct = *(Control::instance());
        if (onpe0 && ct.verbose > 2)
            (*MPIdata::sout) << "LocGridOrbitals::Assign orbitals "
                                "to different LR"
                             << std::endl;
        for (int color = 0; color < chromatic_number_; color++)
        {
            // assign state
            for (short iloc = 0; iloc < subdivx_; iloc++)
            {
                const int gid = overlapping_gids_[iloc][color];
                if (gid != -1)
                {
                    // find storage location in orbitals
                    const ScalarType* const val
                        = orbitals.getGidStorage(gid, iloc);
                    // copy into new psi_
                    if (val != nullptr)
                    {
                        block_vector_.assignLocal(color, iloc, val);
                    }
                }
            }
        }
    }

    assign_tm_.stop();
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::axpy(
    const double alpha, const LocGridOrbitals<ScalarType>& orbitals)
{
    axpy_tm_.start();

    assert(pack_ != nullptr);
    assert(orbitals.pack_ != nullptr);
    assert(overlapping_gids_.size() > 0);

    //    int ione=1;

    if (pack_ == orbitals.pack_)
    {
        block_vector_.axpy(alpha, orbitals.block_vector_);
    }
    else
    {
        for (int color = 0; color < chromatic_number_; color++)
        {
            // assign state
            for (short iloc = 0; iloc < subdivx_; iloc++)
            {
                const int gid = overlapping_gids_[iloc][color];
                assert(gid < numst_);
                if (gid != -1)
                {
                    // find orbital storage in orbitals
                    const ScalarType* const val
                        = orbitals.getGidStorage(gid, iloc);
                    // copy into new psi_
                    if (val != nullptr)
                    {
                        LinearAlgebraUtils<memory_space_type>::MPaxpy(
                            loc_numpt_, alpha, val, getPsi(color, iloc));
                    }
                }
            }
        }
    }
    incrementIterativeIndex();

    axpy_tm_.stop();
}

template <typename ScalarType>
short LocGridOrbitals<ScalarType>::checkOverlap(
    const int st1, const int st2, const short level)
{
    assert(masks4orbitals_);

    return masks4orbitals_->checkOverlap(st1, st2, level);
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::applyMask(const bool first_time)
{
    assert(chromatic_number_ >= 0);
    assert(subdivx_ > 0);
    assert(loc_numpt_ > 0);

    mask_tm_.start();

    const short ln = 0;
    for (int color = 0; color < chromatic_number_; color++)
        for (short iloc = 0; iloc < subdivx_; iloc++)
        {
            const int gid = overlapping_gids_[iloc][color];
            if (gid != -1)
            {
                (masks4orbitals_->getMask(gid))
                    .apply(psi(color), ln, iloc, first_time);
            }
            else
                block_vector_.set_zero(color, iloc);
        }

    incrementIterativeIndex();

    mask_tm_.stop();
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::applyCorrMask(const bool first_time)
{
    mask_tm_.start();

    for (int color = 0; color < chromatic_number_; color++)
    {
        const unsigned int size    = block_vector_.get_allocated_size_storage();
        ScalarType* ipsi_host_view = MemorySpace::Memory<ScalarType,
            memory_space_type>::allocate_host_view(size);
        MemorySpace::Memory<ScalarType, memory_space_type>::copy_view_to_host(
            psi(color), size, ipsi_host_view);

        for (short iloc = 0; iloc < subdivx_; iloc++)
        {
            const int gid = overlapping_gids_[iloc][color];
            if (gid != -1)
            {
                (masks4orbitals_->getCorrMask(gid))
                    .apply(ipsi_host_view, 0, iloc, first_time);
            }
            else
                block_vector_.set_zero(color, iloc);
        }
        MemorySpace::Memory<ScalarType, memory_space_type>::copy_view_to_dev(
            ipsi_host_view, size, psi(color));
        MemorySpace::Memory<ScalarType, memory_space_type>::free_host_view(
            ipsi_host_view);
    }
    incrementIterativeIndex();

    mask_tm_.stop();
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::app_mask(
    const int color, ScalarType* u, const short level) const
{
    mask_tm_.start();
    assert(masks4orbitals_);

    assert(color < chromatic_number_);
    int lnumpt = (loc_numpt_ >> level);
    lnumpt     = (lnumpt >> level);
    lnumpt     = (lnumpt >> level);

    for (short iloc = 0; iloc < subdivx_; iloc++)
    {
        int gid = overlapping_gids_[iloc][color];
        if (gid != -1)
        {
            (masks4orbitals_->getMask(gid)).apply(u, level, iloc);
        }
        else
            memset(u + iloc * lnumpt, 0, lnumpt * sizeof(ScalarType));
    }
    mask_tm_.stop();
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::app_mask(
    const int color, pb::GridFunc<ScalarType>& gu, const short level) const
{
    mask_tm_.start();

    assert(color < chromatic_number_);
    assert(masks4orbitals_);

    const short shift = gu.grid().ghost_pt();
    const int dim0    = gu.grid().dim(0) / subdivx_;
    const int incx    = gu.grid().inc(0);

    const int lnumpt = dim0 * incx;

    for (short iloc = 0; iloc < subdivx_; iloc++)
    {
        const int gid = overlapping_gids_[iloc][color];

        if (gid != -1)
        {
            (masks4orbitals_->getMask(gid)).apply(gu, level, iloc);
        }
        else
        {
            int offset = (shift + dim0 * iloc) * incx;
            assert(offset + lnumpt < static_cast<int>(gu.grid().sizeg()));
            ScalarType* pu = gu.uu() + offset;
            memset(pu, 0, lnumpt * sizeof(ScalarType));
        }
    }
    mask_tm_.stop();
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::init2zero()
{
    for (int icolor = 0; icolor < chromatic_number_; icolor++)
    {
        ScalarType* ipsi = psi(icolor);
        memset(ipsi, 0, numpt_ * sizeof(ScalarType));
    }
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::initGauss(
    const double rc, const std::shared_ptr<LocalizationRegions> lrs)
{
    assert(chromatic_number_ >= 0);
    assert(subdivx_ > 0);

    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    Control& ct     = *(Control::instance());
    if (mmpi.instancePE0() && ct.verbose > 2)
        (*MPIdata::sout) << "Initial orbitals: Gaussians of width " << rc
                         << std::endl;

    const double invrc2 = 1. / (rc * rc);

    const double start0 = grid_.start(0);
    const double start1 = grid_.start(1);
    const double start2 = grid_.start(2);

    const int dim0 = grid_.dim(0) / subdivx_;
    const int dim1 = grid_.dim(1);
    const int dim2 = grid_.dim(2);

    const int incx = dim1 * dim2;
    const int incy = dim2;

    const double hgrid[3] = { grid_.hgrid(0), grid_.hgrid(2), grid_.hgrid(2) };

    Vector3D ll;
    for (short i = 0; i < 3; i++)
        ll[i] = grid_.ll(i);

    const double rmax = 6. * rc;
    for (int icolor = 0; icolor < chromatic_number_; icolor++)
    {
        const unsigned int size    = numpt_;
        ScalarType* ipsi_host_view = MemorySpace::Memory<ScalarType,
            memory_space_type>::allocate_host_view(size);
        MemorySpace::Memory<ScalarType, memory_space_type>::copy_view_to_host(
            psi(icolor), size, ipsi_host_view);

        memset(ipsi_host_view, 0, numpt_ * sizeof(ScalarType));

        for (short iloc = 0; iloc < subdivx_; iloc++)
        {
            const int gid = overlapping_gids_[iloc][icolor];
            if (gid > -1)
            {
                const Vector3D& center(lrs->getCenter(gid));
                Vector3D xc;

                xc[0] = start0 + iloc * dim0 * hgrid[0];
                for (int ix = iloc * dim0; ix < (iloc + 1) * dim0; ix++)
                {
                    xc[1] = start1;

                    for (int iy = 0; iy < dim1; iy++)
                    {
                        xc[2] = start2;
                        for (int iz = 0; iz < dim2; iz++)
                        {
                            const double r = xc.minimage(center, ll, ct.bcWF);
                            if (r < rmax)
                                ipsi_host_view[ix * incx + iy * incy + iz]
                                    = (ScalarType)exp(-r * r * invrc2);
                            else
                                ipsi_host_view[ix * incx + iy * incy + iz] = 0.;

                            xc[2] += hgrid[2];
                        }
                        xc[1] += hgrid[1];
                    }
                    xc[0] += hgrid[0];
                }
            }
        }
        MemorySpace::Memory<ScalarType, memory_space_type>::copy_view_to_dev(
            ipsi_host_view, size, psi(icolor));
        MemorySpace::Memory<ScalarType, memory_space_type>::free_host_view(
            ipsi_host_view);
    }
    resetIterativeIndex();
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::initFourier()
{
    Control& ct = *(Control::instance());
    if (onpe0 && ct.verbose > 2)
        (*MPIdata::sout) << "Initial orbitals: Fourier " << std::endl;

    const double start0 = grid_.start(0) - grid_.origin(0);
    const double start1 = grid_.start(1) - grid_.origin(1);
    const double start2 = grid_.start(2) - grid_.origin(2);

    const int dim0 = grid_.dim(0) / subdivx_;
    const int dim1 = grid_.dim(1);
    const int dim2 = grid_.dim(2);

    const int incx = dim1 * dim2;
    const int incy = dim2;

    const double hgrid[3] = { grid_.hgrid(0), grid_.hgrid(1), grid_.hgrid(2) };

    Vector3D ll;
    for (short i = 0; i < 3; i++)
        ll[i] = grid_.ll(i);

    const double dk[3]
        = { 2. * M_PI / ll[0], 2. * M_PI / ll[1], 2. * M_PI / ll[2] };

    const int cbrtncolors = (int)ceil(cbrt(chromatic_number_));

    for (int icolor = 0; icolor < chromatic_number_; icolor++)
    {
        int kvector[3];
        getkvector(icolor + 1, cbrtncolors, kvector);

        const double kk[3] = { dk[0] * (double)kvector[0],
            dk[1] * (double)kvector[1], dk[2] * (double)kvector[2] };

        ScalarType* ipsi = psi(icolor);
        memset(ipsi, 0, numpt_ * sizeof(ScalarType));

        for (short iloc = 0; iloc < subdivx_; iloc++)
        {

            const int gid = overlapping_gids_[iloc][icolor];
            if (gid > -1)
            {
                double x = start0 + iloc * dim0 * hgrid[0];
                for (int ix = iloc * dim0; ix < (iloc + 1) * dim0; ix++)
                {
                    double y = start1;

                    for (int iy = 0; iy < dim1; iy++)
                    {
                        double z = start2;
                        for (int iz = 0; iz < dim2; iz++)
                        {
                            ipsi[ix * incx + iy * incy + iz]
                                = 1.
                                  - (ScalarType)(cos(kk[0] * x) * cos(kk[1] * y)
                                                 * cos(kk[2] * z));

                            z += hgrid[2];
                        }
                        y += hgrid[1];
                    }
                    x += hgrid[0];
                }
            }
        }
    }
    resetIterativeIndex();
}

template <typename ScalarType>
int LocGridOrbitals<ScalarType>::packStates(
    std::shared_ptr<LocalizationRegions> lrs)
{
    assert(lrs);

    Control& ct = *(Control::instance());

    if (onpe0 && ct.verbose > 2)
        (*MPIdata::sout) << " PACK STATES " << 0 << " to " << numst_
                         << std::endl;

    // compute overlap for all the orbitals on all PEs
    const int dim = ct.globalColoring() ? numst_ : lrs->getNumOverlapGids();

    if (onpe0 && ct.verbose > 2)
        (*MPIdata::sout) << " PACK " << dim << " STATES" << std::endl;

    const bool global = ct.globalColoring();

    MGmol_MPI& mmpi(*(MGmol_MPI::instance()));

    pack_.reset(new FunctionsPacking(lrs, global, mmpi.commSameSpin()));

    assert(pack_->chromatic_number() < 100000);

    return pack_->chromatic_number();
}

#ifdef MGMOL_USE_SCALAPACK
template <typename ScalarType>
void LocGridOrbitals<ScalarType>::multiply_by_matrix(
    const dist_matrix::DistMatrix<DISTMATDTYPE>& dmatrix,
    ScalarType* const product, const int ldp)
{
    ReplicatedWorkSpace<DISTMATDTYPE>& wspace(
        ReplicatedWorkSpace<DISTMATDTYPE>::instance());
    DISTMATDTYPE* work_matrix = wspace.square_matrix();

    // build a local complete matrix from a distributed matrix
    dmatrix.allgather(work_matrix, numst_);

    multiply_by_matrix(0, chromatic_number_, work_matrix, product, ldp);
}
#endif

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::multiply_by_matrix(const int first_color,
    const int ncolors, const DISTMATDTYPE* const matrix, ScalarType* product,
    const int ldp) const
{
    prod_matrix_tm_.start();

    assert(ncolors > 0);
    assert((first_color + ncolors) <= chromatic_number_);
    assert(subdivx_ > 0);

    memset(product, 0, ldp * ncolors * sizeof(ScalarType));

    DISTMATDTYPE* matrix_local = new DISTMATDTYPE[chromatic_number_ * ncolors];

    // loop over subdomains
    for (short iloc = 0; iloc < subdivx_; iloc++)
    {
        // extract block corresponding to local indexes
        matrixToLocalMatrix(iloc, matrix, matrix_local, first_color, ncolors);

        // Compute product for subdomain iloc
        LinearAlgebraUtils<memory_space_type>::MPgemmNN(loc_numpt_, ncolors,
            chromatic_number_, 1., getPsi(0, iloc), lda_, matrix_local,
            chromatic_number_, 0., product + iloc * loc_numpt_, ldp);
    }

#ifdef DEBUG
    for (short iloc = 0; iloc < subdivx_; iloc++)
    {

        for (int i = 0; i < ncolors; i++)
            if (overlapping_gids_[iloc][first_color + i] == -1)
                assert(fabs(*(product + i * ldp + iloc * loc_numpt_
                              + loc_numpt_ / 2))
                       < 1.e-15);
    }
#endif

    delete[] matrix_local;

    prod_matrix_tm_.stop();
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::multiplyByMatrix(
    const SquareLocalMatrices<MATDTYPE, MemorySpace::Host>& matrix,
    ScalarType* product, const int ldp) const
{
    prod_matrix_tm_.start();

    assert(subdivx_ > 0);

    // loop over subdomains
    if (chromatic_number_ > 0)
        for (short iloc = 0; iloc < subdivx_; iloc++)
        {
            const MATDTYPE* const mat = matrix.getSubMatrix(iloc);

#ifdef HAVE_MAGMA
            int const mat_size = matrix.m() * matrix.n();
            std::unique_ptr<MATDTYPE[], void (*)(MATDTYPE*)> mat_dev(
                MemorySpace::Memory<MATDTYPE, memory_space_type>::allocate(
                    mat_size),
                MemorySpace::Memory<MATDTYPE, memory_space_type>::free);
            MemorySpace::copy_to_dev<MATDTYPE>(mat, mat_size, mat_dev.get());
            const MATDTYPE* const mat_alias = mat_dev.get();

#else
            const MATDTYPE* const mat_alias = mat;
#endif

            // Compute product for subdomain iloc
            LinearAlgebraUtils<memory_space_type>::MPgemmNN(loc_numpt_,
                chromatic_number_, chromatic_number_, 1., getPsi(0, iloc), lda_,
                mat_alias, chromatic_number_, 0., product + iloc * loc_numpt_,
                ldp);
        }

    prod_matrix_tm_.stop();
}

// Here the result is stored in one of the matrices used in the multiplication,
// so a temporary arry is necessary
template <typename ScalarType>
void LocGridOrbitals<ScalarType>::multiplyByMatrix(
    const SquareLocalMatrices<MATDTYPE, MemorySpace::Host>& matrix)
{
    prod_matrix_tm_.start();

    if (chromatic_number_ > 0)
    {
        unsigned int const product_size = loc_numpt_ * chromatic_number_;
        std::unique_ptr<ScalarType[], void (*)(ScalarType*)> product(
            MemorySpace::Memory<ScalarType, memory_space_type>::allocate(
                product_size),
            MemorySpace::Memory<ScalarType, memory_space_type>::free);
        // We want to to use:
        // MemorySpace::Memory<ScalarType, memory_space_type>::set(
        //     product.get(), product_size, 0.);
        // but we get an error at linking time from nvptx-none-gcc
#ifdef HAVE_MAGMA
#ifdef HAVE_OPENMP_OFFLOAD
        ScalarType* tmp = product.get();
#pragma omp target teams distribute parallel for is_device_ptr(tmp)
        for (unsigned int i = 0; i < product_size; ++i)
            tmp[i] = 0;
#else
        ScalarType* product_host
            = MemorySpace::Memory<ScalarType, MemorySpace::Host>::allocate(
                product_size);
        std::memset(product_host, 0, product_size * sizeof(ScalarType));
        MemorySpace::copy_to_dev(product_host, product_size, product.get());
        MemorySpace::Memory<ScalarType, MemorySpace::Host>::free(product_host);
#endif
#else
        std::memset(product.get(), 0, product_size * sizeof(ScalarType));
#endif

        const size_t slnumpt = loc_numpt_ * sizeof(ScalarType);

        // loop over subdomains
        for (short iloc = 0; iloc < subdivx_; iloc++)
        {
            ScalarType* phi           = getPsi(0, iloc);
            const MATDTYPE* const mat = matrix.getSubMatrix(iloc);
#ifdef HAVE_MAGMA
            int const mat_size = matrix.m() * matrix.n();
            std::unique_ptr<MATDTYPE[], void (*)(MATDTYPE*)> mat_dev(
                MemorySpace::Memory<MATDTYPE, memory_space_type>::allocate(
                    mat_size),
                MemorySpace::Memory<MATDTYPE, memory_space_type>::free);
            MemorySpace::copy_to_dev<MATDTYPE>(mat, mat_size, mat_dev.get());
            const MATDTYPE* const mat_alias = mat_dev.get();

#else
            const MATDTYPE* const mat_alias = mat;
#endif

            // Compute product for subdomain iloc
            LinearAlgebraUtils<memory_space_type>::MPgemmNN(loc_numpt_,
                chromatic_number_, chromatic_number_, 1., phi, lda_, mat_alias,
                chromatic_number_, 0., product.get(), loc_numpt_);

            for (int color = 0; color < chromatic_number_; color++)
                MemorySpace::Memory<ScalarType, memory_space_type>::copy(
                    product.get() + color * loc_numpt_, slnumpt, phi + color);
        }
    }

    prod_matrix_tm_.stop();
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::multiplyByMatrix(
    const SquareLocalMatrices<MATDTYPE, MemorySpace::Host>& matrix,
    LocGridOrbitals<ScalarType>& product) const
{
    multiplyByMatrix(matrix, product.psi(0), product.lda_);
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::multiply_by_matrix(const int first_color,
    const int ncolors, const DISTMATDTYPE* const matrix,
    LocGridOrbitals<ScalarType>& product) const
{
    multiply_by_matrix(
        first_color, ncolors, matrix, product.psi(0), product.lda_);
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::multiply_by_matrix(
    const DISTMATDTYPE* const matrix,
    LocGridOrbitals<ScalarType>& product) const
{
    multiply_by_matrix(
        0, chromatic_number_, matrix, product.psi(0), product.lda_);
}

#ifdef MGMOL_USE_SCALAPACK
template <typename ScalarType>
void LocGridOrbitals<ScalarType>::multiply_by_matrix(
    const dist_matrix::DistMatrix<DISTMATDTYPE>& matrix, const double alpha,
    LocGridOrbitals<ScalarType>& product)
{
    prod_matrix_tm_.start();

    ScalarType* product_ptr
        = (this == &product) ? new ScalarType[loc_numpt_ * chromatic_number_]
                             : product.getPsi(0);

    ReplicatedWorkSpace<DISTMATDTYPE>& wspace(
        ReplicatedWorkSpace<DISTMATDTYPE>::instance());
    DISTMATDTYPE* work_matrix = wspace.square_matrix();

    matrix.allgather(work_matrix, numst_);

    DISTMATDTYPE* matrix_local
        = new DISTMATDTYPE[chromatic_number_ * chromatic_number_];

    const size_t slnumpt = loc_numpt_ * sizeof(ScalarType);

    // loop over subdomains
    for (short iloc = 0; iloc < subdivx_; iloc++)
    {
        ScalarType* phi = getPsi(0, iloc);

        matrixToLocalMatrix(iloc, work_matrix, matrix_local);

        // Compute loc_numpt_ rows (for subdomain iloc)
        LinearAlgebraUtils<MemorySpace::Host>::MPgemmNN(loc_numpt_,
            chromatic_number_, chromatic_number_, 1., phi, lda_, matrix_local,
            chromatic_number_, alpha, product_ptr, loc_numpt_);

        for (int color = 0; color < chromatic_number_; color++)
            memcpy(
                phi + color * lda_, product_ptr + color * loc_numpt_, slnumpt);
    }

    delete[] matrix_local;
    if (this == &product) delete[] product_ptr;

    prod_matrix_tm_.stop();
}
#endif

template <typename ScalarType>
int LocGridOrbitals<ScalarType>::read_hdf5(HDFrestart& h5f_file)
{
    assert(proj_matrices_ != nullptr);

    Control& ct = *(Control::instance());

    std::string name = "Function";
    int ierr         = read_func_hdf5(h5f_file, name);
    if (ierr < 0)
    {
        (*MPIdata::serr) << "LocGridOrbitals::read_hdf5(): error in reading "
                         << name << ", size=" << name.size() << std::endl;
        return ierr;
    }
    else if (onpe0 && ct.verbose > 2)
    {
        (*MPIdata::sout) << "LocGridOrbitals::read_hdf5(): Read " << ierr
                         << " functions in restart file" << std::endl;
    }

    // Read DM
    if (!ct.fullyOccupied())
    {
        ierr = proj_matrices_->readDM(h5f_file);
        if (ierr < 0)
        {
            (*MPIdata::serr) << "LocGridOrbitals::read_hdf5(): "
                                "error in reading DM"
                             << std::endl;
            return ierr;
        }
    }

    return ierr;
}

template <typename ScalarType>
int LocGridOrbitals<ScalarType>::write(
    HDFrestart& h5f_file, const std::string& name)
{
    Control& ct   = *(Control::instance());
    hid_t file_id = h5f_file.file_id();
    bool iwrite   = h5f_file.active();

    const bool global = ct.globalColoring();
    ColoredRegions colored_regions(*pack_, lrs_, global);

    // Create the dataspace for the dataset.

    hid_t filespace = -1;
    hid_t memspace  = -1;
    if (iwrite)
    {
        // filespace identifier
        filespace = h5f_file.createFilespace();

        // memory dataspace identifier
        memspace = h5f_file.createMemspace();
    }

    hid_t plist_id = h5f_file.createPlist();

    const short precision = ct.out_restart_info > 3 ? 2 : 1;

    if (onpe0 && ct.verbose > 2)
        (*MPIdata::sout) << "Write LocGridOrbitals<ScalarType> " << name
                         << " with precision " << precision << std::endl;
    // loop over global (storage) functions
    for (int color = 0; color < chromatic_number_; color++)
    {
        std::string datasetname(getDatasetName(name, color));
        if (onpe0 && ct.verbose > 2)
            (*MPIdata::sout) << "Write " << datasetname << std::endl;

        // Create chunked dataset.
        hid_t dset_id = -1;

        if (iwrite)
        {
            assert(file_id > -1);

            hid_t dtype_id = outHdfDataType(ct.out_restart_info);
            dset_id        = H5Dcreate2(file_id, datasetname.c_str(), dtype_id,
                       filespace, H5P_DEFAULT, plist_id, H5P_DEFAULT);
            if (dset_id < 0)
            {
                (*MPIdata::serr) << "LocGridOrbitals::write_func_hdf5(), "
                                    "H5Dcreate2 failed!!!"
                                 << std::endl;
                return -1;
            }
        }

        // WARNING:
        // Attributes (both the attribute information and the data it holds)
        // are considered to be metadata on an object.
        // The HDF library has a requirement that all metadata updates be done
        // collectively so all processes see the same stream of metadata
        // updates.

        // Write list of centers and radii
        // const int nst=pack_->nb_orb(color);

        std::vector<double> centers_and_radii;
        int nrec = 0;
#ifdef MGMOL_USE_HDF5P
        if (h5f_file.useHdf5p())
        {
            nrec = colored_regions.getAllCentersAndRadii4color(
                color, centers_and_radii);
        }
        else
#endif
        {
            nrec = colored_regions.getLocCentersAndRadii4color(
                color, centers_and_radii);
            assert(static_cast<int>(centers_and_radii.size()) == 4 * nrec);
        }

        std::vector<int> gids;
#ifdef MGMOL_USE_HDF5P
        if (h5f_file.useHdf5p())
        {
            colored_regions.getAllGids4color(color, gids);
        }
        else
#endif
        {
            colored_regions.getLocGids4color(color, gids);
        }

        if (iwrite)
        {
            writeListCentersAndRadii(dset_id, nrec, centers_and_radii);

            writeGids(dset_id, gids);

            // Write the attribute "Lattice parameters" at "Cell origin"
            std::string attname("Lattice parameters");

            // Create the data space for the attribute "Lattice parameters".
            std::vector<double> attr_data(3);
            attr_data[0] = grid_.ll(0);
            attr_data[1] = grid_.ll(1);
            attr_data[2] = grid_.ll(2);

            mgmol_tools::addAttribute2Dataset(
                dset_id, attname.c_str(), attr_data);

            attr_data[0] = grid_.origin(0);
            attr_data[1] = grid_.origin(1);
            attr_data[2] = grid_.origin(2);

            std::string attname2("Cell origin");
            mgmol_tools::addAttribute2Dataset(
                dset_id, attname2.c_str(), attr_data);
        } // iwrite

        int ierr = h5f_file.writeData(
            psi(color), filespace, memspace, dset_id, precision);
        if (ierr < 0) return ierr;

        // Close/release resources.
        if (iwrite)
        {
            herr_t status = H5Dclose(dset_id);
            if (status < 0)
            {
                (*MPIdata::serr) << "LocGridOrbitals::write_func_"
                                    "hdf5:H5Dclose failed!!!"
                                 << std::endl;
                return -1;
            }
        }

    } // loop over color

    h5f_file.releasePlist(plist_id);

    if (iwrite)
    {
        // close filespace and memspace
        herr_t status = H5Sclose(filespace);
        if (status < 0)
        {
            (*MPIdata::serr) << "H5Sclose filespace failed!!!" << std::endl;
        }
        status = H5Sclose(memspace);
        if (status < 0)
        {
            (*MPIdata::serr) << "H5Sclose memspace failed!!!" << std::endl;
        }
    }

    MGmol_MPI& mmpi(*(MGmol_MPI::instance()));
    mmpi.barrier();

    return 0;
}

template <typename ScalarType>
int LocGridOrbitals<ScalarType>::read_func_hdf5(
    HDFrestart& h5f_file, const std::string& name)
{
    assert(chromatic_number_ >= 0);
    assert(name.size() > 0);
    assert(pack_);

    Control& ct     = *(Control::instance());
    MGmol_MPI& mmpi = *(MGmol_MPI::instance());

    const bool global = ct.globalColoring();
    ColoredRegions colored_regions(*pack_, lrs_, global);

    hsize_t block[3] = { grid_.dim(0), grid_.dim(1), grid_.dim(2) };
    if (h5f_file.gatherDataX())
    {
        block[0] = grid_.gdim(0);
    }

    // Each process defines dataset in memory and writes it to the hyperslab
    // in the file.

    // memory dataspace identifier
    hid_t memspace = H5P_DEFAULT;
    if (h5f_file.active()) memspace = h5f_file.createMemspace();

    // freed automatically, including on the early error returns below
    std::unique_ptr<ScalarType[]> buffer_owner(
        new ScalarType[block[0] * block[1] * block[2]]);
    ScalarType* buffer = buffer_owner.get();

    if (onpe0 && ct.verbose > 2)
    {
        if (h5f_file.gatherDataX())
        {
            (*MPIdata::sout) << "LocGridOrbitals::read_func_hdf5(): Read wave "
                                "functions from "
                             << grid_.mype_env().n_mpi_task(1)
                                    * grid_.mype_env().n_mpi_task(2)
                             << " PEs" << std::endl;
        }
        else
        {
            (*MPIdata::sout) << "LocGridOrbitals::read_func_hdf5():"
                                " Read wave functions "
                             << name << " from all tasks..." << std::endl;
        }
    }

    std::vector<std::set<int>>
        filled; // set of functions already filled by data
    filled.resize(subdivx_);
    int dims[2] = { 0, 0 };

    // get centers corresponding to dataset (stored function) from input file
    std::multimap<std::string, Vector3D> centers_in_dataset;
    int ncenters = h5f_file.getLRCenters(centers_in_dataset, numst_, name);
    if (ncenters < 0) return ncenters;

    SubCell sub_cell(grid_, subdivx_, 0);
    const short precision = ct.restart_info > 3 ? 2 : 1;

    // read one color/dataset at a time
    std::multimap<std::string, Vector3D>::iterator itcenter
        = centers_in_dataset.begin();
    while (itcenter != centers_in_dataset.end())
    {
        std::vector<double> attr_data;
        short attribute_length = 0;

        const std::string key(itcenter->first);

        // checkif dataset exists...
        int err_id = h5f_file.checkDataExistsLocal(key);
        if (h5f_file.gatherDataX()) mmpi.bcast(&err_id, 1);
        if (err_id == 0) break; // dataset does not exists

        if (onpe0 && ct.verbose > 2)
            (*MPIdata::sout) << "Read Dataset " << key << " with precision "
                             << precision << std::endl;

        // Open dataset.
        hid_t dset_id = h5f_file.open_dset(key);
        if (dset_id < 0)
        {
            (*MPIdata::serr) << "LocGridOrbitals::read_func_hdf5() "
                                "--- cannot open "
                             << key << std::endl;
            return dset_id;
        }

        herr_t status = h5f_file.readData(buffer, memspace, dset_id, precision);
        if (status < 0)
        {
            (*MPIdata::serr) << "LocGridOrbitals::read_func_hdf5() "
                                "--- H5Dread failed!!!"
                             << std::endl;
            return -1;
        }

        if (h5f_file.active())
        {
            int natt = readListCentersAndRadii(dset_id, attr_data);
            assert(natt == static_cast<int>(centers_in_dataset.count(key)));

            if (natt < 0) return -1;
            dims[0]          = natt;
            dims[1]          = dims[0] > 0 ? attr_data.size() / dims[0] : 0;
            attribute_length = dims[1];
        }

        status = h5f_file.close_dset(dset_id);
        if (status < 0)
        {
            return status;
        }

        int intdims[2] = { dims[0], dims[1] };

        if (h5f_file.gatherDataX())
        {
            // Bcast size of data and data
            mmpi.bcast(intdims, 2);
            const int dim = intdims[0] * intdims[1];
            assert(dim > 0);
            assert(intdims[1] == 4);
            if (!mmpi.instancePE0()) attr_data.resize(dim);
            mmpi.bcast(&attr_data[0], dim);
        }

        // loop over centers just read
        for (int i = 0; i < intdims[0]; i++)
        {
            assert(attribute_length > 0);
            assert(attr_data.size()
                   > static_cast<unsigned int>(attribute_length * i));
            const Vector3D center(attr_data[attribute_length * i],
                attr_data[attribute_length * i + 1],
                attr_data[attribute_length * i + 2]);
            const double read_radius = attr_data[attribute_length * i + 3];

            // get possible colors for function centered at center
            std::set<int> possible_colors;
            colored_regions.getPossibleColors(center, possible_colors);
            if (possible_colors.size()
                > 0) // read center could not be local anymore...
                for (short iloc = 0; iloc < subdivx_; iloc++)
                {
                    // is this suddomain close enough to center to get data?
                    if (sub_cell.spherePossibleOverlap(
                            center, read_radius, iloc, ct.bcPoisson))
                    {
                        std::set<int> result;
                        // Difference of two sorted ranges
                        set_difference(possible_colors.begin(),
                            possible_colors.end(), filled[iloc].begin(),
                            filled[iloc].end(),
                            inserter(result, result.begin()));
                        if (result.size() > 0)
                        {
                            const int mycolor = (*result.begin());
                            assert(mycolor < chromatic_number_);

                            const int gid = overlapping_gids_[iloc][mycolor];
                            if (gid != -1) // gid may be -1 if no actual mesh
                                           // point is inside sphere!
                            {
                                if (masks4orbitals_->center(gid) == center)
                                {
                                    const int shift = iloc * loc_numpt_;
                                    block_vector_.assignLocal(
                                        mycolor, iloc, buffer + shift);
                                    filled[iloc].insert(mycolor);
                                    //(*MPIdata::sout)<<"Center "<<center<<",
                                    // radius "<<read_radius<<", iloc="<<iloc
                                }
                            } // gid!=-1
                        }
                        else
                        {
                            (*MPIdata::serr)
                                << "result.size()=" << result.size()
                                << ", possible_color="
                                << *possible_colors.begin() << "!!!"
                                << std::endl;
                        }
                    } // overlap
                }

            assert(itcenter != centers_in_dataset.upper_bound(key));

            itcenter++; // increment multimap
        }
        assert(itcenter == centers_in_dataset.upper_bound(key));

    } // end loop over centers_in_dataset

    resetIterativeIndex();

    if (h5f_file.active())
    {
        herr_t status = H5Sclose(memspace);
        if (status < 0)
        {
            (*MPIdata::serr) << "H5Sclose failed!!!" << std::endl;
            return -1;
        }
    }

    return centers_in_dataset.size();
}

// initialize matrix chromatic_number_ by ncolor (for columns first_color to
// first_color+ncolor)
template <typename ScalarType>
void LocGridOrbitals<ScalarType>::matrixToLocalMatrix(const short iloc,
    const DISTMATDTYPE* const matrix, DISTMATDTYPE* const lmatrix) const
{
    matrixToLocalMatrix(iloc, matrix, lmatrix, 0, chromatic_number_);
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::matrixToLocalMatrix(const short iloc,
    const DISTMATDTYPE* const matrix, DISTMATDTYPE* const lmatrix,
    const int first_color, const int ncolor) const
{
    assert(ncolor <= chromatic_number_);
    memset(lmatrix, 0, chromatic_number_ * ncolor * sizeof(DISTMATDTYPE));

    for (int jcolor = 0; jcolor < ncolor; jcolor++)
    {
        const int gidj = overlapping_gids_[iloc][first_color + jcolor];
        if (gidj != -1)
        {
            const int njst = gidj * numst_;
            const int njc  = jcolor * chromatic_number_;
            for (int icolor = 0; icolor < chromatic_number_; icolor++)
            {
                const int gidi = overlapping_gids_[iloc][icolor];
                if (gidi != -1)
                {
                    lmatrix[njc + icolor] = matrix[njst + gidi];
                }
            }
        }
    }
}

// compute the matrix <psi1|B|psi2>
// output: matB
template <typename ScalarType>
void LocGridOrbitals<ScalarType>::computeMatB(
    const LocGridOrbitals<ScalarType>& orbitals,
    const pb::Lap<ScalarType>& LapOper)
{
    if (numst_ == 0) return;

    assert(proj_matrices_ != nullptr);

    matB_tm_.start();
#if DEBUG
    if (onpe0)
        (*MPIdata::sout) << "LocGridOrbitals::computeMatB()" << std::endl;
#endif

    const short bcolor = 32;

    SquareLocalMatrices<MATDTYPE, MemorySpace::Host> ss(
        subdivx_, chromatic_number_);

    ScalarType* work = new ScalarType[lda_ * bcolor];
    memset(work, 0, lda_ * bcolor * sizeof(ScalarType));

    const ScalarType* const orbitals_psi
        = (chromatic_number_ > 0) ? orbitals.block_vector_.vect(0) : nullptr;

    setDataWithGhosts();
    trade_boundaries();

    for (int icolor = 0; icolor < chromatic_number_; icolor += bcolor)
    {
        int nf = bcolor;
        if ((icolor + nf) > chromatic_number_) nf = chromatic_number_ - icolor;

        // Compute nf columns of B|psi> and store it into work
        for (int i = 0; i < nf; i++)
        {
            LapOper.rhs(getFuncWithGhosts(icolor + i), work + i * lda_);
        }

        for (short iloc = 0; iloc < subdivx_; iloc++)
        {

            MATDTYPE* ssiloc = ss.getRawPtr(iloc);

            // calculate nf columns of ssiloc
            LinearAlgebraUtils<memory_space_type>::MPgemmTN(chromatic_number_,
                nf, loc_numpt_, 1., orbitals_psi + iloc * loc_numpt_, lda_,
                work + iloc * loc_numpt_, lda_, 0.,
                ssiloc + icolor * chromatic_number_, chromatic_number_);
        }
    }

    delete[] work;

    const double vel = grid_.vel();
    ss.scal(vel);
    proj_matrices_->initializeMatB(ss);

    matB_tm_.stop();
}

// compute <Phi|B|Phi> and its inverse
template <typename ScalarType>
void LocGridOrbitals<ScalarType>::computeBAndInvB(
    const pb::Lap<ScalarType>& LapOper)
{
    assert(proj_matrices_ != nullptr);

    Control& ct = *(Control::instance());
    if (!ct.Mehrstellen()) return;

    invBmat_tm_.start();

    computeMatB(*this, LapOper);
    proj_matrices_->computeInvB();

    invBmat_tm_.stop();
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::getLocalOverlap(
    SquareLocalMatrices<MATDTYPE, MemorySpace::Host>& ss)
{
    assert(chromatic_number_ >= 0);
    assert(loc_numpt_ > 0);
    assert(grid_.vel() > 1.e-8);
    assert(subdivx_ > 0);

    if (chromatic_number_ != 0)
    {
#ifdef MGMOL_USE_MIXEDP
        getLocalOverlap(*this, ss);
#else
        const ScalarType* const psi = block_vector_.vect(0);

        for (short iloc = 0; iloc < subdivx_; iloc++)
        {
            ss.syrk(iloc, loc_numpt_, psi + iloc * loc_numpt_, lda_);
        }

        // We may need the full matrix
        ss.fillUpperWithLower();

        ss.scal(grid_.vel());
#endif
    }
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::getLocalOverlap(
    const LocGridOrbitals<ScalarType>& orbitals,
    SquareLocalMatrices<MATDTYPE, MemorySpace::Host>& ss)
{
    assert(chromatic_number_ >= 0);

    if (chromatic_number_ != 0)
    {
        computeLocalProduct(
            orbitals.block_vector_.vect(0), orbitals.lda_, ss, false);
    }
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::computeLocalProduct(
    const LocGridOrbitals<ScalarType>& orbitals,
    LocalMatrices<MATDTYPE, MemorySpace::Host>& ss, const bool transpose)
{
    // assert( orbitals.chromatic_number_>=0 );
    assert(orbitals.lda_ > 1);

    if (chromatic_number_ != 0)
        computeLocalProduct(orbitals.psi(0), orbitals.lda_, ss, transpose);
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::computeLocalProduct(
    const ScalarType* const array, const int ld,
    LocalMatrices<MATDTYPE, MemorySpace::Host>& ss, const bool transpose)
{
    assert(loc_numpt_ > 0);
    assert(loc_numpt_ <= ld);
    assert(array != nullptr);
    assert(chromatic_number_ != 0);
    assert(grid_.vel() > 0.);
    assert(subdivx_ > 0);

    const ScalarType* const a = transpose ? array : block_vector_.vect(0);
    const ScalarType* const b = transpose ? block_vector_.vect(0) : array;

    const int lda = transpose ? ld : lda_;
    const int ldb = transpose ? lda_ : ld;

    unsigned int const a_size = numpt_ * ss.m();
    ScalarType* a_host_view   = MemorySpace::Memory<ScalarType,
        memory_space_type>::allocate_host_view(a_size);
    MemorySpace::Memory<ScalarType, memory_space_type>::copy_view_to_host(
        const_cast<ScalarType*>(a), a_size, a_host_view);
    unsigned int const b_size = numpt_ * ss.n();
    ScalarType* b_host_view   = MemorySpace::Memory<ScalarType,
        memory_space_type>::allocate_host_view(b_size);
    MemorySpace::Memory<ScalarType, memory_space_type>::copy_view_to_host(
        const_cast<ScalarType*>(b), b_size, b_host_view);

#ifdef MGMOL_USE_MIXEDP
    // use temporary float data for matrix ss
    LocalMatrices<ScalarType, MemorySpace::Host> ssf(ss.nmat(), ss.m(), ss.n());
#else
    LocalMatrices<ScalarType, MemorySpace::Host>& ssf(ss);
#endif
    for (short iloc = 0; iloc < subdivx_; iloc++)
    {
        ssf.gemm(iloc, loc_numpt_, a_host_view + iloc * loc_numpt_, lda,
            b_host_view + iloc * loc_numpt_, ldb);
    }
    MemorySpace::Memory<ScalarType, memory_space_type>::free_host_view(
        a_host_view);
    MemorySpace::Memory<ScalarType, memory_space_type>::free_host_view(
        b_host_view);
#ifdef MGMOL_USE_MIXEDP
    ss.copy(ssf);
#endif

    ss.scal(grid_.vel());
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::computeDiagonalElementsDotProduct(
    const LocGridOrbitals<ScalarType>& orbitals, std::vector<DISTMATDTYPE>& ss)
{
    assert(numst_ > 0);
    assert(grid_.vel() > 0.);

    memset(&ss[0], 0, numst_ * sizeof(DISTMATDTYPE));

    for (int icolor = 0; icolor < chromatic_number_; icolor++)
        for (short iloc = 0; iloc < subdivx_; iloc++)
        {
            const int gid = overlapping_gids_[iloc][icolor];
            if (gid > -1)
            {
                double alpha
                    = LinearAlgebraUtils<memory_space_type>::MPdot(loc_numpt_,
                        orbitals.getPsi(icolor, iloc), getPsi(icolor, iloc));

                ss[gid] += (DISTMATDTYPE)(alpha * grid_.vel());
            }
        }
    std::vector<DISTMATDTYPE> tmp(ss);
    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    mmpi.allreduce(&tmp[0], &ss[0], numst_, MPI_SUM);
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::computeDiagonalElementsDotProductLocal(
    const LocGridOrbitals<ScalarType>& orbitals, std::vector<DISTMATDTYPE>& ss)
{
    assert(grid_.vel() > 0.);

    /* get locally centered functions */
    std::vector<int> locfcns;
    if (local_cluster_ != nullptr)
    {
        locfcns = local_cluster_->getClusterIndices();
    }
    else
    {
        locfcns = local_gids_;
    }

    /* define data distribution variables */
    const int siz = (int)locfcns.size();
    VariableSizeMatrix<sparserow> diag("diagDot", chromatic_number_ * subdivx_);
    /* initialize sparse diagonal matrix with locally centered indixes */
    diag.setupSparseRows(locfcns);

    /* assemble diagonal matrix */
    //    int ione=1;
    for (int icolor = 0; icolor < chromatic_number_; icolor++)
        for (short iloc = 0; iloc < subdivx_; iloc++)
        {
            const int ifunc = overlapping_gids_[iloc][icolor];
            if (ifunc > -1)
            {
                double alpha
                    = LinearAlgebraUtils<memory_space_type>::MPdot(loc_numpt_,
                        orbitals.getPsi(icolor, iloc), getPsi(icolor, iloc));

                double val = alpha * grid_.vel();
                diag.insertMatrixElement(ifunc, ifunc, val, ADD, true);
            }
        }
    /* do data distribution to update local sums */
    (*distributor_diagdotprod_).augmentLocalData(diag, false);
    /* collect data */
    ss.clear();
    for (int row = 0; row < siz; row++)
    {
        ss.push_back((DISTMATDTYPE)diag.getRowEntry(row, 0));
    }
}

#ifdef MGMOL_USE_SCALAPACK
template <typename ScalarType>
void LocGridOrbitals<ScalarType>::computeGram(
    dist_matrix::DistMatrix<DISTMATDTYPE>& gram_mat)
{
    computeGram(*this, gram_mat);
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::computeGram(
    const LocGridOrbitals<ScalarType>& orbitals,
    dist_matrix::DistMatrix<DISTMATDTYPE>& gram_mat)
{
    SquareLocalMatrices<MATDTYPE, MemorySpace::Host> ss(
        subdivx_, chromatic_number_);

    getLocalOverlap(orbitals, ss);

    LocalMatrices2DistMatrix* sl2dm = LocalMatrices2DistMatrix::instance();

    gram_mat.clear();

    sl2dm->accumulate(ss, gram_mat);
}
#endif

// compute the lower-triangular part of the overlap matrix
template <typename ScalarType>
void LocGridOrbitals<ScalarType>::computeGram(const int verbosity)
{
    assert(proj_matrices_ != nullptr);

    // if( chromatic_number_==0 )return;

    overlap_tm_.start();

#ifdef PRINT_OPERATIONS
    if (onpe0)
        (*MPIdata::sout) << "LocGridOrbitals::computeGram()" << std::endl;
#endif

    assert(subdivx_ > 0);
    assert(subdivx_ < 1000);
    assert(chromatic_number_ >= 0);

    SquareLocalMatrices<MATDTYPE, MemorySpace::Host> ss(
        subdivx_, chromatic_number_);

    getLocalOverlap(ss);

    proj_matrices_->initializeGramMatrix(ss, getIterativeIndex());

    if (verbosity > 1) proj_matrices_->printS((*MPIdata::sout));

    overlap_tm_.stop();
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::computeGramAndInvS(const int verbosity)
{
    assert(proj_matrices_ != nullptr);

    computeGram(verbosity);

    /* Compute inverse of Gram matrix */
    proj_matrices_->computeInvS();
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::checkCond(
    const double tol, const bool flag_stop)
{
    assert(proj_matrices_ != nullptr);

    proj_matrices_->checkCond(tol, flag_stop);
}

template <typename ScalarType>
double LocGridOrbitals<ScalarType>::dotProduct(
    const LocGridOrbitals<ScalarType>& orbitals)
{
    assert(dotProductManager_ != nullptr);
    return dotProductManager_->dotProduct(*this, orbitals);
}

template <typename ScalarType>
double LocGridOrbitals<ScalarType>::dotProduct(
    const LocGridOrbitals<ScalarType>& orbitals, const short dot_type)
{
    dot_product_tm_.start();

    assert(chromatic_number_ >= 0);
    assert(subdivx_ > 0);
    assert(subdivx_ < 1000);

    DotProductManagerFactory<LocGridOrbitals> factory;
    DotProductManager<LocGridOrbitals>* manager = factory.create(dot_type);
    assert(manager != nullptr);

    double dot = manager->dotProduct(*this, orbitals);

    delete manager;

    dot_product_tm_.stop();

    return dot;
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::orthonormalizeLoewdin(
    const bool overlap_uptodate,
    SquareLocalMatrices<MATDTYPE, MemorySpace::Host>* matrixTransform,
    const bool update_matrices)
{
    Control& ct = *(Control::instance());
    if (onpe0 && ct.verbose > 1)
        (*MPIdata::sout) << "LocGridOrbitals::orthonormalizeLoewdin()"
                         << std::endl;

    if (!overlap_uptodate) computeGram(0);

    SquareLocalMatrices<MATDTYPE, MemorySpace::Host>* localP = matrixTransform;
    if (matrixTransform == nullptr)
        localP = new SquareLocalMatrices<MATDTYPE, MemorySpace::Host>(
            subdivx_, chromatic_number_);

    // try with ReplicatedMatrix first
    ProjectedMatrices<ReplicatedMatrix>* projmatrices
        = dynamic_cast<ProjectedMatrices<ReplicatedMatrix>*>(proj_matrices_);
    if (projmatrices)
    {
        projmatrices->computeLoewdinTransform(
            *localP, getIterativeIndex(), update_matrices);
    }
#ifdef MGMOL_USE_SCALAPACK
    else
    {
        ProjectedMatrices<dist_matrix::DistMatrix<DISTMATDTYPE>>* projmatrices
            = dynamic_cast<
                ProjectedMatrices<dist_matrix::DistMatrix<DISTMATDTYPE>>*>(
                proj_matrices_);
        assert(projmatrices != nullptr);
        assert(localP);
        projmatrices->computeLoewdinTransform(
            *localP, getIterativeIndex(), update_matrices);
    }
#endif

    multiplyByMatrix(*localP);

    incrementIterativeIndex();

#if 0 // test
    computeGram(0);
    if( onpe0 && ct.verbose>2 )
        (*MPIdata::sout)<<"LocGridOrbitals::orthonormalizeLoewdin() --- Gram matrix (after):"<<std::endl;
    proj_matrices_->printS(*MPIdata::sout);
#endif

    if (matrixTransform == nullptr) delete localP;
}

template <typename ScalarType>
double LocGridOrbitals<ScalarType>::norm() const
{
    Control& ct = *(Control::instance());

    double norm = 0;

    for (int gid = 0; gid < ct.numst; gid++)
    {
        norm += normState(gid);
    }
    return norm;
}

template <typename ScalarType>
double LocGridOrbitals<ScalarType>::normState(const int gid) const
{
    assert(gid >= 0);

    // find color of state
    int color_gid = -1;
    for (short iloc = 0; iloc < subdivx_; iloc++)
        for (int color = 0; color < chromatic_number_; color++)
        {
            const int pst = overlapping_gids_[iloc][color];
            if (pst == gid)
            {
                assert(color_gid == -1 || color_gid == color);
                color_gid = color;
            }
        }
    double tmp = 0.;
    if (color_gid >= 0)
        for (short iloc = 0; iloc < subdivx_; iloc++)
            if (overlapping_gids_[iloc][color_gid] == gid)
            {
                // diagonal element
                tmp += block_vector_.dot(color_gid, color_gid, iloc);
            }

    double norm     = 0.;
    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    mmpi.allreduce(&tmp, &norm, 1, MPI_SUM);

    return grid_.vel() * norm;
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::orthonormalize2states(
    const int st1, const int st2)
{
    assert(st1 >= 0);
    assert(st2 >= 0);

    Control& ct = *(Control::instance());
    if (onpe0 && ct.verbose > 2)
        (*MPIdata::sout) << "LocGridOrbitals::orthonormalize2states(): " << st1
                         << " and " << st2 << std::endl;
    const int st[2] = { st1, st2 };

    // find color of 2 states
    int color_st[2] = { -1, -1 };
    for (short ic = 0; ic < 2; ++ic)
    {
        color_st[ic] = pack_->getColor(st[ic]);
    }

    // work to do only if one of the states exists locally
    if (color_st[0] >= 0 || color_st[1] >= 0)
    {
        // apply mask of other function
        for (short iloc = 0; iloc < subdivx_; iloc++)
        {
            const int stc1
                = color_st[0] >= 0 ? overlapping_gids_[iloc][color_st[0]] : -1;
            const int stc2
                = color_st[1] >= 0 ? overlapping_gids_[iloc][color_st[1]] : -1;
            if (stc1 == st1 && stc2 == st2)
            {
                (masks4orbitals_->getMask(st1))
                    .apply(psi(color_st[1]), 0, iloc, false);
                (masks4orbitals_->getMask(st2))
                    .apply(psi(color_st[0]), 0, iloc, false);
            }
            else
            {
                if (stc1 == st1 && stc2 == -1)
                    block_vector_.set_zero(color_st[0], iloc);
                if (stc1 == -1 && stc2 == st2)
                    block_vector_.set_zero(color_st[1], iloc);
            }
        }
    }

    double tmp[3]    = { 0., 0., 0. };
    const double vel = grid_.vel();

    for (short iloc = 0; iloc < subdivx_; iloc++)
        for (int ic = 0; ic < 2; ic++)
        {
            const int color_ic = color_st[ic];

            if (color_ic >= 0)
                if (overlapping_gids_[iloc][color_ic] == st[ic])
                {
                    // diagonal element
                    tmp[2 * ic]
                        += vel * block_vector_.dot(color_ic, color_ic, iloc);

                    if (ic == 1)
                    {
                        const int color_jc = color_st[0];
                        if (color_jc >= 0)
                            if (overlapping_gids_[iloc][color_jc] == st[0])
                            {
                                tmp[1] += vel
                                          * block_vector_.dot(
                                              color_ic, color_jc, iloc);
                            }
                    }
                }
        }

    double overlap[3] = { 0., 0., 0. };
    MGmol_MPI& mmpi   = *(MGmol_MPI::instance());
    mmpi.allreduce(&tmp[0], &overlap[0], 3, MPI_SUM);

    // orthogonalize second state
    double alpha = -overlap[1] / overlap[0];
    if (color_st[0] >= 0 && color_st[1] >= 0)
        for (short iloc = 0; iloc < subdivx_; iloc++)
        {
            if (overlapping_gids_[iloc][color_st[0]] == st1
                && overlapping_gids_[iloc][color_st[1]] == st2)
                block_vector_.axpy(alpha, color_st[0], color_st[1], iloc);
        }

    // normalize both states
    const double alpha1 = 1. / sqrt(overlap[0]);
    const double alpha2
        = 1. / sqrt(overlap[2] - overlap[1] * overlap[1] / overlap[0]);
    for (short iloc = 0; iloc < subdivx_; iloc++)
    {
        if (color_st[0] >= 0)
            if (overlapping_gids_[iloc][color_st[0]] == st1)
                block_vector_.scal(alpha1, color_st[0], iloc);

        if (color_st[1] >= 0)
            if (overlapping_gids_[iloc][color_st[1]] == st2)
                block_vector_.scal(alpha2, color_st[1], iloc);
    }

#ifdef DEBUG // testing orthonormality
    tmp[0] = 0.;
    tmp[1] = 0.;
    tmp[2] = 0.;
    for (short iloc = 0; iloc < subdivx_; iloc++)
        for (int ic = 0; ic < 2; ic++)
        {
            const int color_ic = color_st[ic];

            if (color_ic >= 0)
                if (overlapping_gids_[iloc][color_ic] == st1
                    || overlapping_gids_[iloc][color_ic] == st2)
                {
                    // diagonal element
                    tmp[2 * ic]
                        += vel * block_vector_.dot(color_ic, color_ic, iloc);

                    if (ic == 1)
                    {
                        const int color_jc = color_st[0];
                        if (color_jc >= 0)
                            if (overlapping_gids_[iloc][color_jc] == st1)
                            {
                                tmp[1] += vel
                                          * block_vector_.dot(
                                              color_ic, color_jc, iloc);
                            }
                    }
                }
        }

    mmpi.allreduce(&tmp[0], &overlap[0], 3, MPI_SUM);
    if (onpe0 && ct.verbose > 2)
        (*MPIdata::sout) << "Gram matrix = " << overlap[0] << "," << overlap[1]
                         << "," << overlap[2] << std::endl;
#endif
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::multiplyByMatrix2states(const int st1,
    const int st2, const double* mat, LocGridOrbitals<ScalarType>& product)
{
    assert(st1 >= 0);
    assert(st2 >= 0);

    if (chromatic_number_ == 0) return;

    int color_st1 = -1;
    int color_st2 = -1;
    for (short iloc = 0; iloc < subdivx_; iloc++)
        for (int color = 0; color < chromatic_number_; color++)
        {
            const int st = overlapping_gids_[iloc][color];
            if (st == st1)
            {
                color_st1 = color;
                product.block_vector_.set_zero(color, iloc);
            }
            if (st == st2)
            {
                color_st2 = color;
                product.block_vector_.set_zero(color, iloc);
            }
        }

    for (short iloc = 0; iloc < subdivx_; iloc++)
    {
        if (color_st1 >= 0)
            if (overlapping_gids_[iloc][color_st1] == st1)
                product.block_vector_.axpy(
                    mat[0], block_vector_, color_st1, color_st1, iloc);
        if (color_st2 >= 0)
            if (overlapping_gids_[iloc][color_st2] == st2)
                product.block_vector_.axpy(
                    mat[3], block_vector_, color_st2, color_st2, iloc);
        if (color_st1 >= 0 && color_st2 >= 0)
            if (overlapping_gids_[iloc][color_st1] == st1
                && overlapping_gids_[iloc][color_st2] == st2)
            {
                product.block_vector_.axpy(
                    mat[2], block_vector_, color_st1, color_st2, iloc);
                product.block_vector_.axpy(
                    mat[1], block_vector_, color_st2, color_st1, iloc);
            }
    }
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::computeDiagonalGram(
    VariableSizeMatrix<sparserow>& diagS) const
{
    const double vel = grid_.vel();

    for (int color = 0; color < chromatic_number_; color++)
    {
        for (short iloc = 0; iloc < subdivx_; iloc++)
        {
            const int gid = overlapping_gids_[iloc][color];
            if (gid != -1)
            {
                double val
                    = vel * (double)block_vector_.dot(color, color, iloc);
                diagS.insertMatrixElement(gid, gid, val, ADD, true);
            }
        }
    }
    // do data distribution to update local data.
    // All PE's need to know full diagonal entries of
    // overlapping functions, hence append=true.
    distributor_normalize_->augmentLocalData(diagS, true);
#ifdef DEBUG
    Control& ct = *(Control::instance());
    if (onpe0 && ct.verbose > 2)
        for (int i = 0; i < lsize; i++)
            (*MPIdata::sout)
                << "i=" << i << ", diagS[i]=" << diagS.getRowEntry(i, 0)
                << std::endl;
#endif
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::computeInvNorms2(
    std::vector<std::vector<double>>& inv_norms2) const
{
    const int initTabSize = 4096;
    VariableSizeMatrix<sparserow> diagS("diagS", initTabSize);

    computeDiagonalGram(diagS);

    // assign return values
    inv_norms2.resize(subdivx_);
    for (short iloc = 0; iloc < subdivx_; iloc++)
    {
        inv_norms2[iloc].resize(chromatic_number_);
        for (short color = 0; color < chromatic_number_; color++)
        {
            const int gid = overlapping_gids_[iloc][color];
            if (gid != -1)
            {
                inv_norms2[iloc][color] = diagS.get_value(gid, gid);
            }
        }
    }

    for (short iloc = 0; iloc < subdivx_; iloc++)
    {
        for (short color = 0; color < chromatic_number_; color++)
        {
            inv_norms2[iloc][color] = 1. / inv_norms2[iloc][color];
        }
    }
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::normalize()
{
    normalize_tm_.start();

    assert(grid_.vel() > 1.e-8);
    assert(chromatic_number_ >= 0);

    Control& ct = *(Control::instance());

    if (ct.short_sighted)
    {
        const int initTabSize = 4096;
        VariableSizeMatrix<sparserow> diagS("diagSforN", initTabSize);

        computeDiagonalGram(diagS);

        /* Do normalization */
        for (short iloc = 0; iloc < subdivx_; iloc++)
        {
            // Loop over the functions
            for (int color = 0; color < chromatic_number_; color++)
            {
                const int gid = overlapping_gids_[iloc][color];
                if (gid != -1)
                {
                    // normalize state
                    double alpha = 1. / sqrt(diagS.get_value(gid, gid));
                    block_vector_.scal(alpha, color, iloc);
                }
            }
        }
    }
    else
    {
        const double vel = grid_.vel();
        std::vector<double> diagS(numst_, 0.);
        for (int color = 0; color < chromatic_number_; color++)
        {
            for (short iloc = 0; iloc < subdivx_; iloc++)
            {

                const int gid = overlapping_gids_[iloc][color];
                if (gid != -1)
                {
                    diagS[gid] += vel
                                  * static_cast<double>(
                                      block_vector_.dot(color, color, iloc));
                }
            }
        }

        MGmol_MPI& mmpi = *(MGmol_MPI::instance());
        mmpi.split_allreduce_sums_double(&diagS[0], numst_);
#ifdef DEBUG
        if (onpe0 && ct.verbose > 2)
            for (int i = 0; i < numst_; i++)
                (*MPIdata::sout)
                    << "i=" << i << ", diagS[i]=" << diagS[i] << std::endl;
#endif
        for (int i = 0; i < numst_; i++)
        {
            assert(diagS[i] > 1.e-15);
            diagS[i] = 1. / sqrt(diagS[i]);
        }
        for (short iloc = 0; iloc < subdivx_; iloc++)
        {

            // Loop over the functions
            for (int color = 0; color < chromatic_number_; color++)
            {

                const int gid = overlapping_gids_[iloc][color];

                if (gid != -1)
                {
                    // normalize state
                    double alpha = diagS[gid];
                    block_vector_.scal(alpha, color, iloc);
                }
            }
        }
    }

    incrementIterativeIndex();

    normalize_tm_.stop();
}

// modify argument orbitals, by projecting out its component
// along LocGridOrbitals<ScalarType>
template <typename ScalarType>
void LocGridOrbitals<ScalarType>::projectOut(
    LocGridOrbitals<ScalarType>& orbitals, const double scale)
{
    projectOut(orbitals.psi(0), lda_, scale);

#if 0
    // test if projection is now 0
    dist_matrix::DistMatrix<DISTMATDTYPE> tmatrix(product(orbitals));
    if( onpe0 )
        (*MPIdata::sout)<<"LocGridOrbitals::projectOut(), Product after projection:"<<std::endl;
    tmatrix.print((*MPIdata::sout),0,0,5,5);
#endif

    orbitals.incrementIterativeIndex();
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::projectOut(
    ScalarType* const array, const int lda, const double scale)
{
    assert(lda > 1);
    assert(loc_numpt_ > 0);
    assert(chromatic_number_ >= 0);
    assert(lda_ > loc_numpt_);

    SquareLocalMatrices<MATDTYPE, MemorySpace::Host> pmatrix(
        subdivx_, chromatic_number_);

    if (chromatic_number_ != 0) computeLocalProduct(array, lda, pmatrix, false);

        //    pmatrix.scal(grid_.vel());

#ifdef DEBUG
    (*MPIdata::sout) << "LocGridOrbitals::projectOut()" << std::endl;
    (*MPIdata::sout) << "Product before projection" << std::endl;
    pmatrix.print((*MPIdata::sout));
#endif
    proj_matrices_->applyInvS(pmatrix);

    ScalarType* tproduct = new ScalarType[loc_numpt_ * chromatic_number_];
    memset(tproduct, 0, loc_numpt_ * chromatic_number_ * sizeof(ScalarType));

    // loop over subdomains
    for (short iloc = 0; iloc < subdivx_; iloc++)
    {
        ScalarType* phi    = getPsi(0, iloc);
        ScalarType* parray = array + iloc * loc_numpt_;

        MATDTYPE* localMat_iloc = pmatrix.getRawPtr(iloc);

        // Compute loc_numpt_ rows (for subdomain iloc)
        LinearAlgebraUtils<MemorySpace::Host>::MPgemmNN(loc_numpt_,
            chromatic_number_, chromatic_number_, 1., phi, lda_, localMat_iloc,
            chromatic_number_, 0., tproduct, loc_numpt_);

        double minus = -1. * scale;
        for (int j = 0; j < chromatic_number_; j++)
            LinearAlgebraUtils<MemorySpace::Host>::MPaxpy(
                loc_numpt_, minus, tproduct + j * loc_numpt_, parray + j * lda);
    }

    delete[] tproduct;
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::initRand()
{
    Control& ct = *(Control::instance());

    const unsigned dim[3] = { grid_.dim(0), grid_.dim(1), grid_.dim(2) };

    double* xrand = new double[grid_.gdim(0)];
    double* yrand = new double[grid_.gdim(1)];
    double* zrand = new double[grid_.gdim(2)];

    const int loc_length = dim[0] / subdivx_;
    assert(loc_length > 0);
    assert(static_cast<unsigned int>(loc_length) <= dim[0]);

    const int xoff = grid_.istart(0);
    const int yoff = grid_.istart(1);
    const int zoff = grid_.istart(2);

    if (onpe0 && ct.verbose > 2)
        (*MPIdata::sout) << " Initialize " << chromatic_number_
                         << " random global functions" << std::endl;

    ran0();

    // set_zero();

    const int incx = dim[1] * dim[2];
    const int incy = dim[2];

    for (int istate = 0; istate < numst_; istate++)
    {
        // Generate x, y, z random number sequences
        for (unsigned int idx = 0; idx < grid_.gdim(0); idx++)
            xrand[idx] = ran0() - 0.5;
        for (unsigned int idx = 0; idx < grid_.gdim(1); idx++)
            yrand[idx] = ran0() - 0.5;
        for (unsigned int idx = 0; idx < grid_.gdim(2); idx++)
            zrand[idx] = ran0() - 0.5;

        int n = 0;

        for (int color = 0; color < chromatic_number_; color++)
        {
            for (short iloc = 0; iloc < subdivx_; iloc++)
            {
                assert(overlapping_gids_[iloc][color] < numst_);
                const int gid = overlapping_gids_[iloc][color];
                if (gid == istate)
                {

                    for (int ix = loc_length * iloc;
                         ix < loc_length * (iloc + 1); ix++)
                        for (unsigned int iy = 0; iy < dim[1]; iy++)
                            for (unsigned int iz = 0; iz < dim[2]; iz++)
                            {
                                const double alpha = xrand[xoff + ix]
                                                     * yrand[yoff + iy]
                                                     * zrand[zoff + iz];

                                psi(color)[ix * incx + iy * incy + iz]
                                    = alpha * alpha;
                                assert((ix * incx + iy * incy + iz)
                                       < static_cast<unsigned int>(lda_));
                            }
                    n++;

#ifdef DEBUG
                }
                else if (gid == -1)
                {
                    if (!(fabs(psi(color)[iloc * loc_length * incx]) < 1.e-15))
                        (*MPIdata::sout)
                            << "color=" << color << ", iloc=" << iloc
                            << ", val=" << psi(color)[iloc * loc_length]
                            << std::endl;
                    assert(fabs(psi(color)[iloc * loc_length * incx]) < 1.e-15);
#endif
                }
            }
        }
#ifdef DEBUG
        // Make sure that state istate is localized somewhere
        mmpi.allreduce(&n, 1, MPI_SUM);
        if (n <= 0)
        {
            (*MPIdata::serr)
                << "ERROR: state " << istate << " not allocated" << std::endl;
            (*MPIdata::serr)
                << "    chromatic_number_=" << chromatic_number_ << std::endl;
            for (short iloc = 0; iloc < subdivx_; iloc++)
                (*MPIdata::serr)
                    << "    overlapping_gids_[" << iloc << "][" << 0
                    << "]=" << overlapping_gids_[iloc][0] << std::endl;
            mmpi.abort();
        }
#endif
    }

    delete[] xrand;
    delete[] yrand;
    delete[] zrand;

    resetIterativeIndex();
}

// Compute nstates column of Psi^T*A*Psi starting at column 0
#ifdef MGMOL_USE_SCALAPACK
template <typename ScalarType>
void LocGridOrbitals<ScalarType>::addDotWithNcol2Matrix(
    LocGridOrbitals<ScalarType>& Apsi,
    dist_matrix::DistMatrix<DISTMATDTYPE>& matrix) const
{
    addDot_tm_.start();

    assert(chromatic_number_ > 0);

#ifdef DEBUG
    Control& ct = *(Control::instance());
    if (onpe0 && ct.verbose > 2)
        (*MPIdata::sout)
            << "LocGridOrbitals::addDotWithNcol2Matrix for states 0"
            << " to " << chromatic_number_ - 1 << std::endl;
    for (short icolor = 0; icolor < chromatic_number_; icolor++)
    {
        block_vector_.hasnan(icolor);
    }
    for (int icolor = 0; icolor < chromatic_number_; icolor++)
    {
        Apsi.block_vector_.hasnan(icolor);
    }
#endif
    const double vel = grid_.vel();

    SquareLocalMatrices<MATDTYPE, MemorySpace::Host> ss(
        subdivx_, chromatic_number_);

    for (short iloc = 0; iloc < subdivx_; iloc++)
    {
        MATDTYPE* ssiloc = ss.getRawPtr(iloc);

        LinearAlgebraUtils<memory_space_type>::MPgemmTN(chromatic_number_,
            chromatic_number_, loc_numpt_, vel,
            block_vector_.vect(0) + iloc * loc_numpt_, lda_,
            Apsi.getPsi(0, iloc), lda_, 0., ssiloc, chromatic_number_);
    }

    LocalMatrices2DistMatrix* sl2dm = LocalMatrices2DistMatrix::instance();

    sl2dm->accumulate(ss, matrix);

    addDot_tm_.stop();
}
#endif

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::computeGlobalIndexes(
    std::shared_ptr<LocalizationRegions> lrs)
{
    all_overlapping_gids_ = lrs->getOverlapGids();

    lrs->getLocalSubdomainIndices(local_gids_);

    overlapping_gids_.clear();
    overlapping_gids_.resize(subdivx_);
    for (short iloc = 0; iloc < subdivx_; iloc++)
    {
        overlapping_gids_[iloc].resize(chromatic_number_, -1);
    }
    for (std::vector<int>::const_iterator it = all_overlapping_gids_.begin();
         it != all_overlapping_gids_.end(); it++)
    {
        const int gid = *it;
        short color   = pack_->getColor(gid);
        assert(color < chromatic_number_);
        for (short iloc = 0; iloc < subdivx_; iloc++)
        {
            if (lrs->overlapSubdiv(gid, iloc))
            {
                overlapping_gids_[iloc][color] = gid;
            }
        }
    }
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::printTimers(std::ostream& os)
{
    matB_tm_.print(os);
    invBmat_tm_.print(os);
    overlap_tm_.print(os);
    dot_product_tm_.print(os);
    mask_tm_.print(os);
    addDot_tm_.print(os);
    prod_matrix_tm_.print(os);
    get_dm_tm_.print(os);
    assign_tm_.print(os);
    normalize_tm_.print(os);
    axpy_tm_.print(os);
}

template <typename ScalarType>
void LocGridOrbitals<ScalarType>::initWF(
    const std::shared_ptr<LocalizationRegions> lrs)
{
    Control& ct = *(Control::instance());

    if (onpe0 && ct.verbose > 1)
    {
        (*MPIdata::sout) << " Initialize wave functions ..." << std::endl;
    }
    switch (ct.init_type)
    {
        case 1:
            if (onpe0 && ct.verbose > 1)
            {
                (*MPIdata::sout) << " with Gaussian functions..." << std::endl;
            }
            initGauss(ct.init_rc, lrs);
            break;
        case 2:
            if (onpe0 && ct.verbose > 1)
            {
                (*MPIdata::sout) << " with Fourier basis ..." << std::endl;
            }
            initFourier();
            break;
        default:
            if (onpe0 && ct.verbose > 2)
            {
                (*MPIdata::sout) << " with random values ..." << std::endl;
            }
            initRand();

            if (ct.globalColoring())
            {
                // smooth out random functions
                pb::Laph4M<ScalarType> myoper(grid_);
                pb::GridFunc<ScalarType> gf_work(
                    grid_, ct.bcWF[0], ct.bcWF[1], ct.bcWF[2]);
                pb::GridFunc<ScalarType> gf_psi(
                    grid_, ct.bcWF[0], ct.bcWF[1], ct.bcWF[2]);

                if (onpe0 && ct.verbose > 2)
                    (*MPIdata::sout)
                        << " Apply B to initial wave functions" << std::endl;
                for (short icolor = 0; icolor < chromatic_number_; icolor++)
                {
                    gf_psi.assign(psi(icolor));
                    myoper.rhs(gf_psi, gf_work);
                    setPsi(gf_work, icolor);
                    // gf_work.init_vect(psi(icolor),'d');
                }
            }
    }
    resetIterativeIndex();

    // apply masks
    if (ct.init_type == 0)
        applyMask(true);
    else
        applyCorrMask(true);

    if (onpe0 && ct.verbose > 2)
        (*MPIdata::sout) << " Orthonormalize initial wave functions"
                         << std::endl;
    if (ct.isLocMode())
    {
        normalize();
    }
    else
    {
        orthonormalizeLoewdin();
    }

    setDataWithGhosts();
    trade_boundaries();

#ifdef DEBUG
    if (onpe0 && ct.verbose > 2)
        (*MPIdata::sout) << "LocGridOrbitals::init_wf() done" << std::endl;
#endif
}

template class LocGridOrbitals<ORBDTYPE>;

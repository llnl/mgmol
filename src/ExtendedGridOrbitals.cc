// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE
#include "ExtendedGridOrbitals.h"

#include "global.h"

#include "Control.h"
#include "DotProductManagerFactory.h"
#include "GridFunc.h"
#include "Laph4M.h"
#ifdef MGMOL_USE_SCALAPACK
#include "LocalMatrices2DistMatrix.h"
#endif
#include "LocalizationRegions.h"
#include "MPIdata.h"
#include "Mesh.h"
#include "ProjectedMatrices.h"
#include "ReplicatedWorkSpace.h"
#include "SquareLocalMatrices.h"
#include "hdf_tools.h"
#include "lapack_c.h"
#include "memory_space.h"

#include <cmath>
#include <fstream>
#include <mpi.h>
#include <utility>

#ifdef MGMOL_HAS_LIBROM
#include "librom.h"
#endif

#define ORBITAL_OCCUPATION 2.
std::string getDatasetName(const std::string& name, const int color);

template <typename ScalarType>
DotProductManager<ExtendedGridOrbitals<ScalarType>>*
    ExtendedGridOrbitals<ScalarType>::dotProductManager_
    = nullptr;

template <typename ScalarType>
Timer ExtendedGridOrbitals<ScalarType>::matB_tm_(
    "ExtendedGridOrbitals" + std::to_string(8 * sizeof(ScalarType)) + "::matB");
template <typename ScalarType>
Timer ExtendedGridOrbitals<ScalarType>::invBmat_tm_(
    "ExtendedGridOrbitals" + std::to_string(8 * sizeof(ScalarType))
    + "::invBmat");
template <typename ScalarType>
Timer ExtendedGridOrbitals<ScalarType>::overlap_tm_(
    "ExtendedGridOrbitals" + std::to_string(8 * sizeof(ScalarType))
    + "::overlap");
template <typename ScalarType>
Timer ExtendedGridOrbitals<ScalarType>::dot_product_tm_(
    "ExtendedGridOrbitals" + std::to_string(8 * sizeof(ScalarType))
    + "::dot_product");
template <typename ScalarType>
Timer ExtendedGridOrbitals<ScalarType>::addDot_tm_(
    "ExtendedGridOrbitals" + std::to_string(8 * sizeof(ScalarType))
    + "::addDot");
template <typename ScalarType>
Timer ExtendedGridOrbitals<ScalarType>::prod_matrix_tm_(
    "ExtendedGridOrbitals" + std::to_string(8 * sizeof(ScalarType))
    + "::prod_matrix");
template <typename ScalarType>
Timer ExtendedGridOrbitals<ScalarType>::assign_tm_(
    "ExtendedGridOrbitals" + std::to_string(8 * sizeof(ScalarType))
    + "::assign");
template <typename ScalarType>
Timer ExtendedGridOrbitals<ScalarType>::normalize_tm_(
    "ExtendedGridOrbitals" + std::to_string(8 * sizeof(ScalarType))
    + "::normalize");
template <typename ScalarType>
Timer ExtendedGridOrbitals<ScalarType>::axpy_tm_(
    "ExtendedGridOrbitals" + std::to_string(8 * sizeof(ScalarType)) + "::axpy");
template <typename ScalarType>
Timer ExtendedGridOrbitals<ScalarType>::gemm_nn_tm_(
    "ExtendedGridOrbitals" + std::to_string(8 * sizeof(ScalarType))
    + "::gemm_nn");
template <typename ScalarType>
Timer ExtendedGridOrbitals<ScalarType>::gemm_tn_tm_(
    "ExtendedGridOrbitals" + std::to_string(8 * sizeof(ScalarType))
    + "::gemm_tn");

template <typename ScalarType>
ExtendedGridOrbitals<ScalarType>::ExtendedGridOrbitals(std::string name,
    const pb::Grid& my_grid, const short subdivx, const int numst,
    const short bc[3], ProjectedMatricesInterface* proj_matrices,
    std::shared_ptr<LocalizationRegions> lrs, MasksSet* masks,
    MasksSet* corrmasks, ClusterOrbitals* local_cluster, const bool setup_flag)
    : name_(std::move(name)),
      proj_matrices_(proj_matrices),
      block_vector_(my_grid, 1, bc),
      grid_(my_grid)
{
    (void)lrs;
    (void)masks;
    (void)corrmasks;
    (void)local_cluster;

    // preconditions
#ifndef NDEBUG
    assert(subdivx == 1);
#else
    (void)subdivx;
#endif
    assert(proj_matrices != nullptr);

    for (short i = 0; i < 3; i++)
        assert(bc[i] == 0 || bc[i] == 1);
    assert(grid_.size() > 0);

    numst_ = numst;
    numpt_ = grid_.size();
    lda_   = block_vector_.getld();

    assert(numst_ >= 0);

    if (setup_flag) setup();
}

template <typename ScalarType>
ExtendedGridOrbitals<ScalarType>::~ExtendedGridOrbitals()
{
    assert(proj_matrices_ != nullptr);
}

template <typename ScalarType>
ExtendedGridOrbitals<ScalarType>::ExtendedGridOrbitals(const std::string& name,
    const ExtendedGridOrbitals<ScalarType>& A, const bool copy_data)
    : Orbitals(A, copy_data),
      name_(name),
      proj_matrices_(A.proj_matrices_),
      block_vector_(A.block_vector_, copy_data),
      grid_(A.grid_)
{
    assert(A.proj_matrices_ != nullptr);
}

template <typename ScalarType>
ExtendedGridOrbitals<ScalarType>::ExtendedGridOrbitals(const std::string& name,
    const ExtendedGridOrbitals<ScalarType>& A,
    ProjectedMatricesInterface* proj_matrices, const bool copy_data)
    : Orbitals(A, copy_data),
      name_(name),
      proj_matrices_(proj_matrices),
      block_vector_(A.block_vector_, copy_data),
      grid_(A.grid_)
{
    assert(proj_matrices != nullptr);

    // setup new projected_matrices object
    proj_matrices_->setup(overlapping_gids_);
}

template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::copyDataFrom(
    const ExtendedGridOrbitals& src)
{
    assert(proj_matrices_ != nullptr);

    block_vector_.copyDataFrom(src.block_vector_);

    setIterativeIndex(src);
}

template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::setDotProduct(const short dot_type)
{
    DotProductManagerFactory<ExtendedGridOrbitals> factory;

    dotProductManager_ = factory.create(dot_type);

    assert(dotProductManager_ != nullptr);
}

template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::setup()
{
    Control& ct = *(Control::instance());

    // preconditions
    assert(proj_matrices_ != nullptr);

    if (ct.verbose > 0)
        printWithTimeStamp(
            "ExtendedGridOrbitals::setup()...", (*MPIdata::sout));

    computeGlobalIndexes();

    bool skinny_stencil = !ct.Mehrstellen();

    block_vector_.initialize(overlapping_gids_, skinny_stencil);

    proj_matrices_->setup(overlapping_gids_);

    if (ct.verbose > 0)
        printWithTimeStamp(
            "ExtendedGridOrbitals::setup() done...", (*MPIdata::sout));
}

template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::reset(MasksSet* masks,
    MasksSet* corrmasks, std::shared_ptr<LocalizationRegions> lrs)
{
    (void)masks;
    (void)corrmasks;
    (void)lrs;

    // free some old data
    block_vector_.clear();
    setIterativeIndex(-10);

    // reset
    setup();
}

template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::assign(
    const ExtendedGridOrbitals& orbitals)
{
    assert(proj_matrices_ != nullptr);

    assign_tm_.start();

    setIterativeIndex(orbitals);

    block_vector_.copyDataFrom(orbitals.block_vector_);

    assign_tm_.stop();
}

template <typename ScalarType>
template <typename CoeffType>
void ExtendedGridOrbitals<ScalarType>::axpy(
    const CoeffType alpha, const ExtendedGridOrbitals<ScalarType>& orbitals)
{
    axpy_tm_.start();

    block_vector_.axpy(alpha, orbitals.block_vector_);

    incrementIterativeIndex();

    axpy_tm_.stop();
}

template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::initGauss(
    const double rc, const std::shared_ptr<LocalizationRegions> lrs)
{
    assert(numst_ >= 0);

    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    Control& ct     = *(Control::instance());
    if (mmpi.instancePE0() && ct.verbose > 2)
        (*MPIdata::sout) << "Initial orbitals: Gaussians of width " << rc
                         << std::endl;

    const double invrc2 = 1. / (rc * rc);

    const double start0 = grid_.start(0);
    const double start1 = grid_.start(1);
    const double start2 = grid_.start(2);

    const int dim0 = grid_.dim(0);
    const int dim1 = grid_.dim(1);
    const int dim2 = grid_.dim(2);

    const int incx = dim1 * dim2;
    const int incy = dim2;

    const double hgrid[3] = { grid_.hgrid(0), grid_.hgrid(2), grid_.hgrid(2) };

    Vector3D ll;
    for (short i = 0; i < 3; i++)
        ll[i] = grid_.ll(i);

    const double rmax = 6. * rc;
    for (int icolor = 0; icolor < numst_; icolor++)
    {
        ScalarType* ipsi             = psi(icolor);
        unsigned int const ipsi_size = numpt_;
        ScalarType* ipsi_host_view   = MemorySpace::Memory<ScalarType,
            memory_space_type>::allocate_host_view(ipsi_size);
        MemorySpace::Memory<ScalarType, memory_space_type>::copy_view_to_host(
            ipsi, ipsi_size, ipsi_host_view);
        MemorySpace::Memory<ScalarType, MemorySpace::Host>::set(
            ipsi_host_view, ipsi_size, 0);

        const Vector3D& center(lrs->getCenter(icolor));
        Vector3D xc;

        xc[0] = start0;
        for (int ix = 0; ix < dim0; ix++)
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
                            = static_cast<ScalarType>(exp(-r * r * invrc2));
                    else
                        ipsi_host_view[ix * incx + iy * incy + iz] = 0.;

                    xc[2] += hgrid[2];
                }
                xc[1] += hgrid[1];
            }
            xc[0] += hgrid[0];
        }

        MemorySpace::Memory<ScalarType, memory_space_type>::copy_view_to_dev(
            ipsi_host_view, ipsi_size, ipsi);
        MemorySpace::Memory<ScalarType, memory_space_type>::free_host_view(
            ipsi_host_view);
    }
    resetIterativeIndex();
}

template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::initFourier()
{
    Control& ct = *(Control::instance());
    if (onpe0 && ct.verbose > 2)
        (*MPIdata::sout) << "Initial orbitals: Fourier " << std::endl;

    const double start0 = grid_.start(0) - grid_.origin(0);
    const double start1 = grid_.start(1) - grid_.origin(1);
    const double start2 = grid_.start(2) - grid_.origin(2);

    const int dim0 = grid_.dim(0);
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

    const int cbrtncolors = (int)ceil(cbrt(numst_));

    for (int icolor = 0; icolor < numst_; icolor++)
    {
        int index = icolor + 1;
        int kvector[3];
        getkvector(index, cbrtncolors, kvector);

        const double kk[3] = { dk[0] * (double)kvector[0],
            dk[1] * (double)kvector[1], dk[2] * (double)kvector[2] };

        ScalarType* ipsi             = psi(icolor);
        unsigned int const ipsi_size = numpt_;
        ScalarType* ipsi_host_view   = MemorySpace::Memory<ScalarType,
            memory_space_type>::allocate_host_view(ipsi_size);
        MemorySpace::Memory<ScalarType, memory_space_type>::copy_view_to_host(
            ipsi, ipsi_size, ipsi_host_view);
        MemorySpace::Memory<ScalarType, MemorySpace::Host>::set(
            ipsi_host_view, numpt_, 0);

        // TODO this can be done on the GPU with OpenMP
        double x = start0;
        for (int ix = 0; ix < dim0; ix++)
        {
            double y = start1;

            for (int iy = 0; iy < dim1; iy++)
            {
                double z = start2;
                for (int iz = 0; iz < dim2; iz++)
                {
                    ipsi_host_view[ix * incx + iy * incy + iz]
                        = 1.
                          - static_cast<ScalarType>(std::cos(kk[0] * x)
                                                    * std::cos(kk[1] * y)
                                                    * std::cos(kk[2] * z));

                    z += hgrid[2];
                }
                y += hgrid[1];
            }
            x += hgrid[0];
        }

        MemorySpace::Memory<ScalarType, memory_space_type>::copy_view_to_dev(
            ipsi_host_view, ipsi_size, ipsi);
        MemorySpace::Memory<ScalarType, memory_space_type>::free_host_view(
            ipsi_host_view);
    }
    resetIterativeIndex();
}

template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::multiply_by_matrix(
    const DISTMATDTYPE* const matrix, ScalarType* product, const int ldp) const
{
    prod_matrix_tm_.start();

    unsigned int const product_size = numst_ * ldp;
    ScalarType* product_host_view   = MemorySpace::Memory<ScalarType,
        memory_space_type>::allocate_host_view(product_size);
    MemorySpace::Memory<ScalarType, memory_space_type>::copy_view_to_host(
        product, product_size, product_host_view);

    unsigned int const phi_size = numpt_ * numst_;
    ScalarType* phi_host_view   = MemorySpace::Memory<ScalarType,
        memory_space_type>::allocate_host_view(phi_size);
    MemorySpace::Memory<ScalarType, memory_space_type>::copy_view_to_host(
        getPsi(0), phi_size, phi_host_view);

    // TODO this can be done on the GPU
    gemm_nn_tm_.start();
    LinearAlgebraUtils<MemorySpace::Host>::MPgemmNN(numpt_, numst_, numst_, 1.,
        phi_host_view, lda_, matrix, numst_, 0., product_host_view, ldp);
    gemm_nn_tm_.stop();

    MemorySpace::Memory<ScalarType, memory_space_type>::free_host_view(
        phi_host_view);

    MemorySpace::Memory<ScalarType, memory_space_type>::copy_view_to_dev(
        product_host_view, product_size, product);
    MemorySpace::Memory<ScalarType, memory_space_type>::free_host_view(
        product_host_view);

    prod_matrix_tm_.stop();
}

#ifdef HAVE_MAGMA
template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::multiplyByMatrix(
    const SquareLocalMatrices<MATDTYPE, MemorySpace::Host>& matrix,
    ScalarType* product, const int ldp) const
{
    SquareLocalMatrices<ScalarType, MemorySpace::Device> matdev(
        matrix.nmat(), matrix.m());
    matdev.assign(matrix);

    multiplyByMatrix(matdev, product, ldp);
}
#endif

template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::multiplyByMatrix(
    const SquareLocalMatrices<MATDTYPE, memory_space_type>& matrix,
    ScalarType* product, const int ldp) const
{
    assert(matrix.nmat() == 1);

    gemm_nn_tm_.start();

    const MATDTYPE* const mat = matrix.getSubMatrix();

    LinearAlgebraUtils<memory_space_type>::MPgemmNN(numpt_, numst_, numst_, 1.,
        getPsi(0), lda_, mat, numst_, 0., product, ldp);

    gemm_nn_tm_.stop();
}

// Here the result is stored in one of the matrices used in the multiplication,
// so a temporary arry is necessary
#ifdef HAVE_MAGMA
template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::multiplyByMatrix(
    const SquareLocalMatrices<MATDTYPE, MemorySpace::Host>& matrix)
{
    SquareLocalMatrices<ScalarType, MemorySpace::Device> matdev(
        matrix.nmat(), matrix.m());
    matdev.assign(matrix);

    multiplyByMatrix(matdev);
}
#endif

template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::multiplyByMatrix(
    const SquareLocalMatrices<MATDTYPE, memory_space_type>& matrix)
{
    ScalarType* product
        = MemorySpace::Memory<ScalarType, memory_space_type>::allocate(
            numpt_ * numst_);

    multiplyByMatrix(matrix, product, numpt_);

    MemorySpace::Memory<ScalarType, memory_space_type>::copy(
        product, numpt_ * numst_, getPsi(0));

    MemorySpace::Memory<ScalarType, memory_space_type>::free(product);
}

template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::multiplyByMatrix(
    const SquareLocalMatrices<MATDTYPE, MemorySpace::Host>& matrix,
    ExtendedGridOrbitals<ScalarType>& product) const
{
    multiplyByMatrix(matrix, product.psi(0), product.lda_);
}

template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::multiply_by_matrix(
    const DISTMATDTYPE* const matrix,
    ExtendedGridOrbitals<ScalarType>& product) const
{
    multiply_by_matrix(matrix, product.psi(0), product.lda_);
}

#ifdef MGMOL_USE_SCALAPACK
template <>
template <>
void ExtendedGridOrbitals<ORBDTYPE>::multiply_by_matrix(
    const dist_matrix::DistMatrix<DISTMATDTYPE>& matrix, const double alpha,
    ExtendedGridOrbitals<ORBDTYPE>& product)
{
    multiply_by_DistMatrix(matrix, alpha, product);
}
#endif

template <>
template <>
void ExtendedGridOrbitals<ORBDTYPE>::multiply_by_matrix(
    const ReplicatedMatrix& matrix, const double alpha,
    ExtendedGridOrbitals<ORBDTYPE>& product)
{
    multiply_by_ReplicatedMatrix(matrix, alpha, product);
}

#ifdef MGMOL_USE_SCALAPACK
template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::multiply_by_DistMatrix(
    const dist_matrix::DistMatrix<DISTMATDTYPE>& matrix, const double alpha,
    ExtendedGridOrbitals<ORBDTYPE>& product)
{
    prod_matrix_tm_.start();

    ScalarType* product_ptr
        = (this == &product)
              ? MemorySpace::Memory<ScalarType, MemorySpace::Host>::allocate(
                  numpt_ * numst_)
              : product.getPsi(0);

    ReplicatedWorkSpace<DISTMATDTYPE>& wspace(
        ReplicatedWorkSpace<DISTMATDTYPE>::instance());
    DISTMATDTYPE* work_matrix = wspace.square_matrix();

    matrix.allgather(work_matrix, numst_);

    const size_t slnumpt = numpt_ * sizeof(ScalarType);

    unsigned int const phi_size = numpt_ * numst_;
    ScalarType* phi_host_view   = MemorySpace::Memory<ScalarType,
        memory_space_type>::allocate_host_view(phi_size);
    MemorySpace::Memory<ScalarType, memory_space_type>::copy_view_to_host(
        getPsi(0), phi_size, phi_host_view);

    // TODO this can be done on the GPU
    gemm_nn_tm_.start();
    LinearAlgebraUtils<MemorySpace::Host>::MPgemmNN(numpt_, numst_, numst_, 1.,
        phi_host_view, lda_, work_matrix, numst_, alpha, product_ptr, numpt_);
    gemm_nn_tm_.stop();

    for (int color = 0; color < numst_; color++)
        memcpy(phi_host_view + color * lda_, product_ptr + color * numpt_,
            slnumpt);

    MemorySpace::Memory<ScalarType, memory_space_type>::copy_view_to_dev(
        phi_host_view, phi_size, getPsi(0));
    MemorySpace::Memory<ScalarType, memory_space_type>::free_host_view(
        phi_host_view);

    if (this == &product)
        MemorySpace::Memory<ScalarType, MemorySpace::Host>::free(product_ptr);

    prod_matrix_tm_.stop();
}
#endif

template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::multiply_by_ReplicatedMatrix(
    const ReplicatedMatrix& matrix, const double alpha,
    ExtendedGridOrbitals<ORBDTYPE>& product)
{
    gemm_nn_tm_.start();

#ifdef HAVE_MAGMA
    magma_trans_t magma_transa = magma_trans_const('n');
    magma_trans_t magma_transb = magma_trans_const('n');

    auto& magma_singleton = MagmaSingleton::get_magma_singleton();

    ScalarType* product_ptr
        = (this == &product)
              ? MemorySpace::Memory<ScalarType, MemorySpace::Device>::allocate(
                  numst_ * lda_)
              : product.getPsi(0);

    magmablas_dgemm(magma_transa, magma_transb, numpt_, numst_, numst_, 1.,
        block_vector_.vect(0), lda_, matrix.data(), matrix.ld(), alpha,
        product_ptr, lda_, magma_singleton.queue_);

    if (this == &product)
    {
        MemorySpace::Memory<ScalarType, MemorySpace::Device>::copy(
            product_ptr, numst_ * lda_, block_vector_.vect(0));

        MemorySpace::Memory<ScalarType, MemorySpace::Device>::free(product);
    }
#else
    ScalarType* product_ptr
        = (this == &product)
              ? MemorySpace::Memory<ScalarType, MemorySpace::Host>::allocate(
                  numst_ * lda_)
              : product.getPsi(0);
    LinearAlgebraUtils<MemorySpace::Host>::MPgemmNN(numpt_, numst_, numst_, 1.,
        block_vector_.vect(0), lda_, matrix.data(), matrix.ld(), alpha,
        product_ptr, lda_);

    if (this == &product)
    {
        memcpy(block_vector_.vect(0), product_ptr,
            numst_ * lda_ * sizeof(ScalarType));

        MemorySpace::Memory<ScalarType, MemorySpace::Host>::free(product_ptr);
    }
#endif

    gemm_nn_tm_.stop();
}

template <typename ScalarType>
int ExtendedGridOrbitals<ScalarType>::read_hdf5(HDFrestart& h5f_file)
{
    assert(proj_matrices_ != nullptr);

    Control& ct = *(Control::instance());

    std::string name = "Function";
    int ierr         = read_func_hdf5(h5f_file, name);
    if (ierr < 0)
    {
        (*MPIdata::serr) << "ExtendedGridOrbitals<ScalarType>::read_hdf5(): "
                            "error in reading "
                         << name << ", size=" << name.size() << std::endl;
        return ierr;
    }
    else if (onpe0 && ct.verbose > 2)
    {
        (*MPIdata::sout) << "ExtendedGridOrbitals::read_hdf5(): Read " << ierr
                         << " functions in restart file" << std::endl;
    }

    // Read DM
    if (!ct.fullyOccupied())
    {
        ierr = proj_matrices_->readDM(h5f_file);
        if (ierr < 0)
        {
            (*MPIdata::serr) << "ExtendedGridOrbitals<ScalarType>::read_hdf5():"
                                " error in reading DM"
                             << std::endl;
            return ierr;
        }
    }

    resetIterativeIndex();

    return ierr;
}

template <typename ScalarType>
int ExtendedGridOrbitals<ScalarType>::write(
    HDFrestart& h5f_file, const std::string& name)
{
    if (onpe0)
        (*MPIdata::sout) << "ExtendedGridOrbitals::write_func_hdf5()...\n";
    Control& ct   = *(Control::instance());
    hid_t file_id = h5f_file.file_id();
    bool iwrite   = h5f_file.active();

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
        (*MPIdata::sout) << "Write ExtendedGridOrbitals " << name
                         << " with precision " << precision << std::endl;
    // loop over global (storage) functions
    for (int color = 0; color < numst_; color++)
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
                (*MPIdata::serr) << "ExtendedGridOrbitals::write_func_hdf5(), "
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

        std::vector<int> gids;
        gids.push_back(color);

        if (iwrite)
        {
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

        unsigned int const psi_size = numpt_;
        ScalarType* psi_host_view   = MemorySpace::Memory<ScalarType,
            memory_space_type>::allocate_host_view(psi_size);
        MemorySpace::Memory<ScalarType, memory_space_type>::copy_view_to_host(
            psi(color), psi_size, psi_host_view);

        int ierr = h5f_file.writeData(
            psi_host_view, filespace, memspace, dset_id, precision);
        if (ierr < 0) return ierr;

        // Close/release resources.
        if (iwrite)
        {
            herr_t status = H5Dclose(dset_id);
            if (status < 0)
            {
                (*MPIdata::serr)
                    << "ExtendedGridOrbitals<ScalarType>::write_func_hdf5:"
                       "H5Dclose failed!!!"
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

// read all the data sets with names starting with "name"
template <typename ScalarType>
int ExtendedGridOrbitals<ScalarType>::read_func_hdf5(
    HDFrestart& h5f_file, const std::string& name)
{
    assert(numst_ >= 0);
    assert(name.size() > 0);

    Control& ct     = *(Control::instance());
    MGmol_MPI& mmpi = *(MGmol_MPI::instance());

    hsize_t block[3] = { grid_.dim(0), grid_.dim(1), grid_.dim(2) };
    if (h5f_file.gatherDataX())
    {
        block[0] = grid_.gdim(0);
    }

    // Each process defines dataset in memory and writes it to the hyperslab
    // in the file.

    // memory dataspace identifier
    hid_t memspace = (h5f_file.active()) ? h5f_file.createMemspace() : 0;

    ScalarType* buffer = new ScalarType[block[0] * block[1] * block[2]];

    if (onpe0 && ct.verbose > 2)
    {
        if (h5f_file.gatherDataX())
        {
            (*MPIdata::sout) << "ExtendedGridOrbitals::read_func_"
                                "hdf5(): Read wave "
                                "functions from "
                             << grid_.mype_env().n_mpi_task(1)
                                    * grid_.mype_env().n_mpi_task(2)
                             << " PEs" << std::endl;
        }
        else
        {
            (*MPIdata::sout) << "ExtendedGridOrbitals::read_func_hdf5(): Read "
                                "wave functions "
                             << name << " from all tasks..." << std::endl;
        }
    }

    const short precision = ct.restart_info > 3 ? 2 : 1;

    for (int icolor = 0; icolor < numst_; icolor++)
    {
        const std::string datasetname(getDatasetName(name, icolor));

        // check if dataset exists...
        int err_id = h5f_file.checkDataExists(datasetname);
        if (h5f_file.gatherDataX()) mmpi.bcast(&err_id, 1);
        if (err_id == 0) break; // dataset does not exists

        if (onpe0 && ct.verbose > 2)
            (*MPIdata::sout) << "Read Dataset " << datasetname
                             << " with precision " << precision << std::endl;

        // Open dataset.
        hid_t dset_id = h5f_file.open_dset(datasetname);
        if (dset_id < 0)
        {
            (*MPIdata::serr) << "ExtendedGridOrbitals<ScalarType>::read_func_"
                                "hdf5() --- cannot open "
                             << datasetname << std::endl;
            return dset_id;
        }

        herr_t status = h5f_file.readData(buffer, memspace, dset_id, precision);
        if (status < 0)
        {
            (*MPIdata::serr)
                << "ExtendedGridOrbitals<ScalarType>::read_func_hdf5() --- "
                   "H5Dread failed!!!"
                << std::endl;
            return -1;
        }

        status = h5f_file.close_dset(dset_id);
        if (status < 0)
        {
            return status;
        }

#ifdef HAVE_MAGMA
        ScalarType* buffer_dev
            = MemorySpace::Memory<ScalarType, MemorySpace::Device>::allocate(
                numpt_);
        MemorySpace::copy_to_dev(buffer, numpt_, buffer_dev);
#else
        ScalarType* buffer_dev = buffer;
#endif
        block_vector_.assignLocal(icolor, 0, buffer_dev);
#ifdef HAVE_MAGMA
        MemorySpace::Memory<ScalarType, MemorySpace::Device>::free(buffer_dev);
#endif
    }

    delete[] buffer;
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

    return numst_;
}

// compute the matrix <psi1|B|psi2>
// output: matB
template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::computeMatB(
    const ExtendedGridOrbitals<ScalarType>& orbitals,
    const pb::Lap<ScalarType>& LapOper)
{
    if (numst_ == 0) return;

    assert(proj_matrices_ != nullptr);

    matB_tm_.start();
#if DEBUG
    if (onpe0)
        (*MPIdata::sout) << "ExtendedGridOrbitals::computeMatB()" << std::endl;
#endif

    const short bcolor = 32;

    SquareLocalMatrices<MATDTYPE, MemorySpace::Host> ss(1, numst_);

    ScalarType* work = new ScalarType[lda_ * bcolor];
    memset(work, 0, lda_ * bcolor * sizeof(ScalarType));

    ScalarType* const orbitals_psi
        = (numst_ > 0) ? orbitals.block_vector_.vect(0) : nullptr;
    const unsigned int orbitals_psi_size
        = orbitals.block_vector_.get_allocated_size_storage();
    ScalarType* orbitals_psi_host_view = MemorySpace::Memory<ScalarType,
        memory_space_type>::allocate_host_view(orbitals_psi_size);
    MemorySpace::Memory<ScalarType, memory_space_type>::copy_view_to_host(
        orbitals_psi, orbitals_psi_size, orbitals_psi_host_view);

    setDataWithGhosts();
    trade_boundaries();

    for (int icolor = 0; icolor < numst_; icolor += bcolor)
    {
        int nf = bcolor;
        if ((icolor + nf) > numst_) nf = numst_ - icolor;

        // Compute nf columns of B|psi> and store it into work
        for (int i = 0; i < nf; i++)
        {
            LapOper.rhs(getFuncWithGhosts(icolor + i), work + i * lda_);
        }

        MATDTYPE* ss0 = ss.getRawPtr(0);

        // calculate nf columns of ss0
        LinearAlgebraUtils<memory_space_type>::MPgemmTN(numst_, nf, numpt_,
            grid_.vel(), orbitals_psi_host_view, lda_, work, lda_, 0.,
            ss0 + icolor * numst_, numst_);
    }

    MemorySpace::Memory<ScalarType, memory_space_type>::free_host_view(
        orbitals_psi_host_view);
    delete[] work;

    proj_matrices_->initializeMatB(ss);

    matB_tm_.stop();
}

// compute <Phi|B|Phi> and its inverse
template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::computeBAndInvB(
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
void ExtendedGridOrbitals<ScalarType>::getLocalOverlap(
    SquareLocalMatrices<MATDTYPE, MemorySpace::Host>& ss)
{
    assert(numst_ >= 0);
    assert(numpt_ > 0);
    assert(grid_.vel() > 1.e-8);

    if (numst_ != 0)
    {
        getLocalOverlap(*this, ss);
    }
}

template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::getLocalOverlap(
    const ExtendedGridOrbitals& orbitals,
    SquareLocalMatrices<MATDTYPE, MemorySpace::Host>& ss)
{
    assert(numst_ >= 0);

    if (numst_ != 0)
    {
        computeLocalProduct(
            orbitals.block_vector_.vect(0), orbitals.lda_, ss, false);
    }
}

template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::computeLocalProduct(
    const ExtendedGridOrbitals<ScalarType>& orbitals,
    LocalMatrices<MATDTYPE, MemorySpace::Host>& ss, const bool transpose)
{
    // assert( orbitals.numst_>=0 );
    assert(orbitals.lda_ > 1);

    if (numst_ != 0)
        computeLocalProduct(orbitals.psi(0), orbitals.lda_, ss, transpose);
}

#ifdef HAVE_MAGMA
template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::computeLocalProduct(
    const ScalarType* const array, const int ld,
    LocalMatrices<MATDTYPE, MemorySpace::Host>& ss, const bool transpose)
{
    LocalMatrices<ScalarType, MemorySpace::Device> sdev(
        ss.nmat(), ss.m(), ss.n());

    computeLocalProduct(array, ld, sdev, transpose);

    ss.assign(sdev);
}
#endif

template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::computeLocalProduct(
    const ScalarType* const array, const int ld,
    LocalMatrices<MATDTYPE, memory_space_type>& ss, const bool transpose)
{
    gemm_tn_tm_.start();

    assert(numpt_ > 0);
    assert(numpt_ <= ld);
    assert(array != nullptr);
    assert(numst_ != 0);
    assert(grid_.vel() > 0.);

    Control& ct = *(Control::instance());
    if (onpe0 && ct.verbose > 2)
        std::cout << "ExtendedGridOrbitals::computeLocalProduct()..."
                  << std::endl;

    const ScalarType* const a = transpose ? array : block_vector_.vect(0);
    const ScalarType* const b = transpose ? block_vector_.vect(0) : array;

    const int lda = transpose ? ld : lda_;
    const int ldb = transpose ? lda_ : ld;

    LinearAlgebraUtils<memory_space_type>::MPgemmTN(numst_, numst_, numpt_,
        grid_.vel(), a, lda, b, ldb, 0., ss.getRawPtr(0), ss.m());

    gemm_tn_tm_.stop();
}

template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::computeDiagonalElementsDotProduct(
    const ExtendedGridOrbitals<ScalarType>& orbitals,
    std::vector<DISTMATDTYPE>& ss) const
{
    assert(numst_ > 0);
    assert(grid_.vel() > 0.);

    for (int icolor = 0; icolor < numst_; icolor++)
    {
        ss[icolor]   = 0.;
        double alpha = LinearAlgebraUtils<memory_space_type>::MPdot(
            numpt_, orbitals.getPsi(icolor), getPsi(icolor));

        ss[icolor] += (DISTMATDTYPE)(alpha * grid_.vel());
    }
    std::vector<DISTMATDTYPE> tmp(ss);
    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    mmpi.allreduce(&tmp[0], &ss[0], numst_, MPI_SUM);
}

#ifdef MGMOL_USE_SCALAPACK
template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::computeGram(
    dist_matrix::DistMatrix<DISTMATDTYPE>& gram_mat)
{
    SquareLocalMatrices<MATDTYPE, MemorySpace::Host> ss(1, numst_);

    getLocalOverlap(ss);

    LocalMatrices2DistMatrix* sl2dm = LocalMatrices2DistMatrix::instance();

    gram_mat.clear();

    sl2dm->accumulate(ss, gram_mat);
}

template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::computeGram(
    const ExtendedGridOrbitals& orbitals,
    dist_matrix::DistMatrix<DISTMATDTYPE>& gram_mat)
{
    SquareLocalMatrices<MATDTYPE, MemorySpace::Host> ss(1, numst_);

    getLocalOverlap(orbitals, ss);

    // make a DistMatrix out of ss
    LocalMatrices2DistMatrix* sl2dm = LocalMatrices2DistMatrix::instance();

    gram_mat.clear();

    sl2dm->accumulate(ss, gram_mat);
}
#endif

// compute the lower-triangular part of the overlap matrix
template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::computeGram(const int verbosity)
{
    assert(proj_matrices_ != nullptr);

    overlap_tm_.start();

#ifdef PRINT_OPERATIONS
    if (onpe0)
        (*MPIdata::sout) << "ExtendedGridOrbitals::computeGram()" << std::endl;
#endif

    assert(1 > 0);
    assert(1 < 1000);
    assert(numst_ >= 0);

    SquareLocalMatrices<MATDTYPE, MemorySpace::Host> ss(1, numst_);

    getLocalOverlap(ss);

    proj_matrices_->initializeGramMatrix(ss, getIterativeIndex());

    if (verbosity > 1) proj_matrices_->printS((*MPIdata::sout));

    overlap_tm_.stop();
}

template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::computeGramAndInvS(const int verbosity)
{
    assert(proj_matrices_ != nullptr);

    computeGram(verbosity);

    /* Compute inverse of Gram matrix */
    proj_matrices_->computeInvS();
}

template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::checkCond(
    const double tol, const bool flag_stop)
{
    assert(proj_matrices_ != nullptr);

    proj_matrices_->checkCond(tol, flag_stop);
}

template <typename ScalarType>
double ExtendedGridOrbitals<ScalarType>::dotProduct(
    const ExtendedGridOrbitals<ScalarType>& orbitals)
{
    assert(dotProductManager_ != nullptr);
    return dotProductManager_->dotProduct(*this, orbitals);
}

template <typename ScalarType>
double ExtendedGridOrbitals<ScalarType>::dotProduct(
    const ExtendedGridOrbitals<ScalarType>& orbitals, const short dot_type)
{
    dot_product_tm_.start();

    DotProductManagerFactory<ExtendedGridOrbitals> factory;
    DotProductManager<ExtendedGridOrbitals>* manager = factory.create(dot_type);
    assert(manager != nullptr);

    double dot = manager->dotProduct(*this, orbitals);

    delete manager;

    dot_product_tm_.stop();

    return dot;
}

template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::orthonormalizeLoewdin(
    const bool overlap_uptodate,
    SquareLocalMatrices<MATDTYPE, MemorySpace::Host>* matrixTransform,
    const bool update_matrices)
{
    Control& ct = *(Control::instance());
    if (onpe0 && ct.verbose > 1)
        (*MPIdata::sout) << "ExtendedGridOrbitals::orthonormalizeLoewdin()"
                         << std::endl;

    if (!overlap_uptodate) computeGram(0);

    SquareLocalMatrices<MATDTYPE, MemorySpace::Host>* localP = matrixTransform;
    if (matrixTransform == nullptr)
        localP
            = new SquareLocalMatrices<MATDTYPE, MemorySpace::Host>(1, numst_);

    incrementIterativeIndex();

    bool multbymat = false;
    // try with ReplicatedMatrix first
    {
        ProjectedMatrices<ReplicatedMatrix>* projmatrices
            = dynamic_cast<ProjectedMatrices<ReplicatedMatrix>*>(
                proj_matrices_);
        if (projmatrices)
        {
            projmatrices->computeLoewdinTransform(
                *localP, getIterativeIndex(), update_matrices);
            multiplyByMatrix(*localP);

            projmatrices->setGram2Id(getIterativeIndex());

            multbymat = true;
        }
    }
    if (!multbymat)
    {
#ifdef MGMOL_USE_SCALAPACK
        ProjectedMatrices<dist_matrix::DistMatrix<DISTMATDTYPE>>* projmatrices
            = dynamic_cast<
                ProjectedMatrices<dist_matrix::DistMatrix<DISTMATDTYPE>>*>(
                proj_matrices_);
        if (projmatrices)
        {
            projmatrices->computeLoewdinTransform(
                *localP, getIterativeIndex(), update_matrices);
            multiplyByMatrix(*localP);

            projmatrices->setGram2Id(getIterativeIndex());
        }
#endif
    }

    if (matrixTransform == nullptr) delete localP;
}

template <typename ScalarType>
double ExtendedGridOrbitals<ScalarType>::norm() const
{
    double norm = 0;

    for (int gid = 0; gid < numst_; gid++)
    {
        norm += normState(gid);
    }
    return norm;
}

template <typename ScalarType>
double ExtendedGridOrbitals<ScalarType>::normState(const int gid) const
{
    assert(gid >= 0);

    double tmp = 0.;

    // diagonal element
    tmp += block_vector_.dot(gid, gid, 0);
    // cout<<"gid="<<gid<<", tmp="<<tmp<<endl;

    double norm     = 0.;
    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    mmpi.allreduce(&tmp, &norm, 1, MPI_SUM);

    return grid_.vel() * norm;
}

template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::orthonormalize2states(
    const int st1, const int st2)
{
    assert(st1 >= 0);
    assert(st2 >= 0);

    Control& ct = *(Control::instance());
    if (onpe0 && ct.verbose > 2)
        (*MPIdata::sout) << "ExtendedGridOrbitals::orthonormalize2states(): "
                         << st1 << " and " << st2 << std::endl;
    const int st[2] = { st1, st2 };

    double tmp[3]    = { 0., 0., 0. };
    const double vel = grid_.vel();

    for (int ic = 0; ic < 2; ic++)
    {
        const int color_ic = st[ic];

        // diagonal element
        tmp[2 * ic] += vel * block_vector_.dot(color_ic, color_ic, 0);

        if (ic == 1)
        {
            const int color_jc = st[0];

            tmp[1] += vel * block_vector_.dot(color_ic, color_jc, 0);
        }
    }

    double overlap[3] = { 0., 0., 0. };
    MGmol_MPI& mmpi   = *(MGmol_MPI::instance());
    mmpi.allreduce(&tmp[0], &overlap[0], 3, MPI_SUM);

    // orthogonalize second state
    double alpha = -overlap[1] / overlap[0];
    block_vector_.axpy(alpha, st[0], st[1], 0);

    // normalize both states
    const double alpha1 = 1. / sqrt(overlap[0]);
    const double alpha2
        = 1. / sqrt(overlap[2] - overlap[1] * overlap[1] / overlap[0]);
    block_vector_.scal(alpha1, st[0], 0);
    block_vector_.scal(alpha2, st[1], 0);

#if 1 // testing orthonormality
    tmp[0] = 0.;
    tmp[1] = 0.;
    tmp[2] = 0.;
    for (int ic = 0; ic < 2; ic++)
    {
        const int color_ic = st[ic];

        // diagonal element
        tmp[2 * ic] += vel * block_vector_.dot(color_ic, color_ic, 0);

        if (ic == 1)
        {
            const int color_jc = st[0];

            tmp[1] += vel * block_vector_.dot(color_ic, color_jc, 0);
        }
    }

    mmpi.allreduce(&tmp[0], &overlap[0], 3, MPI_SUM);
    if (onpe0 && ct.verbose > 2)
        (*MPIdata::sout) << "Gram matrix = " << overlap[0] << "," << overlap[1]
                         << "," << overlap[2] << std::endl;
#endif
}

template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::multiplyByMatrix2states(const int st1,
    const int st2, const double* mat, ExtendedGridOrbitals<ScalarType>& product)
{
    assert(st1 >= 0);
    assert(st2 >= 0);
    assert(1 == 1);

    // if( onpe0 && ct.verbose>2 )
    //  (*MPIdata::sout)<<"ExtendedGridOrbitals<ScalarType>::multiplyByMatrix2states()"<<endl;

    product.block_vector_.set_zero(st1, 0);
    product.block_vector_.set_zero(st2, 0);

    product.block_vector_.axpy(mat[0], block_vector_, st1, st1, 0);
    product.block_vector_.axpy(mat[3], block_vector_, st2, st2, 0);
    product.block_vector_.axpy(mat[2], block_vector_, st1, st2, 0);
    product.block_vector_.axpy(mat[1], block_vector_, st2, st1, 0);
}

template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::computeInvNorms2(
    std::vector<std::vector<double>>& inv_norms2) const
{
    std::vector<double> diagS(numst_);

    computeDiagonalElementsDotProduct(*this, diagS);

    inv_norms2.resize(1);
    inv_norms2[0].resize(numst_);

    for (short color = 0; color < numst_; color++)
    {
        double alpha = 1. / diagS[color];

        inv_norms2[0][color] = alpha;
    }
}

template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::normalize()
{
    normalize_tm_.start();

    assert(grid_.vel() > 1.e-8);
    assert(numst_ >= 0);

    // if( onpe0 && ct.verbose>2 )
    //        (*MPIdata::sout)<<"Normalize
    //        ExtendedGridOrbitals<ScalarType>"<<endl;

    //    const double vel = grid_.vel();
    std::vector<double> diagS(numst_);

    computeDiagonalElementsDotProduct(*this, diagS);

    for (int color = 0; color < numst_; color++)
    {
#ifdef DEBUG
        if (onpe0 && ct.verbose > 2)
            for (int i = 0; i < numst_; i++)
                (*MPIdata::sout)
                    << "i=" << i << ", diagS[i]=" << diagS[i] << std::endl;
#endif
        assert(diagS[color] > 1.e-15);
        diagS[color] = 1. / sqrt(diagS[color]);

        block_vector_.scal(diagS[color], color, 0);
    }

    incrementIterativeIndex();

    normalize_tm_.stop();
}

// modify argument orbitals, by projecting out its component
// along ExtendedGridOrbitals<ScalarType>
template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::projectOut(
    ExtendedGridOrbitals<ScalarType>& orbitals)
{
    projectOut(orbitals.psi(0), lda_);

#if 0
    // test if projection is now 0
    dist_matrix::DistMatrix<DISTMATDTYPE> tmatrix(product(orbitals));
    if( onpe0 )
        (*MPIdata::sout)<<"ExtendedGridOrbitals::projectOut(), Product after projection:"<<endl;
    tmatrix.print((*MPIdata::sout),0,0,5,5);
#endif

    orbitals.incrementIterativeIndex();
}

template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::projectOut(
    ScalarType* const array, const int lda)
{
    assert(lda > 1);
    assert(numpt_ > 0);
    assert(numst_ >= 0);
    assert(lda_ >= numpt_);

    SquareLocalMatrices<MATDTYPE, MemorySpace::Host> lmatrix(1, numst_);

    if (numst_ != 0) computeLocalProduct(array, lda, lmatrix, false);

#ifdef DEBUG
    (*MPIdata::sout) << "ExtendedGridOrbitals::projectOut()" << std::endl;
    (*MPIdata::sout) << "Product before projection" << std::endl;
    pmatrix.print((*MPIdata::sout));
#endif
    proj_matrices_->applyInvS(lmatrix);

    ScalarType* tproduct = new ScalarType[numpt_ * numst_];
    memset(tproduct, 0, numpt_ * numst_ * sizeof(ScalarType));

    unsigned int const phi_size = numpt_ * numst_;
    ScalarType* phi_host_view   = MemorySpace::Memory<ScalarType,
        memory_space_type>::allocate_host_view(phi_size);
    MemorySpace::Memory<ScalarType, memory_space_type>::copy_view_to_host(
        getPsi(0), phi_size, phi_host_view);

    MATDTYPE* localMat = lmatrix.getRawPtr();

    // TODO this can be done on the GPU
    // Compute numpt_ rows (for subdomain 0)
    gemm_nn_tm_.start();
    LinearAlgebraUtils<MemorySpace::Host>::MPgemmNN(numpt_, numst_, numst_, 1.,
        phi_host_view, lda_, localMat, numst_, 0., tproduct, numpt_);
    gemm_nn_tm_.stop();

    MemorySpace::Memory<ScalarType, memory_space_type>::free_host_view(
        phi_host_view);

    ScalarType* parray             = array + 0 * numpt_;
    unsigned int const parray_size = numst_ * lda;
    ScalarType* parray_host_view   = MemorySpace::Memory<ScalarType,
        memory_space_type>::allocate_host_view(parray_size);
    MemorySpace::Memory<ScalarType, memory_space_type>::copy_view_to_host(
        parray, parray_size, parray_host_view);

    ScalarType minus = -1.;
    for (int j = 0; j < numst_; j++)
        LinearAlgebraUtils<MemorySpace::Host>::MPaxpy(
            numpt_, minus, tproduct + j * numpt_, parray_host_view + j * lda);

    MemorySpace::Memory<ScalarType, memory_space_type>::copy_view_to_dev(
        parray_host_view, parray_size, parray);

    MemorySpace::Memory<ScalarType, memory_space_type>::free_host_view(
        parray_host_view);

    delete[] tproduct;
}

template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::initRand()
{
    Control& ct = *(Control::instance());

    const unsigned dim[3] = { grid_.dim(0), grid_.dim(1), grid_.dim(2) };

    std::vector<double> xrand(grid_.gdim(0));
    std::vector<double> yrand(grid_.gdim(1));
    std::vector<double> zrand(grid_.gdim(2));

    const int loc_length = dim[0] / 1;
    assert(loc_length > 0);
    assert(static_cast<unsigned int>(loc_length) <= dim[0]);

    const int xoff = grid_.istart(0);
    const int yoff = grid_.istart(1);
    const int zoff = grid_.istart(2);

    if (onpe0 && ct.verbose > 2)
        (*MPIdata::sout) << " Initialize " << numst_
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

        unsigned int const size    = numpt_;
        ScalarType* psi_state_view = MemorySpace::Memory<ScalarType,
            memory_space_type>::allocate_host_view(size);
        MemorySpace::Memory<ScalarType, memory_space_type>::copy_view_to_host(
            psi(istate), size, psi_state_view);

        for (int ix = loc_length * 0; ix < loc_length; ix++)
            for (unsigned int iy = 0; iy < dim[1]; iy++)
                for (unsigned int iz = 0; iz < dim[2]; iz++)
                {
                    const double alpha = xrand[xoff + ix] * yrand[yoff + iy]
                                         * zrand[zoff + iz];

                    psi_state_view[ix * incx + iy * incy + iz] = alpha * alpha;

                    assert((ix * incx + iy * incy + iz)
                           < static_cast<unsigned int>(lda_));
                }

        MemorySpace::Memory<ScalarType, memory_space_type>::copy_view_to_dev(
            psi_state_view, size, psi(istate));
        MemorySpace::Memory<ScalarType, memory_space_type>::free_host_view(
            psi_state_view);
    }

    resetIterativeIndex();
}

#ifdef MGMOL_USE_SCALAPACK
template <>
template <>
void ExtendedGridOrbitals<ORBDTYPE>::addDotWithNcol2Matrix(
    ExtendedGridOrbitals<ORBDTYPE>& Apsi,
    dist_matrix::DistMatrix<double>& matrix) const
{
    addDotWithNcol2DistMatrix(Apsi, matrix);
}

template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::addDotWithNcol2DistMatrix(
    ExtendedGridOrbitals<ScalarType>& Apsi,
    dist_matrix::DistMatrix<double>& matrix) const
{
    addDot_tm_.start();

    assert(numst_ > 0);

    Control& ct = *(Control::instance());
    if (onpe0 && ct.verbose > 2)
        std::cout << "ExtendedGridOrbitals::addDotWithNcol2DistMatrix()..."
                  << std::endl;

    const double vel = grid_.vel();

    // replicated matrix
    const int size_work = numst_ * numst_;
    std::vector<double> work(size_work);

    unsigned int const block_vector_size = numpt_ * numst_;
    ScalarType* block_vector_host_view   = MemorySpace::Memory<ScalarType,
        memory_space_type>::allocate_host_view(block_vector_size);
    MemorySpace::Memory<ScalarType, memory_space_type>::copy_view_to_host(
        block_vector_.vect(0), block_vector_size, block_vector_host_view);

    unsigned int const phi_size = numpt_ * numst_;
    ScalarType* phi_host_view   = MemorySpace::Memory<ScalarType,
        memory_space_type>::allocate_host_view(phi_size);
    MemorySpace::Memory<ScalarType, memory_space_type>::copy_view_to_host(
        Apsi.getPsi(0), phi_size, phi_host_view);

    // TODO this can be done on the GPU
    gemm_tn_tm_.start();
    LinearAlgebraUtils<memory_space_type>::MPgemmTN(numst_, numst_, numpt_, vel,
        block_vector_host_view + 0 * numpt_, lda_, phi_host_view, lda_, 0.,
        work.data(), numst_);
    gemm_tn_tm_.stop();

    MemorySpace::Memory<ScalarType, memory_space_type>::free_host_view(
        phi_host_view);

    MemorySpace::Memory<ScalarType, memory_space_type>::free_host_view(
        block_vector_host_view);

    std::vector<double> work2(size_work);
    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    mmpi.allreduce(work.data(), work2.data(), numst_ * numst_, MPI_SUM);

    // replicated -> DistMatrix
    matrix.add(work2.data(), numst_);

    addDot_tm_.stop();
}
#endif

template <>
template <>
void ExtendedGridOrbitals<ORBDTYPE>::addDotWithNcol2Matrix(
    ExtendedGridOrbitals<ORBDTYPE>& Apsi, ReplicatedMatrix& matrix) const
{
    addDotWithNcol2ReplicatedMatrix(Apsi, matrix);
}

template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::addDotWithNcol2ReplicatedMatrix(
    ExtendedGridOrbitals<ScalarType>& Apsi, ReplicatedMatrix& matrix) const
{
    addDot_tm_.start();

    Control& ct = *(Control::instance());
    if (onpe0 && ct.verbose > 2)
        std::cout
            << "ExtendedGridOrbitals::addDotWithNcol2ReplicatedMatrix()..."
            << std::endl;

    ReplicatedMatrix tmp("tmp", numst_, numst_);
    const double vel = grid_.vel();

    gemm_tn_tm_.start();
#ifdef HAVE_MAGMA
    magma_trans_t magma_transa = magma_trans_const('t');
    magma_trans_t magma_transb = magma_trans_const('n');

    auto& magma_singleton = MagmaSingleton::get_magma_singleton();

    magmablas_dgemm(magma_transa, magma_transb, numst_, numst_, numpt_, vel,
        block_vector_.vect(0), lda_, Apsi.getPsi(0), lda_, 0., tmp.data(),
        tmp.ld(), magma_singleton.queue_);
#else
    LinearAlgebraUtils<memory_space_type>::MPgemmTN(numst_, numst_, numpt_, vel,
        block_vector_.vect(0), lda_, Apsi.getPsi(0), lda_, 0., tmp.data(),
        tmp.ld());
#endif
    gemm_tn_tm_.stop();

    tmp.consolidate();

    matrix.axpy(1., tmp);

    addDot_tm_.stop();
}

template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::computeGlobalIndexes()
{
    overlapping_gids_.clear();
    overlapping_gids_.resize(1);
    overlapping_gids_[0].resize(numst_, -1);
    for (int gid = 0; gid < numst_; gid++)
    {
        overlapping_gids_[0][gid] = gid;
    }
}

template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::printTimers(std::ostream& os)
{
    matB_tm_.print(os);
    invBmat_tm_.print(os);
    overlap_tm_.print(os);
    dot_product_tm_.print(os);
    addDot_tm_.print(os);
    prod_matrix_tm_.print(os);
    gemm_nn_tm_.print(os);
    gemm_tn_tm_.print(os);
    assign_tm_.print(os);
    normalize_tm_.print(os);
    axpy_tm_.print(os);
}

template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::initWF(
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
                for (short icolor = 0; icolor < numst_; icolor++)
                {
                    gf_psi.assign(psi(icolor));
                    myoper.rhs(gf_psi, gf_work);
                    setPsi(gf_work, icolor);
                }
            }
    }

    // needs to mask one layer of values when using 0 BC for
    // wavefunctions the next two lines do that
    setDataWithGhosts();
    trade_boundaries();
    setToDataWithGhosts();

    resetIterativeIndex();

    if (onpe0 && ct.verbose > 2)
        (*MPIdata::sout)
            << " Normalize or Orthonormalize initial wave functions"
            << std::endl;
    if (ct.isLocMode())
    {
        normalize();
        // ortho_norm_local();
    }
    else
    {
        // orthonormalize();
        orthonormalizeLoewdin();
    }

    setDataWithGhosts();

    // needs to mask one layer of values when using 0 BC for wavefunctions
    // the next two lines do that
    trade_boundaries();
    setToDataWithGhosts();

#ifdef DEBUG
    if (onpe0 && ct.verbose > 2)
        (*MPIdata::sout) << "ExtendedGridOrbitals::init_wf() done" << std::endl;
#endif
}

#ifdef MGMOL_HAS_LIBROM
template <typename ScalarType>
void ExtendedGridOrbitals<ScalarType>::set(std::string file_path, int rdim)
{
    const int dim = getLocNumpt();

    CAROM::BasisReader reader(file_path);
    CAROM::Matrix* orbital_basis = reader.getSpatialBasis(rdim);

    Control& ct  = *(Control::instance());
    Mesh* mymesh = Mesh::instance();
    pb::GridFunc<ORBDTYPE> gf_psi(
        mymesh->grid(), ct.bcWF[0], ct.bcWF[1], ct.bcWF[2]);
    CAROM::Vector psi;
    for (int i = 0; i < rdim; ++i)
    {
        orbital_basis->getColumn(i, psi);
        gf_psi.assign(psi.getData());
        setPsi(gf_psi, i);
    }
}
#endif

template void ExtendedGridOrbitals<ORBDTYPE>::axpy(
    const ORBDTYPE alpha, const ExtendedGridOrbitals<ORBDTYPE>&);

template class ExtendedGridOrbitals<ORBDTYPE>;

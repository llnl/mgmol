// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE
#include "ReplicatedMatrix.h"
#include "LocalMatrices2ReplicatedMatrix.h"
#include "ReplicatedVector.h"
#include "memory_space.h"
#include "random.h"

#ifdef HAVE_MAGMA
#include "magma_v2.h"
using Memory                = MemorySpace::Memory<double, MemorySpace::Device>;
constexpr double gpuroundup = 32;
#else
#include "MGmol_blas1.h"
#include "blas3_c.h"
#include "fc_mangle.h"
#include "lapack_c.h"
using Memory = MemorySpace::Memory<double, MemorySpace::Host>;
#endif

#include <iostream>

MPI_Comm ReplicatedMatrix::comm_ = MPI_COMM_NULL;
bool ReplicatedMatrix::onpe0_    = false;

static int roundup(const int n)
{
#ifdef HAVE_MAGMA
    return magma_roundup(n, gpuroundup);
#else
    return n;
#endif
}

void rotateSym(ReplicatedMatrix& mat, const ReplicatedMatrix& rotation_matrix,
    ReplicatedMatrix& work)
{
    work.symm('l', 'l', 1., mat, rotation_matrix, 0.);
    mat.gemm('t', 'n', 1., rotation_matrix, work, 0.);
}

void getProcrustesTransform(ReplicatedMatrix& q, ReplicatedMatrix& yyt)
{
    const bool onpe0 = ReplicatedMatrix::onpe0_;

    if (onpe0) std::cout << "getProcrustesTransform()" << std::endl;

    const int nst    = q.m();
    const double tol = 0.5;

    ReplicatedMatrix z("z", nst, nst);
    ReplicatedMatrix yt("yt", nst, nst);

    std::vector<double> eigenvalues(nst);

    // get z and yt
    q.gesvd('V', 'V', eigenvalues, z, yt);
    if (onpe0)
    {
        std::cout << "Procrustes: Small singular values: ";
        for (int i = 0; i < nst; i++)
            if (eigenvalues[i] < tol) std::cout << "  " << eigenvalues[i];
        std::cout << std::endl;
    }

    // get q=z*yt
    q.gemm('n', 'n', 1., z, yt, 0.);

    // now calculate yyt matrix
    std::vector<double> diag(nst);
    for (int i = 0; i < nst; i++)
        diag[i] = (eigenvalues[i] < tol) ? 0. : 1.;
    ReplicatedMatrix gamma("Gamma", diag.data(), nst);
    z.gemm('t', 'n', 1., yt, gamma, 0.);
    yyt.gemm('n', 'n', 1., z, yt, 0.);
}

ReplicatedMatrix::ReplicatedMatrix(
    const std::string name, const int m, const int n)
    : dim_(m),
      ld_(roundup(dim_)),
      data_(Memory::allocate(dim_ * ld_), Memory::free),
      name_(name)
{
    assert(m == n);

    clear();
}

ReplicatedMatrix::ReplicatedMatrix(const std::string name, const int n)
    : dim_(n),
      ld_(roundup(n)),
      data_(Memory::allocate(dim_ * ld_), Memory::free),
      name_(name)
{
    clear();
}

ReplicatedMatrix::ReplicatedMatrix(
    const std::string name, const double* const diagonal, const int m)
    : dim_(m),
      ld_(roundup(dim_)),
      data_(Memory::allocate(dim_ * ld_), Memory::free),
      name_(name)
{
    clear();

#ifdef HAVE_MAGMA
    auto& magma_singleton = MagmaSingleton::get_magma_singleton();

    magma_dsetvector(
        dim_, diagonal, 1, data_.get(), ld_ + 1, magma_singleton.queue_);
#else
    int ione = 1;
    int incy = ld_ + 1;
    DCOPY(&dim_, diagonal, &ione, data_.get(), &incy);
#endif
}

ReplicatedMatrix::ReplicatedMatrix(const ReplicatedMatrix& mat)
    : dim_(mat.dim_),
      ld_(mat.ld_),
      data_(Memory::allocate(dim_ * ld_), Memory::free)
{
#ifdef HAVE_MAGMA
    auto& magma_singleton = MagmaSingleton::get_magma_singleton();

    magma_dcopymatrix(dim_, dim_, mat.data_.get(), mat.ld_, data_.get(), ld_,
        magma_singleton.queue_);
#else
    memcpy(data_.get(), mat.data_.get(), ld_ * dim_ * sizeof(double));
#endif
}

ReplicatedMatrix& ReplicatedMatrix::operator=(const ReplicatedMatrix& rhs)
{
    if (this != &rhs)
    {
        ld_  = rhs.ld_;
        dim_ = rhs.dim_;
        data_.reset(Memory::allocate(dim_ * ld_));

#ifdef HAVE_MAGMA
        auto& magma_singleton = MagmaSingleton::get_magma_singleton();

        magma_dcopymatrix(dim_, dim_, rhs.data_.get(), rhs.ld_, data_.get(),
            ld_, magma_singleton.queue_);
#else
        memcpy(data_.get(), rhs.data_.get(), ld_ * dim_ * sizeof(double));
#endif
    }
    return *this;
}

ReplicatedMatrix::~ReplicatedMatrix() { }

void ReplicatedMatrix::getsub(
    const ReplicatedMatrix& src, int m, int n, int ia, int ja)
{
#ifdef HAVE_MAGMA

    auto& magma_singleton = MagmaSingleton::get_magma_singleton();

    magma_dcopymatrix(m, n, src.data_.get() + ja * src.ld_ + ia, src.ld_,
        data_.get(), ld_, magma_singleton.queue_);
#else
    char uplo = 'a';
    int lda   = src.ld_;
    int ldb   = ld_;
    DLACPY(&uplo, &m, &n, src.data_.get() + ja * src.ld_ + ia, &lda,
        data_.get(), &ldb);
#endif
}

void ReplicatedMatrix::consolidate()
{
    assert(comm_ != MPI_COMM_NULL);

    std::vector<double> mat(dim_ * ld_);
#ifdef HAVE_MAGMA
    std::vector<double> mat_sum(dim_ * ld_);
    double* mat_sum_data = mat_sum.data();

    auto& magma_singleton = MagmaSingleton::get_magma_singleton();

    // copy from GPU to CPU
    magma_dgetmatrix(
        dim_, dim_, data_.get(), ld_, mat.data(), ld_, magma_singleton.queue_);
#else
    double* mat_sum_data = data_.get();
    memcpy(mat.data(), data_.get(), dim_ * ld_ * sizeof(double));
#endif
    MPI_Allreduce(
        mat.data(), mat_sum_data, dim_ * ld_, MPI_DOUBLE, MPI_SUM, comm_);

#ifdef HAVE_MAGMA
    // copy from CPU to GPU
    magma_dsetmatrix(dim_, dim_, mat_sum.data(), ld_, data_.get(), ld_,
        magma_singleton.queue_);
#endif
}

void ReplicatedMatrix::bcast(const int root)
{
    assert(comm_ != MPI_COMM_NULL);

#ifdef HAVE_MAGMA
    std::vector<double> mat(dim_ * ld_);
    auto& magma_singleton = MagmaSingleton::get_magma_singleton();

    // copy from GPU to CPU
    magma_dgetmatrix(
        dim_, dim_, data_.get(), ld_, mat.data(), ld_, magma_singleton.queue_);
    double* data = mat.data();
#else
    double* data = data_.get();
#endif
    MPI_Bcast(data, dim_ * ld_, MPI_DOUBLE, root, comm_);

#ifdef HAVE_MAGMA
    // copy from CPU to GPU
    magma_dsetmatrix(
        dim_, dim_, data, ld_, data_.get(), ld_, magma_singleton.queue_);
#endif
}

void ReplicatedMatrix::assign(
    const ReplicatedMatrix& src, const int ib, const int jb)
{
    assert(this != &src);
#ifdef HAVE_MAGMA

    auto& magma_singleton = MagmaSingleton::get_magma_singleton();

    magma_dcopymatrix(src.dim_, src.dim_, src.data_.get(), src.ld_,
        data_.get() + jb * ld_ + ib, ld_, magma_singleton.queue_);
#else
    char uplo    = 'a';
    int dim      = src.dim_;
    int lda      = src.ld_;
    int ldb      = ld_;
    DLACPY(&uplo, &dim, &dim, src.data_.get(), &lda,
        data_.get() + jb * ld_ + ib, &ldb);
#endif
}

template <>
void ReplicatedMatrix::assign(
    SquareLocalMatrices<double, MemorySpace::Host>& src)
{
#ifdef HAVE_MAGMA
    auto& magma_singleton = MagmaSingleton::get_magma_singleton();

    magma_dsetmatrix(src.m(), src.n(), src.getSubMatrix(), src.n(), data_.get(),
        ld_, magma_singleton.queue_);
#else
    LocalMatrices2ReplicatedMatrix* l2r
        = LocalMatrices2ReplicatedMatrix::instance();
    l2r->convert(src, *this, dim_, 0.);
#endif
}

template <>
void ReplicatedMatrix::assign(
    SquareLocalMatrices<double, MemorySpace::Device>& src)
{
    assert(src.n() == dim_);

    // current implementation restriction
    assert(src.nmat() == 1);

#ifdef HAVE_MAGMA
    auto& magma_singleton = MagmaSingleton::get_magma_singleton();

    magma_dcopymatrix(src.n(), src.n(), src.getRawPtr(), src.n(), data_.get(),
        ld_, magma_singleton.queue_);
#else
    // copy columns of matrix
    for (int j = 0; j < dim_; j++)
        memcpy(data_.get() + j * ld_, src.getRawPtr() + j * src.n(),
            dim_ * sizeof(double));
#endif
}

void ReplicatedMatrix::assign(const double* const src, const int ld)
{
#ifdef HAVE_MAGMA
    auto& magma_singleton = MagmaSingleton::get_magma_singleton();

    magma_dcopymatrix(
        dim_, dim_, src, ld, data_.get(), ld_, magma_singleton.queue_);
#else
    // copy columns of matrix
    for (int j = 0; j < dim_; j++)
        memcpy(data_.get() + j * ld_, src + j * ld, dim_ * sizeof(double));
#endif
}

void ReplicatedMatrix::add(const SquareSubMatrix<double>& mat)
{
    const std::vector<int>& gid(mat.getGids());
    const int n = gid.size();
    assert(n == dim_);

    std::vector<double> src(n * n);

    for (int j = 0; j < n; j++)
    {
        assert(gid[j] >= 0);

        for (int i = 0; i < n; i++)
        {
            src[i + j * n] = mat.getLocalValue(i, j);
        }
    }

#ifdef HAVE_MAGMA
    std::unique_ptr<double, void (*)(double*)> src_dev(
        Memory::allocate(dim_ * ld_), Memory::free);

    auto& magma_singleton = MagmaSingleton::get_magma_singleton();

    // copy to GPU
    magma_dsetmatrix(dim_, dim_, src.data(), dim_, src_dev.get(), ld_,
        magma_singleton.queue_);

    // add to object data
    magmablas_dgeadd(dim_, dim_, 1., src_dev.get(), ld_, data_.get(), ld_,
        magma_singleton.queue_);
#else
    double* data = data_.get();
    for (int j = 0; j < dim_; j++)
        for (int i = 0; i < dim_; i++)
            data[i + j * ld_] += src[i + j * n];
#endif
}

void ReplicatedMatrix::init(const double* const ha, const int lda)
{
#ifdef HAVE_MAGMA
    auto& magma_singleton = MagmaSingleton::get_magma_singleton();

    magma_dsetmatrix(
        dim_, dim_, ha, lda, data_.get(), ld_, magma_singleton.queue_);
#else
    for (int j = 0; j < dim_; j++)
        memcpy(data_.get() + ld_ * j, ha + lda * j, dim_ * sizeof(double));
#endif
}

void ReplicatedMatrix::get(double* ha, const int lda) const
{
    assert(ha != nullptr);
#ifdef HAVE_MAGMA
    auto& magma_singleton = MagmaSingleton::get_magma_singleton();

    magma_dgetmatrix(
        dim_, dim_, data_.get(), ld_, ha, lda, magma_singleton.queue_);
#else
    for (int j = 0; j < dim_; j++)
        memcpy(ha + lda * j, data_.get() + ld_ * j, dim_ * sizeof(double));
#endif
}

void ReplicatedMatrix::getDiagonalValues(double* ha)
{
#ifdef HAVE_MAGMA
    auto& magma_singleton = MagmaSingleton::get_magma_singleton();

    magma_dgetvector(dim_, data_.get(), ld_ + 1, ha, 1, magma_singleton.queue_);
#else
    int dim  = dim_;
    int incx = ld_ + 1;
    int incy = 1;
    DCOPY(&dim, data_.get(), &incx, ha, &incy);
#endif
}

void ReplicatedMatrix::axpy(const double alpha, const ReplicatedMatrix& a)
{
#ifdef HAVE_MAGMA
    auto& magma_singleton = MagmaSingleton::get_magma_singleton();

    magmablas_dgeadd(dim_, dim_, alpha, a.data_.get(), a.ld_, data_.get(), ld_,
        magma_singleton.queue_);
#else
    int dim  = dim_ * ld_;
    int ione = 1;
    DAXPY(&dim, &alpha, a.data_.get(), &ione, data_.get(), &ione);
#endif
}

void ReplicatedMatrix::setRandom(const double minv, const double maxv)
{
    std::vector<double> mat(dim_ * dim_);

    generateRandomData(mat, minv, maxv);

#ifdef HAVE_MAGMA
    auto& magma_singleton = MagmaSingleton::get_magma_singleton();

    magma_dsetmatrix(
        dim_, dim_, mat.data(), dim_, data_.get(), ld_, magma_singleton.queue_);
#else
    double* data = data_.get();
    for (int j = 0; j < dim_; j++)
        for (int i = 0; i < dim_ * (int)ld_; i++)
            data[j * ld_ + i] = mat[j * dim_ + i];
#endif
}

void ReplicatedMatrix::identity()
{
#ifdef HAVE_MAGMA
    auto& magma_singleton = MagmaSingleton::get_magma_singleton();

    magmablas_dlaset(MagmaFull, dim_, dim_, 0.0, 1.0, data_.get(), ld_,
        magma_singleton.queue_);
#else
    double* data = data_.get();
    memset(data, 0, dim_ * ld_ * sizeof(double));
    for (int i = 0; i < dim_; i++)
        data[i * ld_ + i] = 1.;
#endif
}

void ReplicatedMatrix::scal(const double alpha)
{
    int size = dim_ * ld_;
#ifdef HAVE_MAGMA
    auto& magma_singleton = MagmaSingleton::get_magma_singleton();

    magma_dscal(size, alpha, data_.get(), 1, magma_singleton.queue_);
#else
    int ione = 1;
    DSCAL(&size, &alpha, data_.get(), &ione);
#endif
}

// this = alpha * transpose(A) + beta * this
void ReplicatedMatrix::transpose(
    const double alpha, const ReplicatedMatrix& a, const double beta)
{
#ifdef HAVE_MAGMA
    auto& magma_singleton = MagmaSingleton::get_magma_singleton();

    double* dwork;
    magma_int_t ret = magma_dmalloc(&dwork, dim_ * ld_);
    if (ret != MAGMA_SUCCESS)
    {
        std::cerr << "magma_dmalloc failed!" << std::endl;
    }

    magmablas_dtranspose(
        dim_, dim_, a.data_.get(), a.ld_, dwork, ld_, magma_singleton.queue_);

    magmablas_dgeadd2(dim_, dim_, alpha, dwork, ld_, beta, data_.get(), ld_,
        magma_singleton.queue_);

    magma_singleton.sync();
    magma_free(dwork);
#else
    double* data  = data_.get();
    double* adata = a.data_.get();
    for (int i = 0; i < dim_; i++)
    {
        for (int j = 0; j < dim_; j++)
        {
            data[j * ld_ + i]
                = beta * data[j * ld_ + i] + alpha * adata[i * ld_ + j];
        }
    }
#endif
}

void ReplicatedMatrix::gemm(const char transa, const char transb,
    const double alpha, const ReplicatedMatrix& a, const ReplicatedMatrix& b,
    const double beta)
{
#ifdef HAVE_MAGMA
    magma_trans_t magma_transa = magma_trans_const(transa);
    magma_trans_t magma_transb = magma_trans_const(transb);

    auto& magma_singleton = MagmaSingleton::get_magma_singleton();

    magmablas_dgemm(magma_transa, magma_transb, dim_, dim_, dim_, alpha,
        a.data_.get(), a.ld_, b.data_.get(), b.ld_, beta, data_.get(), ld_,
        magma_singleton.queue_);
#else
    int ld  = ld_;
    int ald = a.ld_;
    int bld = b.ld_;
    DGEMM(&transa, &transb, &dim_, &dim_, &dim_, &alpha, a.data_.get(), &ald,
        b.data_.get(), &bld, &beta, data_.get(), &ld);
#endif
}

void ReplicatedMatrix::symm(const char side, const char uplo,
    const double alpha, const ReplicatedMatrix& a, const ReplicatedMatrix& b,
    const double beta)
{
#ifdef HAVE_MAGMA
    magma_side_t magma_side = magma_side_const(side);
    magma_uplo_t magma_uplo = magma_uplo_const(uplo);

    auto& magma_singleton = MagmaSingleton::get_magma_singleton();

    magma_dsymm(magma_side, magma_uplo, dim_, dim_, alpha, a.data_.get(), a.ld_,
        b.data_.get(), b.ld_, beta, data_.get(), ld_, magma_singleton.queue_);
#else
    int ld  = ld_;
    int ald = a.ld_;
    int bld = b.ld_;
    DSYMM(&side, &uplo, &dim_, &dim_, &alpha, a.data_.get(), &ald,
        b.data_.get(), &bld, &beta, data_.get(), &ld);
#endif
}

int ReplicatedMatrix::potrf(char uplo)
{
    assert(data_.get());

    int info;
#ifdef HAVE_MAGMA
    magma_uplo_t magma_uplo = magma_uplo_const(uplo);

    magma_dpotrf_gpu(magma_uplo, dim_, data_.get(), ld_, &info);
#else
    int ld = ld_;
    DPOTRF(&uplo, &dim_, data_.get(), &ld, &info);
#endif
    if (info != 0)
        std::cerr << "ReplicatedMatrix::potrf() failed, info = " << info
                  << std::endl;

    return info;
}

void ReplicatedMatrix::getrf(std::vector<int>& ipiv)
{
    int info;
#ifdef HAVE_MAGMA
    magma_dgetrf_gpu(dim_, dim_, data_.get(), ld_, ipiv.data(), &info);
#else
    int ld = ld_;
    DGETRF(&dim_, &dim_, data_.get(), &ld, ipiv.data(), &info);
#endif
    if (info != 0)
        std::cerr << "magma_dgetrf_gpu failed, info = " << info << std::endl;
}

int ReplicatedMatrix::potri(char uplo)
{
    int info;
#ifdef HAVE_MAGMA
    magma_uplo_t magma_uplo = magma_uplo_const(uplo);

    magma_dpotri_gpu(magma_uplo, dim_, data_.get(), ld_, &info);
#else
    int ld = ld_;
    DPOTRI(&uplo, &dim_, data_.get(), &ld, &info);
#endif
    if (info != 0)
        std::cerr << "magma_dpotri_gpu failed, info = " << info << std::endl;

    return info;
}

// Solve a system of linear equations A*X = B with a symmetric
// positive definite matrix A using the Cholesky factorization
// A = U**T*U or A = L*L**T computed by potrf
void ReplicatedMatrix::potrs(char uplo, ReplicatedMatrix& b)
{
    int info;
#ifdef HAVE_MAGMA
    magma_uplo_t magma_uplo = magma_uplo_const(uplo);

    magma_dpotrs_gpu(
        magma_uplo, dim_, 1, data_.get(), ld_, b.data(), dim_, &info);
#else
    int ione = 1;
    int ld   = ld_;
    DPOTRS(&uplo, &dim_, &ione, data_.get(), &ld, b.data(), &dim_, &info);
#endif
    if (info != 0) std::cerr << "dpotrs failed, info = " << info << std::endl;
}

void ReplicatedMatrix::potrs(char uplo, ReplicatedVector& b)
{
    int info;
#ifdef HAVE_MAGMA
    magma_uplo_t magma_uplo = magma_uplo_const(uplo);

    magma_dpotrs_gpu(
        magma_uplo, dim_, 1, data_.get(), ld_, b.data(), dim_, &info);
#else
    int ione = 1;
    int ld   = ld_;
    DPOTRS(&uplo, &dim_, &ione, data_.get(), &ld, b.data(), &dim_, &info);
#endif
    if (info != 0) std::cerr << "dpotrs failed, info = " << info << std::endl;
}

void ReplicatedMatrix::getrs(
    char trans, ReplicatedMatrix& b, std::vector<int>& ipiv)
{
    int info;
#ifdef HAVE_MAGMA
    magma_trans_t magma_trans = magma_trans_const(trans);

    magma_dgetrs_gpu(magma_trans, dim_, dim_, data_.get(), ld_, ipiv.data(),
        b.data_.get(), b.ld_, &info);
#else
    int ld  = ld_;
    int bld = b.ld_;
    DGETRS(&trans, &dim_, &dim_, data_.get(), &ld, ipiv.data(), b.data_.get(),
        &bld, &info);
#endif
    if (info != 0)
        std::cerr << "magma_dgetrs_gpu failed, info = " << info << std::endl;
}

void ReplicatedMatrix::syev(
    char jobz, char uplo, std::vector<double>& evals, ReplicatedMatrix& z)
{
    int info;
#ifdef HAVE_MAGMA
    magma_vec_t magma_jobz  = magma_vec_const(jobz);
    magma_uplo_t magma_uplo = magma_uplo_const(uplo);

    auto& magma_singleton = MagmaSingleton::get_magma_singleton();

    // copy matrix into z
    magmablas_dlacpy(MagmaFull, dim_, dim_, data_.get(), ld_, z.data_.get(),
        z.ld_, magma_singleton.queue_);
    magma_int_t nb = magma_get_ssytrd_nb(dim_);
    magma_int_t lwork
        = std::max(2 * dim_ + dim_ * nb, 1 + 6 * dim_ + 2 * dim_ * dim_);

    int liwork = 3 + 5 * dim_;

    std::vector<double> wa(dim_ * dim_);
    std::vector<double> work(lwork);
    std::vector<int> iwork(liwork);

    magma_dsyevd_gpu(magma_jobz, magma_uplo, dim_, z.data_.get(), z.ld_,
        evals.data(), wa.data(), dim_, work.data(), lwork, iwork.data(), liwork,
        &info);
#else
    memcpy(z.data_.get(), data_.get(), dim_ * ld_ * sizeof(double));
    int lwork = 3 * dim_ - 1;
    std::vector<double> work(lwork);
    int zld = z.ld_;
    DSYEV(&jobz, &uplo, &dim_, z.data_.get(), &zld, evals.data(), work.data(),
        &lwork, &info);
#endif
    if (info != 0)
        std::cerr << "ReplicatedMatrix::syev() failed, info = " << info
                  << std::endl;
    // for(auto& d : evals)std::cout<<d<<std::endl;
}

// singular value decomposition: (*this) = u * diag(s) * vt
// (the content of this matrix is destroyed)
void ReplicatedMatrix::gesvd(const char jobu, const char jobvt,
    std::vector<double>& s, ReplicatedMatrix& u, ReplicatedMatrix& vt)
{
    assert(u.dim_ == dim_);
    assert(vt.dim_ == dim_);
    assert(static_cast<int>(s.size()) >= dim_);

    // 'V' (compute the min(m,n) first vectors) is used with ScaLAPACK,
    // but it is 'S' for LAPACK and MAGMA
    const char ju  = (jobu == 'V' || jobu == 'v') ? 'S' : jobu;
    const char jvt = (jobvt == 'V' || jobvt == 'v') ? 'S' : jobvt;

    int info;
#ifdef HAVE_MAGMA
    magma_vec_t magma_jobu  = magma_vec_const(ju);
    magma_vec_t magma_jobvt = magma_vec_const(jvt);

    auto& magma_singleton = MagmaSingleton::get_magma_singleton();

    // the CPU interface of MAGMA is used: copy data from GPU to CPU
    std::vector<double> a_host(dim_ * dim_);
    std::vector<double> u_host(dim_ * dim_);
    std::vector<double> vt_host(dim_ * dim_);
    magma_dgetmatrix(dim_, dim_, data_.get(), ld_, a_host.data(), dim_,
        magma_singleton.queue_);

    magma_int_t lwork = -1;
    double qwork;
    magma_dgesvd(magma_jobu, magma_jobvt, dim_, dim_, a_host.data(), dim_,
        s.data(), u_host.data(), dim_, vt_host.data(), dim_, &qwork, lwork,
        &info);
    lwork = static_cast<magma_int_t>(qwork);
    std::vector<double> work(lwork);
    magma_dgesvd(magma_jobu, magma_jobvt, dim_, dim_, a_host.data(), dim_,
        s.data(), u_host.data(), dim_, vt_host.data(), dim_, work.data(), lwork,
        &info);

    // copy results back to GPU
    magma_dsetmatrix(dim_, dim_, u_host.data(), dim_, u.data_.get(), u.ld_,
        magma_singleton.queue_);
    magma_dsetmatrix(dim_, dim_, vt_host.data(), dim_, vt.data_.get(), vt.ld_,
        magma_singleton.queue_);
#else
    int ld   = ld_;
    int ldu  = u.ld_;
    int ldvt = vt.ld_;

    // query optimal size of workspace
    int lwork = -1;
    double qwork;
    DGESVD(&ju, &jvt, &dim_, &dim_, data_.get(), &ld, s.data(), u.data_.get(),
        &ldu, vt.data_.get(), &ldvt, &qwork, &lwork, &info);
    if (info == 0)
    {
        lwork = static_cast<int>(qwork);
        std::vector<double> work(lwork);
        DGESVD(&ju, &jvt, &dim_, &dim_, data_.get(), &ld, s.data(),
            u.data_.get(), &ldu, vt.data_.get(), &ldvt, work.data(), &lwork,
            &info);
    }
#endif
    if (info != 0)
        std::cerr << "ReplicatedMatrix::gesvd() failed, info = " << info
                  << std::endl;
}

void ReplicatedMatrix::sygst(int itype, char uplo, const ReplicatedMatrix& b)
{
    int info;
#ifdef HAVE_MAGMA
    magma_uplo_t magma_uplo = magma_uplo_const(uplo);
    magma_int_t magma_itype = static_cast<magma_int_t>(itype);

    magma_dsygst_gpu(magma_itype, magma_uplo, dim_, data_.get(), ld_,
        b.data_.get(), b.ld_, &info);
#else
    int ld  = ld_;
    int bld = b.ld_;
    DSYGST(&itype, &uplo, &dim_, data_.get(), &ld, b.data_.get(), &bld, &info);
#endif
    if (info != 0)
        std::cerr << "magma_dsygst_gpu failed, info = " << info << std::endl;
}

void ReplicatedMatrix::trmm(const char side, const char uplo, const char trans,
    const char diag, const double alpha, const ReplicatedMatrix& a)
{
#ifdef HAVE_MAGMA
    magma_side_t magma_side   = magma_side_const(side);
    magma_uplo_t magma_uplo   = magma_uplo_const(uplo);
    magma_trans_t magma_trans = magma_trans_const(trans);
    magma_diag_t magma_diag   = magma_diag_const(diag);

    auto& magma_singleton = MagmaSingleton::get_magma_singleton();

    magma_dtrmm(magma_side, magma_uplo, magma_trans, magma_diag, dim_, dim_,
        alpha, a.data_.get(), a.ld_, data_.get(), ld_, magma_singleton.queue_);
#else
    int ld  = ld_;
    int ald = a.ld_;
    DTRMM(&side, &uplo, &trans, &diag, &dim_, &dim_, &alpha, a.data_.get(),
        &ald, data_.get(), &ld);
#endif
}

void ReplicatedMatrix::trtrs(const char uplo, const char trans, const char diag,
    ReplicatedMatrix& b) const
{
#ifdef HAVE_MAGMA
    magma_uplo_t magma_uplo   = magma_uplo_const(uplo);
    magma_trans_t magma_trans = magma_trans_const(trans);
    magma_diag_t magma_diag   = magma_diag_const(diag);

    auto& magma_singleton = MagmaSingleton::get_magma_singleton();

    magma_dtrsm(MagmaLeft, magma_uplo, magma_trans, magma_diag, dim_, dim_, 1.,
        data_.get(), ld_, b.data_.get(), b.ld_, magma_singleton.queue_);
#else
    double one = 1.;
    char side  = 'L';
    int ld     = ld_;
    int bld    = b.ld_;
    DTRSM(&side, &uplo, &trans, &diag, &dim_, &dim_, &one, data_.get(), &ld,
        b.data_.get(), &bld);
#endif
}

// get max in absolute value of column j
int ReplicatedMatrix::iamax(const int j, double& val)
{
#ifdef HAVE_MAGMA
    auto& magma_singleton = MagmaSingleton::get_magma_singleton();

    int indx
        = magma_idamax(dim_, data_.get() + j * ld_, 1, magma_singleton.queue_)
          - 1;
    magma_dgetvector(
        dim_, data_.get() + j * ld_ + indx, 1, &val, 1, magma_singleton.queue_);

    return indx;
#else
    int ione = 1;
    int indx = IDAMAX(&dim_, data_.get() + j * ld_, &ione) - 1;
    val      = *(data_.get() + j * ld_ + indx);
#endif
    return indx;
}

double ReplicatedMatrix::nrm2(const int j)
{
#ifdef HAVE_MAGMA
    (void)os;
    std::cerr << "ReplicatedMatrix::nrm2() not implemented" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#else
    int ione = 1;
    return DNRM2(&dim_, data_.get() + j * ld_, &ione);
#endif
}

void ReplicatedMatrix::setVal(const int i, const int j, const double val)
{
#ifdef HAVE_MAGMA
    auto& magma_singleton = MagmaSingleton::get_magma_singleton();

    // this call does not look correct...
    magma_dsetvector(
        dim_, &val, 1, data_.get() + j * ld_ + i, 1, magma_singleton.queue_);
#else
    *(data_.get() + j * ld_ + i) = val;
#endif
}

void ReplicatedMatrix::setDiagonal(const std::vector<double>& diag_values)
{
#ifdef HAVE_MAGMA
    auto& magma_singleton = MagmaSingleton::get_magma_singleton();

    magma_dsetvector(dim_, diag_values.data(), 1, data_.get(), ld_ + 1,
        magma_singleton.queue_);
#else
    double* data                 = data_.get();
    for (int i = 0; i < dim_; i++)
        data[i * (ld_ + 1)] = diag_values[i];
#endif
}

double ReplicatedMatrix::trace() const
{
    const std::vector<double> val(dim_, 1.);
#ifdef HAVE_MAGMA
    auto& magma_singleton = MagmaSingleton::get_magma_singleton();

    // this is a little contorted, but it works for now...
    std::unique_ptr<double, void (*)(double*)> tmp_dev(
        Memory::allocate(dim_ * ld_), Memory::free);
    magma_dsetvector(
        dim_, val.data(), 1, tmp_dev.get(), 1, magma_singleton.queue_);

    return magma_ddot(
        dim_, data_.get(), ld_ + 1, tmp_dev.get(), 1, magma_singleton.queue_);
#else
    int ione = 1;
    int ldp  = ld_ + 1;
    return DDOT(&dim_, data_.get(), &ldp, val.data(), &ione);
#endif
}

double ReplicatedMatrix::traceProduct(const ReplicatedMatrix& matrix) const
{
    double trace = 0.;

#ifdef HAVE_MAGMA
    auto& magma_singleton = MagmaSingleton::get_magma_singleton();

    for (int i = 0; i < dim_; i++)
        trace += magma_ddot(dim_, data_.get() + i, ld_,
            matrix.data_.get() + matrix.ld_ * i, 1, magma_singleton.queue_);
#else
    int ione = 1;
    int ld   = ld_;
    for (int i = 0; i < dim_; i++)
        trace += DDOT(&dim_, data_.get() + i, &ld,
            matrix.data_.get() + matrix.ld_ * i, &ione);
#endif

    return trace;
}

double ReplicatedMatrix::norm(char ty)
{
#ifdef HAVE_MAGMA
    magma_norm_t magma_ty = magma_norm_const(ty);

    auto& magma_singleton = MagmaSingleton::get_magma_singleton();

    double* dwork;
    magma_dmalloc(&dwork, dim_);
    double norm_val = magmablas_dlange(magma_ty, dim_, dim_, data_.get(), ld_,
        dwork, lwork, magma_singleton.queue_);

    magma_singleton.sync();
    magma_free(dwork);

    return norm_val;
#else
    std::vector<double> dwork(dim_);
    int ld = ld_;
    return DLANGE(&ty, &dim_, &dim_, data_.get(), &ld, dwork.data());
#endif
}

void ReplicatedMatrix::trset(const char uplo)
{
#ifdef HAVE_MAGMA
    auto& magma_singleton = MagmaSingleton::get_magma_singleton();

    std::vector<double> mat(dim_ * dim_);

    magma_dgetmatrix(
        dim_, dim_, data_.get(), ld_, mat.data(), dim_, magma_singleton.queue_);
#else
    double* mat = data_.get();
#endif

    if (uplo == 'l' || uplo == 'L')
    {
        for (int j = 0; j < dim_; j++)
            for (int i = 0; i < j; i++)
                mat[i + j * dim_] = 0.;
    }
    else
    {
        for (int j = 0; j < dim_; j++)
            for (int i = j + 1; i < dim_; i++)
                mat[i + j * dim_] = 0.;
    }

#ifdef HAVE_MAGMA
    magma_dsetmatrix(
        dim_, dim_, mat.data(), dim_, data_.get(), ld_, magma_singleton.queue_);
#endif
}

void ReplicatedMatrix::clear()
{
#ifdef HAVE_MAGMA
    auto& magma_singleton = MagmaSingleton::get_magma_singleton();

    magmablas_dlaset(MagmaFull, dim_, dim_, 0.0, 0.0, data_.get(), ld_,
        magma_singleton.queue_);
#else
    memset(data_.get(), 0, dim_ * ld_ * sizeof(double));
#endif
}

void ReplicatedMatrix::print(std::ostream& os, const int ia, const int ja,
    const int ma, const int na) const
{
    const int m = std::min(ma, std::max(dim_ - ia, 0));
    const int n = std::min(na, std::max(dim_ - ja, 0));

#ifdef HAVE_MAGMA
    auto& magma_singleton = MagmaSingleton::get_magma_singleton();

    std::vector<double> mat(dim_ * dim_);

    magma_dgetmatrix(
        dim_, dim_, data_.get(), ld_, mat.data(), dim_, magma_singleton.queue_);
#else
    const double* const mat = data_.get();
#endif
    if (onpe0_)
        for (int i = ia; i < m; i++)
        {
            for (int j = ja; j < n; j++)
                os << mat[i + j * dim_] << "   ";
            os << std::endl;
        }
}

// add shift to diagonal, to shift eigenvalues
void ReplicatedMatrix::shift(const double shift)
{
    double* mat = data_.get();
    for (int i = 0; i < dim_; i++)
        mat[i + i * dim_] += shift;
}

void ReplicatedMatrix::printMM(std::ostream& os) const
{
    (void)os;
    std::cerr << "ReplicatedMatrix::printMM() not implemented" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}

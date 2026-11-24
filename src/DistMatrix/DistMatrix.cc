// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "MGmol_scalapack.h"

#include "BlacsContext.h"
#include "DistMatrix.h"
#include "DistVector.h"
#include "MGmol_MPI.h"
#include "random.h"

#include <cassert>
#include <iomanip>
#include <iostream>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#ifdef MGMOL_USE_SCALAPACK
#include "blacs.h"
#endif

#include "MGmol_abort.h"

namespace dist_matrix
{

#ifndef MGMOL_USE_SCALAPACK
int NUMROC(int* a, int* b, int* c, int* d, int* e) { return *a; }
int INDXL2G(Pint indxloc, Pint nb, Pint iproc, Pint isrcproc, Pint nprocs)
{
    return *indxloc;
}
#endif

template <class T>
DistMatrix<T>::DistMatrix(const std::string& name)
    : object_name_(name),
      bc_(*default_bc_),
      comm_global_(default_bc_->comm_global()),
      m_(0),
      n_(0),
      mb_(0),
      nb_(0)
{
    assert(default_bc_ != nullptr);
}

template <class T>
DistMatrix<T>::DistMatrix(const std::string& name, const BlacsContext& bc)
    : object_name_(name),
      bc_(bc),
      comm_global_(bc.comm_global()),
      m_(0),
      n_(0),
      mb_(0),
      nb_(0)
{
}

// Construct a DistMatrix of dimensions m,n
template <class T>
DistMatrix<T>::DistMatrix(
    const std::string& name, const BlacsContext& bc, const int m, const int n)
    : object_name_(name), bc_(bc), comm_global_(bc.comm_global())
{
    resize(m, n, distmatrix_def_block_size_, distmatrix_def_block_size_);
}

template <class T>
DistMatrix<T>::DistMatrix(const std::string& name, const int m, const int n)
    : object_name_(name),
      bc_(*default_bc_),
      comm_global_(default_bc_->comm_global())
{
    assert(default_bc_ != nullptr);

    resize(m, n, distmatrix_def_block_size_, distmatrix_def_block_size_);
}

template <class T>
DistMatrix<T>::DistMatrix(const std::string& name, const int m)
    : object_name_(name),
      bc_(*default_bc_),
      comm_global_(default_bc_->comm_global())
{
    assert(default_bc_ != nullptr);

    resize(m, m, distmatrix_def_block_size_, distmatrix_def_block_size_);
}

// Construct a DistMatrix of dimensions m,n
template <class T>
DistMatrix<T>::DistMatrix(const std::string& name, const BlacsContext& bc,
    const int m, const int n, const int mb, const int nb)
    : object_name_(name), bc_(bc), comm_global_(bc.comm_global())
{
    resize(m, n, mb, nb);
}

// Construct a DistMatrix of dimensions m,n
template <class T>
DistMatrix<T>::DistMatrix(const std::string& name, const int m, const int n,
    const int mb, const int nb)
    : object_name_(name),
      bc_(*default_bc_),
      comm_global_(default_bc_->comm_global())
{
    resize(m, n, mb, nb);
}

////////////////////////////////////////////////////////////////////////////////
template <class T>
DistMatrix<T>::DistMatrix(const DistMatrix<T>& a)
    : bc_(a.bc_), comm_global_(a.bc_.comm_global())
{
    object_name_ = a.object_name_ + "_duplicate";
    resize(a.m(), a.n(), a.mb(), a.nb());
    *this = a;
}

////////////////////////////////////////////////////////////////////////////////
template <class T>
void DistMatrix<T>::resize(const int m, const int n, const int mb, const int nb)
{
    assert(m > 0);
    assert(n > 0);
    assert(mb > 0);
    assert(nb > 0);

    m_ = m;
    n_ = n;
#ifdef MGMOL_USE_SCALAPACK
    mb_ = std::min(mb, m);
    nb_ = std::min(nb, n);
#else
    mb_ = m;
    nb_ = n;
#endif
    ictxt_       = bc_.ictxt();
    nprow_       = bc_.nprow();
    npcol_       = bc_.npcol();
    myrow_       = bc_.myrow();
    mycol_       = bc_.mycol();
    active_      = myrow_ >= 0;
    int isrcproc = 0;
    mloc_        = NUMROC(&m_, &mb_, &myrow_, &isrcproc, &nprow_);
    nloc_        = NUMROC(&n_, &nb_, &mycol_, &isrcproc, &npcol_);

    // set leading dimension of val array to mloc_;
    lld_ = mloc_;
    if (lld_ == 0) lld_ = 1;

    // total and local number of blocks
    mblocks_ = 0;
    nblocks_ = 0;
    if (active_)
    {
        mblocks_ = (mloc_ + mb_ - 1) / mb_;
        nblocks_ = (nloc_ + nb_ - 1) / nb_;
    }

    m_incomplete_ = mloc_ % mb_ != 0;
    n_incomplete_ = nloc_ % nb_ != 0;

    desc_[0] = 1;
    desc_[1] = ictxt_;
    desc_[2] = m_;
    desc_[3] = n_;
    desc_[4] = mb_;
    desc_[5] = nb_;
    desc_[6] = 0;
    desc_[7] = 0;
    desc_[8] = lld_;

    // allocate local space
    if (active_)
    {
        size_ = mloc_ * nloc_;
        val_.resize(size_);
    }
    else
    {
        size_ = 0;
    }
    clear();
}

////////////////////////////////////////////////////////////////////////////////
template <class T>
void DistMatrix<T>::clear(void)
{
    if (active_) memset(val_.data(), 0, size_ * sizeof(T));
}

////////////////////////////////////////////////////////////////////////////////
// Compute the global index of a distributed matrix entry pointed to by the
// local index of the process
template <class T>
int DistMatrix<T>::indxl2grow(const int indxloc) const
{
    const int isrcproc = 0;
    const int findx    = indxloc + 1;
    return INDXL2G(&findx, &mb_, &myrow_, &isrcproc, &nprow_) - 1;
}

template <class T>
int DistMatrix<T>::indxl2gcol(const int indxloc) const
{
    const int isrcproc = 0;
    const int findx    = indxloc + 1;
    return INDXL2G(&findx, &nb_, &mycol_, &isrcproc, &npcol_) - 1;
}

////////////////////////////////////////////////////////////////////////////////
// set diagonal of local blocks of DistMatrix to global diag_values
template <class T>
void DistMatrix<T>::setDiagonal(const std::vector<T>& diag_values)
{
    assert((int)diag_values.size() == m_);

    if (active_)
        for (int i = 0; i < mloc_; i++)
        {
            const int ig = indxl2grow(i);
            for (int j = 0; j < nloc_; j++)
            {
                const int jg = indxl2gcol(j);
                if (ig == jg) val_[i + j * mloc_] = diag_values[ig];
            }
        }
}

// add shift to diagonal, to shift eigenvalues
template <class T>
void DistMatrix<T>::shift(const T shift)
{
    if (active_)
        for (int i = 0; i < mloc_; i++)
        {
            const int ig = indxl2grow(i);
            for (int j = 0; j < nloc_; j++)
            {
                const int jg = indxl2gcol(j);
                if (ig == jg) val_[i + j * mloc_] += shift;
            }
        }
}

template <class T>
void DistMatrix<T>::setRandom(const T minv, const T maxv)
{
    std::vector<T> mat(m_ * n_);

    generateRandomData(mat, minv, maxv);

    this->init(mat.data(), m_);
}

////////////////////////////////////////////////////////////////////////////////

template <class T>
double DistMatrix<T>::traceProduct(const DistMatrix<T>& x) const
{
    assert(ictxt_ == x.ictxt());
    double sum  = 0.;
    double tsum = 0.;
    if (active_)
    {
        assert(m_ == x.m());
        assert(n_ == x.n());
        assert(mloc_ == x.mloc());
        assert(nloc_ == x.nloc());
        assert(x.size_ == size_);
        tsum = LinearAlgebraUtils<MemorySpace::Host>::MPdot(
            size_, &val_[0], &x.val_[0]);
    }
#ifdef MGMOL_USE_SCALAPACK
    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    mmpi.allreduce(&tsum, &sum, 1, MPI_SUM);
#else
    sum = tsum;
#endif
    return sum;
}

////////////////////////////////////////////////////////////////////////////////
template <class T>
void DistMatrix<T>::scal(const double alpha)
{
    if (alpha == 1.0) return;
    if (alpha == 0.0)
    {
        clear();
        return;
    }
    if (active_)
        LinearAlgebraUtils<MemorySpace::Host>::MPscal(size_, alpha, &val_[0]);
}

////////////////////////////////////////////////////////////////////////////////
template <class T>
void DistMatrix<T>::axpy(const double alpha, const DistMatrix<T>& x)
{
    assert(ictxt_ == x.ictxt());
    //  int ione=1;
    assert(m_ == x.m());
    assert(n_ == x.n());
    assert(mloc_ == x.mloc_);
    assert(nloc_ == x.nloc_);
    assert(size_ == x.size_);
    if (active_)
        LinearAlgebraUtils<MemorySpace::Host>::MPaxpy(
            size_, alpha, &x.val_[0], &val_[0]);
}

////////////////////////////////////////////////////////////////////////////////
// identity: initialize matrix to identity
////////////////////////////////////////////////////////////////////////////////
template <>
void DistMatrix<double>::identity(void)
{
#ifdef MGMOL_USE_SCALAPACK
    char uplo    = 'n';
    double alpha = 0.0;
    double beta  = 1.0;
    int ione     = 1;
    if (active_)
        pdlaset(&uplo, &m_, &n_, &alpha, &beta, &val_[0], &ione, &ione, desc_);
#else
    clear();
    for (int i = 0; i < std::min(m_, n_); i++)
        val_[i + mloc_ * i] = 1.0;
#endif
}

template <>
void DistMatrix<float>::identity(void)
{
#ifdef MGMOL_USE_SCALAPACK
    char uplo   = 'n';
    float alpha = 0.0;
    float beta  = 1.0;
    int ione    = 1;
    if (active_)
        pslaset(&uplo, &m_, &n_, &alpha, &beta, &val_[0], &ione, &ione, desc_);
#else
    clear();
    for (int i = 0; i < std::min(m_, n_); i++)
        val_[i + mloc_ * i] = 1.0;
#endif
}

////////////////////////////////////////////////////////////////////////////////
template <>
DistMatrix<double>& DistMatrix<double>::operator=(const DistMatrix<double>& src)
{
    if (this == &src) return *this;

    // cout << " DistMatrix<double>::operator= for "<<object_name_<<" and
    // "<<a.object_name_ << std::endl;
    if (src.m() != m_ || src.n() != n_)
        mgmol_abort(" DistMatrix<double>::operator=: size inconsistency");

#ifdef MGMOL_USE_SCALAPACK
    if (src.mb_ == mb_ && src.nb_ == nb_ && src.bc_ == bc_)
#endif
    {
        // simple copy
        if (active_)
        {
            for (int i = 0; i < 9; i++)
            {
                // cout << " desc_[" << i << "] = " << desc_[i] << " / "
                //     << " src.desc_[" << i << "] = " << src.desc_[i] <<
                //     std::endl;
                assert(desc_[i] == src.desc_[i]);
            }
            // cout << " DistMatrix<double>::operator=: simple copy" <<
            // std::endl;
            memcpy(&val_[0], &src.val_[0], size_ * sizeof(double));
        }
    }
#ifdef MGMOL_USE_SCALAPACK
    else
    {
        // redistribute using function pdgemr2d
        // (see file REDIST/SRC/pdgemr.c in ScaLAPACK)
        int ione = 1;
        // get a context englobing at least all processors included
        // in either *this context or src context
        BlacsContext bc(comm_global_);
        int gictxt = bc.ictxt();
        pdgemr2d_tm_.start();
        pdgemr2d(&m_, &n_, src.val_.data(), &ione, &ione, src.desc(), &val_[0],
            &ione, &ione, desc_, &gictxt);
        pdgemr2d_tm_.stop();
    }
#endif
    return *this;
}

template <>
DistMatrix<float>& DistMatrix<float>::operator=(const DistMatrix<float>& src)
{
    if (this == &src) return *this;

    // cout << " DistMatrix<float>::operator= for "<<object_name_<<" and
    // "<<a.object_name_ << std::endl;
    if (src.m() != m_ || src.n() != n_)
        mgmol_abort(" DistMatrix<float>::operator=: size inconsistency");

#ifdef MGMOL_USE_SCALAPACK
    if (src.mb_ == mb_ && src.nb_ == nb_ && src.bc_ == bc_)
#endif
    {
        // simple copy
        if (active_)
        {
            for (int i = 0; i < 9; i++)
            {
                // cout << " desc_[" << i << "] = " << desc_[i] << " / "
                //     << " src.desc_[" << i << "] = " << src.desc_[i] <<
                //     std::endl;
                assert(desc_[i] == src.desc_[i]);
            }
            // cout << " DistMatrix<double>::operator=: simple copy" <<
            // std::endl;
            memcpy(&val_[0], &src.val_[0], size_ * sizeof(float));
        }
    }
#ifdef MGMOL_USE_SCALAPACK
    else
    {
        // redistribute using function pdgemr2d
        // (see file REDIST/SRC/pdgemr.c in ScaLAPACK)
        int ione = 1;
        // get a context englobing at least all processors included
        // in either *this context or src context
        BlacsContext bc(comm_global_);
        int gictxt = bc.ictxt();
        pdgemr2d_tm_.start();
        psgemr2d(&m_, &n_, src.val_.data(), &ione, &ione, src.desc(), &val_[0],
            &ione, &ione, desc_, &gictxt);
        pdgemr2d_tm_.stop();
    }
#endif
    return *this;
}
////////////////////////////////////////////////////////////////////////////////
// Copy a distributed matrix src into another in a block at position (ib,jb)
template <>
DistMatrix<double>& DistMatrix<double>::assign(
    const DistMatrix<double>& src, const int ib, const int jb)
{
    if (this == &src) return *this;

    int m = src.m();
    int n = src.n();

    if (ib + m > m_ || jb + n > n_)
        mgmol_abort(" DistMatrix<double>::assign: size inconsistency");

#ifdef MGMOL_USE_SCALAPACK
    if (src.bc_ == bc_ && src.m_ == m_ && src.n_ == n_ && src.mb_ == mb_
        && src.nb_ == nb_)
#endif
    {
        // simple copy
        if (active_)
        {
            for (int j = 0; j < n; j++)
            {
                memcpy(&val_[(j + jb) * m_ + ib], &src.val_[j * m],
                    m * sizeof(double));
            }
        }
    }
#ifdef MGMOL_USE_SCALAPACK
    else
    {
        // redistribute
        // get global context index
        int pib  = ib + 1;
        int pjb  = jb + 1;
        int ione = 1;
        BlacsContext bc(comm_global_);
        int gictxt = bc.ictxt();
        assert(m + ib <= m_);
        assert(n + jb <= n_);
        pdgemr2d(&m, &n, src.val_.data(), &ione, &ione, src.desc(), val_.data(),
            &pib, &pjb, desc_, &gictxt);
    }
#endif
    return *this;
}

template <>
DistMatrix<float>& DistMatrix<float>::assign(
    const DistMatrix<float>& src, const int ib, const int jb)
{
    if (this == &src) return *this;

    int m = src.m();
    int n = src.n();

    if (ib + m > m_ || jb + n > n_)
        mgmol_abort(" DistMatrix<float>::assign: size inconsistency");

#ifdef MGMOL_USE_SCALAPACK
    if (src.bc_ == bc_ && src.m_ == m_ && src.n_ == n_ && src.mb_ == mb_
        && src.nb_ == nb_)
#endif
    {
        // simple copy
        if (active_)
        {
            for (int j = 0; j < n; j++)
            {
                memcpy(&val_[(j + jb) * m_ + ib], &src.val_[j * m],
                    m * sizeof(float));
            }
        }
    }
#ifdef MGMOL_USE_SCALAPACK
    else
    {
        // redistribute
        // get global context index
        int pib  = ib + 1;
        int pjb  = jb + 1;
        int ione = 1;
        BlacsContext bc(comm_global_);
        int gictxt = bc.ictxt();
        assert(m + ib <= m_);
        assert(n + jb <= n_);
        psgemr2d(&m, &n, src.val_.data(), &ione, &ione, src.desc(), val_.data(),
            &pib, &pjb, desc_, &gictxt);
    }
#endif
    return *this;
}

////////////////////////////////////////////////////////////////////////////////
// matrix-vector multiplication
// this = alpha*op(A)*b+beta*this
////////////////////////////////////////////////////////////////////////////////
template <>
void DistMatrix<double>::gemv(const char transa, const double alpha,
    const DistMatrix<double>& a, const DistMatrix<double>& b, const double beta)
{
    assert(ictxt_ == a.ictxt());
    assert(ictxt_ == b.ictxt());

    if (active_)
    {
        int m, n;
        if (transa == 'N' || transa == 'n')
        {
            m = a.m();
            n = a.n();
            assert(a.m() == m_);
        }
        else
        {
            m = a.n();
            n = a.m();
            assert(a.n() == m_);
        }

#ifdef MGMOL_USE_SCALAPACK
        int ione = 1;
        pdgemv(&transa, &m, &n, &alpha, &a.val_[0], &ione, &ione, a.desc(),
            &b.val_[0], &ione, &ione, b.desc(), &ione, &beta, &val_[0], &ione,
            &ione, desc_, &ione);
#else
        int ione            = 1;
        DGEMV(&transa, &m, &n, &alpha, &a.val_[0], &a.lld_, &b.val_[0], &ione,
            &beta, &val_[0], &ione);
#endif
    }
}

// y = *this * v
template <>
void DistMatrix<double>::matvec(
    dist_matrix::DistVector<double>& v, dist_matrix::DistVector<double>& y)
{
    if (active_)
    {
        char transa  = 'N';
        double alpha = 1.;
        double beta  = 0.;
#ifdef MGMOL_USE_SCALAPACK
        int ione = 1;
        pdgemv(&transa, &m_, &n_, &alpha, &val_[0], &ione, &ione, desc(),
            &v.val_[0], &ione, &ione, v.desc(), &ione, &beta, &y.val_[0], &ione,
            &ione, y.desc_, &ione);
#else
        int ione = 1;
        DGEMV(&transa, &m_, &n_, &alpha, &val_[0], &lld_, &v.val_[0], &ione,
            &beta, &y.val_[0], &ione);
#endif
    }
}

template <>
void DistMatrix<float>::gemv(const char transa, const float alpha,
    const DistMatrix<float>& a, const DistMatrix<float>& b, const float beta)
{
    assert(ictxt_ == a.ictxt());
    assert(ictxt_ == b.ictxt());

    if (active_)
    {
        int m, n;
        if (transa == 'N' || transa == 'n')
        {
            m = a.m();
            n = a.n();
            assert(a.m() == m_);
        }
        else
        {
            m = a.n();
            n = a.m();
            assert(a.n() == m_);
        }

#ifdef MGMOL_USE_SCALAPACK
        int ione = 1;
        psgemv(&transa, &m, &n, &alpha, &a.val_[0], &ione, &ione, a.desc(),
            &b.val_[0], &ione, &ione, b.desc(), &ione, &beta, &val_[0], &ione,
            &ione, desc_, &ione);
#else
        int ione = 1;
        sgemv(&transa, &m, &n, &alpha, &a.val_[0], &a.lld_, &b.val_[0], &ione,
            &beta, &val_[0], &ione);
#endif
    }
}

template <>
void DistMatrix<double>::symv(const char uplo, const double alpha,
    const DistMatrix<double>& a, const DistMatrix<double>& b, const double beta)
{
    assert(ictxt_ == a.ictxt());
    assert(ictxt_ == b.ictxt());

    if (active_)
    {
        assert(a.n() == m_);

#ifdef MGMOL_USE_SCALAPACK
        int ione = 1;
        pdsymv(&uplo, &m_, &alpha, &a.val_[0], &ione, &ione, a.desc(),
            &b.val_[0], &ione, &ione, b.desc(), &ione, &beta, &val_[0], &ione,
            &ione, desc_, &ione);
#else
        int ione = 1;
        DSYMV(&uplo, &m_, &alpha, &a.val_[0], &a.lld_, &b.val_[0], &ione, &beta,
            &val_[0], &ione);
#endif
    }
}

template <>
void DistMatrix<float>::symv(const char uplo, const float alpha,
    const DistMatrix<float>& a, const DistMatrix<float>& b, const float beta)
{
    assert(ictxt_ == a.ictxt());
    assert(ictxt_ == b.ictxt());

    if (active_)
    {
        assert(a.n() == m_);

#ifdef MGMOL_USE_SCALAPACK
        int ione = 1;
        pssymv(&uplo, &m_, &alpha, &a.val_[0], &ione, &ione, a.desc(),
            &b.val_[0], &ione, &ione, b.desc(), &ione, &beta, &val_[0], &ione,
            &ione, desc_, &ione);
#else
        int ione = 1;
        ssymv(&uplo, &m_, &alpha, &a.val_[0], &a.lld_, &b.val_[0], &ione, &beta,
            &val_[0], &ione);
#endif
    }
}

////////////////////////////////////////////////////////////////////////////////
// matrix multiplication
// this = alpha*op(A)*op(B)+beta*this
////////////////////////////////////////////////////////////////////////////////
template <>
void DistMatrix<double>::gemm(const char transa, const char transb,
    const double alpha, const DistMatrix<double>& a,
    const DistMatrix<double>& b, const double beta)
{
    assert(ictxt_ == a.ictxt());
    assert(ictxt_ == b.ictxt());

    if (active_)
    {
        int m, n, k;
        if (transa == 'N' || transa == 'n')
        {
            m = a.m();
            k = a.n();
            assert(a.m() == m_);
        }
        else
        {
            m = a.n();
            k = a.m();
            assert(a.n() == m_);
        }
        if (transb == 'N' || transb == 'n')
        {
            n = b.n();
            assert(k == b.m());
        }
        else
        {
            n = b.m();
            assert(k == b.n());
        }

#ifdef MGMOL_USE_SCALAPACK
        int ione = 1;
        pdgemm(&transa, &transb, &m, &n, &k, &alpha, &a.val_[0], &ione, &ione,
            a.desc(), &b.val_[0], &ione, &ione, b.desc(), &beta, &val_[0],
            &ione, &ione, desc_);
#else
        dgemm(&transa, &transb, &m, &n, &k, &alpha, &a.val_[0], &a.lld_,
            &b.val_[0], &b.lld_, &beta, &val_[0], &lld_);
#endif
    }
}

template <>
void DistMatrix<float>::gemm(const char transa, const char transb,
    const float alpha, const DistMatrix<float>& a, const DistMatrix<float>& b,
    const float beta)
{
    assert(ictxt_ == a.ictxt());
    assert(ictxt_ == b.ictxt());

    if (active_)
    {
        int m, n, k;
        if (transa == 'N' || transa == 'n')
        {
            m = a.m();
            k = a.n();
            assert(a.m() == m_);
        }
        else
        {
            m = a.n();
            k = a.m();
            assert(a.n() == m_);
        }
        if (transb == 'N' || transb == 'n')
        {
            n = b.n();
            assert(k == b.m());
        }
        else
        {
            n = b.m();
            assert(k == b.n());
        }

#ifdef MGMOL_USE_SCALAPACK
        int ione = 1;
        psgemm(&transa, &transb, &m, &n, &k, &alpha, &a.val_[0], &ione, &ione,
            a.desc(), &b.val_[0], &ione, &ione, b.desc(), &beta, &val_[0],
            &ione, &ione, desc_);
#else
        sgemm(&transa, &transb, &m, &n, &k, &alpha, &a.val_[0], &a.lld_,
            &b.val_[0], &b.lld_, &beta, &val_[0], &lld_);
#endif
    }
}
////////////////////////////////////////////////////////////////////////////////
// symmetrix_matrix * matrix multiplication
// this = beta * this + alpha * a * b
// this = beta * this + alpha * b * a
// with "a" symetric
////////////////////////////////////////////////////////////////////////////////
template <>
void DistMatrix<double>::symm(const char side, const char uplo,
    const double alpha, const DistMatrix<double>& a,
    const DistMatrix<double>& b, const double beta)
{
    assert(ictxt_ == a.ictxt());
    assert(ictxt_ == b.ictxt());

    if (active_)
    {
        assert(a.n() == a.m());
        if (side == 'L' || side == 'l')
        {
            assert(m_ == a.m());
            assert(n_ == b.n());
            assert(a.n() == b.m());
        }
        else
        {
            assert(n_ == a.m());
            assert(m_ == b.n());
            assert(a.m() == b.n());
        }

#ifdef MGMOL_USE_SCALAPACK
        int ione = 1;
        pdsymm(&side, &uplo, &m_, &n_, &alpha, &a.val_[0], &ione, &ione,
            a.desc(), &b.val_[0], &ione, &ione, b.desc(), &beta, &val_[0],
            &ione, &ione, desc_);
#else
        dsymm(&side, &uplo, &m_, &n_, &alpha, &a.val_[0], &a.lld_, &b.val_[0],
            &b.lld_, &beta, &val_[0], &lld_);
#endif
    }
}

template <>
void DistMatrix<float>::symm(const char side, const char uplo,
    const float alpha, const DistMatrix<float>& a, const DistMatrix<float>& b,
    const float beta)
{
    assert(ictxt_ == a.ictxt());
    assert(ictxt_ == b.ictxt());

    if (active_)
    {
        assert(a.n() == a.m());
        if (side == 'L' || side == 'l')
        {
            assert(m_ == a.m());
            assert(n_ == b.n());
            assert(a.n() == b.m());
        }
        else
        {
            assert(n_ == a.m());
            assert(m_ == b.n());
            assert(a.m() == b.n());
        }

#ifdef MGMOL_USE_SCALAPACK
        int ione = 1;
        pssymm(&side, &uplo, &m_, &n_, &alpha, &a.val_[0], &ione, &ione,
            a.desc(), &b.val_[0], &ione, &ione, b.desc(), &beta, &val_[0],
            &ione, &ione, desc_);
#else
        ssymm(&side, &uplo, &m_, &n_, &alpha, &a.val_[0], &a.lld_, &b.val_[0],
            &b.lld_, &beta, &val_[0], &lld_);
#endif
    }
}

////////////////////////////////////////////////////////////////////////////////
// Compute a matrix-matrix product for a real triangular
// matrix or its transpose.
// *this = alpha op(A) * *this    if side=='l'
// *this = alpha * *this * op(A)  if side=='r'
// where op(A) = A or trans(A)
// alpha is a scalar, *this is an m by n matrix, and A is a unit or non-unit,
// upper- or lower-triangular matrix.
////////////////////////////////////////////////////////////////////////////////
template <>
void DistMatrix<double>::trmm(const char side, const char uplo,
    const char trans, const char diag, const double alpha,
    const DistMatrix<double>& a)
{
    if (active_)
    {
        assert(a.m_ == a.n_);
        if (side == 'L' || side == 'l')
        {
            assert(a.n_ == m_);
        }
        else
        {
            assert(a.n_ == n_);
        }
#ifdef MGMOL_USE_SCALAPACK
        int ione = 1;
        pdtrmm(&side, &uplo, &trans, &diag, &m_, &n_, &alpha, &a.val_[0], &ione,
            &ione, a.desc_, &val_[0], &ione, &ione, desc_);
#else
        dtrmm(&side, &uplo, &trans, &diag, &m_, &n_, &alpha, &a.val_[0], &a.m_,
            &val_[0], &m_);
#endif
    }
}

template <>
void DistMatrix<float>::trmm(const char side, const char uplo, const char trans,
    const char diag, const float alpha, const DistMatrix<float>& a)
{
    if (active_)
    {
        assert(a.m_ == a.n_);
        if (side == 'L' || side == 'l')
        {
            assert(a.n_ == m_);
        }
        else
        {
            assert(a.n_ == n_);
        }
#ifdef MGMOL_USE_SCALAPACK
        int ione = 1;
        pstrmm(&side, &uplo, &trans, &diag, &m_, &n_, &alpha, &a.val_[0], &ione,
            &ione, a.desc_, &val_[0], &ione, &ione, desc_);
#else
        strmm(&side, &uplo, &trans, &diag, &m_, &n_, &alpha, &a.val_[0], &a.m_,
            &val_[0], &m_);
#endif
    }
}
////////////////////////////////////////////////////////////////////////////////
// solve one of the matrix equations
// op( A )*X = alpha*(*this), or X*op( A ) = alpha*(*this)
// where alpha is a scalar, X and (*this) are m by n matrices, A is a unit,
// or non-unit,  upper or lower triangular matrix  and  op( A )  is one  of
// op( A ) = A   or   op( A ) = A'.
//
// The matrix X is overwritten on (*this).
//
// op( A )*X = alpha*(*this)     if side=='l'
// X*op( A ) = alpha*(*this)     if side=='r'
// where op(A) = A or trans(A)
////////////////////////////////////////////////////////////////////////////////
template <>
void DistMatrix<double>::trsm(const char side, const char uplo,
    const char trans, const char diag, const double alpha,
    const DistMatrix<double>& a)
{
    if (active_)
    {
        assert(a.m_ == a.n_);
        if (side == 'L' || side == 'l')
        {
            assert(a.n_ == m_);
        }
        else
        {
            assert(a.n_ == n_);
        }
#ifdef MGMOL_USE_SCALAPACK
        int ione = 1;
        pdtrsm(&side, &uplo, &trans, &diag, &m_, &n_, &alpha, &a.val_[0], &ione,
            &ione, a.desc_, &val_[0], &ione, &ione, desc_);
#else
        dtrsm(&side, &uplo, &trans, &diag, &m_, &n_, &alpha, &a.val_[0], &a.m_,
            &val_[0], &m_);
#endif
    }
}

template <>
void DistMatrix<float>::trsm(const char side, const char uplo, const char trans,
    const char diag, const float alpha, const DistMatrix<float>& a)
{
    if (active_)
    {
        assert(a.m_ == a.n_);
        if (side == 'L' || side == 'l')
        {
            assert(a.n_ == m_);
        }
        else
        {
            assert(a.n_ == n_);
        }
#ifdef MGMOL_USE_SCALAPACK
        int ione = 1;
        pstrsm(&side, &uplo, &trans, &diag, &m_, &n_, &alpha, &a.val_[0], &ione,
            &ione, a.desc_, &val_[0], &ione, &ione, desc_);
#else
        strsm(&side, &uplo, &trans, &diag, &m_, &n_, &alpha, &a.val_[0], &a.m_,
            &val_[0], &m_);
#endif
    }
}
////////////////////////////////////////////////////////////////////////////////
// Solves a triangular system of the form A * X = B or
// A**T * X = B, where A is a triangular matrix of  order  N,
// and  B  is an N-by-NRHS matrix.
// Output in B.
////////////////////////////////////////////////////////////////////////////////
template <>
void DistMatrix<double>::trtrs(const char uplo, const char trans,
    const char diag, DistMatrix<double>& b) const
{
    int info;
    if (active_)
    {
        assert(m_ == n_);

#ifdef MGMOL_USE_SCALAPACK
        int ione = 1;
        pdtrtrs(&uplo, &trans, &diag, &m_, &b.n_, &val_[0], &ione, &ione, desc_,
            &b.val_[0], &ione, &ione, b.desc_, &info);
#else
        dtrtrs(&uplo, &trans, &diag, &m_, &b.n_, &val_[0], &m_, &b.val_[0],
            &b.m_, &info);
#endif
        if (info != 0)
        {
            MGMOL_DISTMATRIX_ERROR(
                "trtrs, object " << object_name_ << ", info=" << info);
        }
    }
}

template <>
void DistMatrix<float>::trtrs(const char uplo, const char trans,
    const char diag, DistMatrix<float>& b) const
{
    int info;
    if (active_)
    {
        assert(m_ == n_);

#ifdef MGMOL_USE_SCALAPACK
        int ione = 1;
        pstrtrs(&uplo, &trans, &diag, &m_, &b.n_, &val_[0], &ione, &ione, desc_,
            &b.val_[0], &ione, &ione, b.desc_, &info);
#else
        strtrs(&uplo, &trans, &diag, &m_, &b.n_, &val_[0], &m_, &b.val_[0],
            &b.m_, &info);
#endif
        if (info != 0)
        {
            MGMOL_DISTMATRIX_ERROR(
                "trtrs, object " << object_name_ << ", info=" << info);
        }
    }
}
////////////////////////////////////////////////////////////////////////////////
// Computes the Cholesky factorization of a square real
// symmetric positive definite distributed matrix
////////////////////////////////////////////////////////////////////////////////
template <>
int DistMatrix<double>::potrf(char uplo)
{
    potrf_tm_.start();

    int info = 0;
    if (active_)
    {
        assert(m_ == n_);

#ifdef MGMOL_USE_SCALAPACK
        int ione = 1;
        pdpotrf(&uplo, &m_, val_.data(), &ione, &ione, desc_, &info);
#else
        dpotrf(&uplo, &m_, val_.data(), &m_, &info);
#endif
        if (info != 0)
        {
            MGMOL_DISTMATRIX_ERROR(
                "potrf, object " << object_name_ << ", info=" << info);
        }
    }

    potrf_tm_.stop();

    return info;
}

template <>
int DistMatrix<float>::potrf(char uplo)
{
    potrf_tm_.start();

    int info = 0;
    if (active_)
    {
        assert(m_ == n_);

#ifdef MGMOL_USE_SCALAPACK
        int ione = 1;
        pspotrf(&uplo, &m_, &val_[0], &ione, &ione, desc_, &info);
#else
        spotrf(&uplo, &m_, &val_[0], &m_, &info);
#endif
        if (info != 0)
        {
            MGMOL_DISTMATRIX_ERROR(
                "potrf, object " << object_name_ << ", info=" << info);
        }
    }

    potrf_tm_.stop();

    return info;
}
////////////////////////////////////////////////////////////////////////////////
// Computes an LU factorization of a general m by n real
// distributed matrix using partial pivoting with row interchanges
////////////////////////////////////////////////////////////////////////////////
template <>
void DistMatrix<double>::getrf(std::vector<int>& ipiv)
{
    int info;
    if (active_)
    {
#ifdef MGMOL_USE_SCALAPACK
        int ione = 1;
        ipiv.resize(mb_ + mloc_);
        pdgetrf(&m_, &n_, &val_[0], &ione, &ione, desc_, &ipiv[0], &info);
#else
        ipiv.resize(n_);
        dgetrf(&m_, &n_, &val_[0], &m_, &ipiv[0], &info);
#endif
        if (info != 0)
        {
            MGMOL_DISTMATRIX_ERROR(
                "getrf, object " << object_name_ << ", info=" << info);
        }
    }
}

template <>
void DistMatrix<float>::getrf(std::vector<int>& ipiv)
{
    int info;
    if (active_)
    {
#ifdef MGMOL_USE_SCALAPACK
        int ione = 1;
        ipiv.resize(mb_ + mloc_);
        psgetrf(&m_, &n_, &val_[0], &ione, &ione, desc_, &ipiv[0], &info);
#else
        ipiv.resize(n_);
        sgetrf(&m_, &n_, &val_[0], &m_, &ipiv[0], &info);
#endif
        if (info != 0)
        {
            MGMOL_DISTMATRIX_ERROR(
                "getrf, object " << object_name_ << ", info=" << info);
        }
    }
}
////////////////////////////////////////////////////////////////////////////////
// Solve a system of linear equations A*X = B with a symmetric posi-
//  tive definite matrix A using the Cholesky factorization
// A = U**T*U or A = L*L**T computed by potrf
////////////////////////////////////////////////////////////////////////////////
template <>
void DistMatrix<double>::potrs(char uplo, DistMatrix<double>& b)
{
    int info;
    if (active_)
    {
        assert(m_ == n_);
        assert(b.m_ == m_);
        assert(b.n_ <= n_);
#ifdef MGMOL_USE_SCALAPACK
        int ione = 1;
        pdpotrs(&uplo, &m_, &b.n_, &val_[0], &ione, &ione, desc_, &b.val_[0],
            &ione, &ione, b.desc_, &info);
#else
        dpotrs(&uplo, &m_, &b.n_, &val_[0], &m_, &b.val_[0], &b.m_, &info);
#endif
        if (info != 0)
        {
            MGMOL_DISTMATRIX_ERROR(
                "potrs, object " << object_name_ << ", info = " << info);
        }
    }
}

template <>
void DistMatrix<float>::potrs(char uplo, DistMatrix<float>& b)
{
    int info;
    if (active_)
    {
        assert(m_ == n_);
        assert(b.m_ == m_);
        assert(b.n_ <= n_);
#ifdef MGMOL_USE_SCALAPACK
        int ione = 1;
        pspotrs(&uplo, &m_, &b.n_, &val_[0], &ione, &ione, desc_, &b.val_[0],
            &ione, &ione, b.desc_, &info);
#else
        spotrs(&uplo, &m_, &b.n_, &val_[0], &m_, &b.val_[0], &b.m_, &info);
#endif
        if (info != 0)
        {
            MGMOL_DISTMATRIX_ERROR(
                "potrs, object " << object_name_ << ", info = " << info);
        }
    }
}
////////////////////////////////////////////////////////////////////////////////
// Solve a system of linear equations A * X = B or A' * X = B with a general
// N-by-N matrix A using the LU factorization computed by getrf()
////////////////////////////////////////////////////////////////////////////////
template <>
void DistMatrix<double>::getrs(
    char trans, DistMatrix<double>& b, std::vector<int>& ipiv)
{
    int info;
    if (active_)
    {
        assert(m_ == n_);
        assert(b.m_ == m_);
        assert(b.n_ <= n_);
#ifdef MGMOL_USE_SCALAPACK
        int ione = 1;
        assert((int)ipiv.size() == (mb_ + mloc_));
        pdgetrs(&trans, &m_, &b.n_, &val_[0], &ione, &ione, desc_, &ipiv[0],
            &b.val_[0], &ione, &ione, b.desc_, &info);
#else
        assert((int)ipiv.size() == m_);
        dgetrs(&trans, &m_, &b.n_, &val_[0], &m_, &ipiv[0], &b.val_[0], &b.m_,
            &info);
#endif
        if (info != 0)
        {
            MGMOL_DISTMATRIX_ERROR(
                "getrs, object " << object_name_ << ", info = " << info);
        }
    }
}

template <>
void DistMatrix<float>::getrs(
    char trans, DistMatrix<float>& b, std::vector<int>& ipiv)
{
    int info;
    if (active_)
    {
        assert(m_ == n_);
        assert(b.m_ == m_);
        assert(b.n_ <= n_);
#ifdef MGMOL_USE_SCALAPACK
        int ione = 1;
        assert((int)ipiv.size() == (mb_ + mloc_));
        psgetrs(&trans, &m_, &b.n_, &val_[0], &ione, &ione, desc_, &ipiv[0],
            &b.val_[0], &ione, &ione, b.desc_, &info);
#else
        assert((int)ipiv.size() == m_);
        sgetrs(&trans, &m_, &b.n_, &val_[0], &m_, &ipiv[0], &b.val_[0], &b.m_,
            &info);
#endif
        if (info != 0)
        {
            MGMOL_DISTMATRIX_ERROR(
                "getrs, object " << object_name_ << ", info = " << info);
        }
    }
}
////////////////////////////////////////////////////////////////////////////////
// Compute  the inverse of a real symmetric positive definite matrix
// using the Cholesky factorization A = U**T*U or A = L*L**T computed
// by DistMatrix<double>::potrf
////////////////////////////////////////////////////////////////////////////////
template <>
int DistMatrix<double>::potri(char uplo)
{
    potri_tm_.start();

    int info = 0;
    if (active_)
    {
        assert(m_ == n_);

#ifdef MGMOL_USE_SCALAPACK
        int ione = 1;
        pdpotri(&uplo, &m_, &val_[0], &ione, &ione, desc_, &info);
#else
        dpotri(&uplo, &m_, &val_[0], &m_, &info);
#endif
        if (info != 0)
        {
            MGMOL_DISTMATRIX_ERROR(
                "potri, object " << object_name_ << ", info=" << info);
        }
    }
    potri_tm_.stop();

    return info;
}

template <>
int DistMatrix<float>::potri(char uplo)
{
    potri_tm_.start();

    int info = 0;
    if (active_)
    {
        assert(m_ == n_);

#ifdef MGMOL_USE_SCALAPACK
        int ione = 1;
        pspotri(&uplo, &m_, &val_[0], &ione, &ione, desc_, &info);
#else
        spotri(&uplo, &m_, &val_[0], &m_, &info);
#endif
        if (info != 0)
        {
            MGMOL_DISTMATRIX_ERROR(
                "potri, object " << object_name_ << ", info=" << info);
        }
    }
    potri_tm_.stop();

    return info;
}

// DTRTRI - computes the inverse of a real upper or lower triangular matrix A
template <>
int DistMatrix<double>::trtri(char uplo, char diag)
{
    trtri_tm_.start();

    int info = 0;
    if (active_)
    {
        assert(m_ == n_);

#ifdef MGMOL_USE_SCALAPACK
        int ione = 1;
        pdtrtri(&uplo, &diag, &m_, &val_[0], &ione, &ione, desc_, &info);
#else
        dtrtri(&uplo, &diag, &m_, &val_[0], &m_, &info);
#endif
        if (info != 0)
        {
            MGMOL_DISTMATRIX_ERROR(
                "trtri, object " << object_name_ << ", info=" << info);
        }
    }
    trtri_tm_.stop();

    return info;
}

////////////////////////////////////////////////////////////////////////////////
// returns the value of the 1-norm, or the Frobenius norm,
// or the infinity norm, or the element of largest absolute value of a
// distributed matrix
//
// std::max(abs(A(i,j))) if ty== 'm' or 'M'
// 1-norm           if ty == '1'
// infinity-norm    if ty=='I' or 'i'
// Frobenius norm   if ty=='F' or 'f'
template <>
double DistMatrix<double>::norm(char ty)
{
    int lwork       = 0;
    double norm_val = 1.;
    if (active_)
    {
#ifdef MGMOL_USE_SCALAPACK
        int ione = 1;
        if (ty == 'I' || ty == 'i') lwork = mloc_;
        if (ty == '1') lwork = nloc_;

        std::vector<double> work(lwork);

        norm_val
            = pdlange(&ty, &m_, &n_, &val_[0], &ione, &ione, desc_, &work[0]);
#else
        if (ty == 'I' || ty == 'i') lwork = m_;
        std::vector<double> work(lwork);
        norm_val = dlange(&ty, &m_, &n_, &val_[0], &m_, &work[0]);
#endif
    }
    MPI_Bcast(&norm_val, 1, MPI_DOUBLE, 0, comm_global_);
    return norm_val;
}

template <>
double DistMatrix<float>::norm(char ty)
{
    int lwork      = 0;
    float norm_val = 1.;
    if (active_)
    {
#ifdef MGMOL_USE_SCALAPACK
        int ione = 1;
        if (ty == 'I' || ty == 'i') lwork = mloc_;
        if (ty == '1') lwork = nloc_;

        std::vector<float> work(lwork);

        norm_val = (double)pslange(
            &ty, &m_, &n_, &val_[0], &ione, &ione, desc_, &work[0]);
#else
        if (ty == 'I' || ty == 'i') lwork = m_;
        std::vector<float> work(lwork);
        norm_val = (double)slange(&ty, &m_, &n_, &val_[0], &m_, &work[0]);
#endif
    }
    MPI_Bcast(&norm_val, 1, MPI_FLOAT, 0, comm_global_);
    return norm_val;
}

////////////////////////////////////////////////////////////////////////////////
// estimate the reciprocal of the condition number (in the 1-norm) of a
// real symmetric positive definite matrix given by its Cholesky factorization
// U or L such that A  = U**T*U or A = L*L**T
// (as computed by DistMatrix<double>::potrf)
// anorm: 1-norm of matrix A
////////////////////////////////////////////////////////////////////////////////
template <>
double DistMatrix<double>::pocon(char uplo, double anorm)
{
    int info;
    double rcond = 1.;
    if (active_)
    {
        assert(m_ == n_);

#ifdef MGMOL_USE_SCALAPACK
        int ione   = 1;
        int lwork  = 2 * mloc_ + 3 * nloc_ + nb_ * std::min(npcol_, nprow_);
        lwork      = std::max(lwork, 1);
        int liwork = std::max(mloc_, 1);
        std::vector<double> work_vec(lwork);
        std::vector<int> iwork_vec(liwork);
        double* work = work_vec.data();
        int* iwork   = iwork_vec.data();
        pdpocon(&uplo, &m_, &val_[0], &ione, &ione, desc_, &anorm, &rcond, work,
            &lwork, iwork, &liwork, &info);
        if (info != 0)
        {
            std::cerr << "mloc_=" << mloc_ << ", nloc_=" << nloc_ << std::endl;
            std::cerr << "npcol_=" << npcol_ << ", nprow_=" << nprow_
                      << std::endl;
            std::cerr << "lwork=" << lwork << ", but should be at least "
                      << work[0] << std::endl;
            std::cerr << "liwork=" << liwork << ", but should be at least "
                      << iwork[0] << std::endl;
        }
#else
        std::vector<double> work(3 * m_);
        std::vector<int> iwork(m_);
        dpocon(&uplo, &m_, &val_[0], &m_, &anorm, &rcond, work.data(),
            iwork.data(), &info);
#endif
        if (info != 0)
        {
            MGMOL_DISTMATRIX_ERROR(
                "pocon, object " << object_name_ << ", info=" << info);
        }
    }
    return rcond;
}

template <>
double DistMatrix<float>::pocon(char uplo, float anorm)
{
    int info;
    float rcond = 1.;
    if (active_)
    {
        assert(m_ == n_);

#ifdef MGMOL_USE_SCALAPACK
        int ione   = 1;
        int lwork  = 2 * mloc_ + 3 * nloc_ + nb_ * std::min(npcol_, nprow_);
        lwork      = std::max(lwork, 1);
        int liwork = std::max(mloc_, 1);
        std::vector<float> work_vec(lwork);
        std::vector<int> iwork_vec(liwork);
        float* work = work_vec.data();
        int* iwork  = iwork_vec.data();
        pspocon(&uplo, &m_, &val_[0], &ione, &ione, desc_, &anorm, &rcond, work,
            &lwork, iwork, &liwork, &info);
        if (info != 0)
        {
            std::cerr << "mloc_=" << mloc_ << ", nloc_=" << nloc_ << std::endl;
            std::cerr << "npcol_=" << npcol_ << ", nprow_=" << nprow_
                      << std::endl;
            std::cerr << "lwork=" << lwork << ", but should be at least "
                      << work[0] << std::endl;
            std::cerr << "liwork=" << liwork << ", but should be at least "
                      << iwork[0] << std::endl;
        }
#else
        std::vector<float> work(3 * m_);
        std::vector<int> iwork(m_);
        spocon(&uplo, &m_, &val_[0], &m_, &anorm, &rcond, work.data(),
            iwork.data(), &info);
#endif
        if (info != 0)
        {
            MGMOL_DISTMATRIX_ERROR(
                "pocon, object " << object_name_ << ", info=" << info);
        }
    }
    return (double)rcond;
}
////////////////////////////////////////////////////////////////////////////////
// symmetric rank k update
// this = beta * this + alpha * A * A^T  (trans=='n')
// this = beta * this + alpha * A^T * A  (trans=='t')
////////////////////////////////////////////////////////////////////////////////
template <>
void DistMatrix<double>::syrk(
    char uplo, char trans, double alpha, DistMatrix<double>& a, double beta)
{
    assert(ictxt_ == a.ictxt());
    assert(n_ == m_); // *this must be a square matrix

    if (active_)
    {
        int n, k;
        if (trans == 'N' || trans == 'n')
        {
            n = m_;
            k = a.n();
        }
        else
        {
            n = m_;
            k = a.m();
        }

#ifdef MGMOL_USE_SCALAPACK
        int ione = 1;
        pdsyrk(&uplo, &trans, &n, &k, &alpha, &a.val_[0], &ione, &ione,
            a.desc(), &beta, &val_[0], &ione, &ione, desc_);
#else
        dsyrk(&uplo, &trans, &n, &k, &alpha, &a.val_[0], &a.m_, &beta, &val_[0],
            &m_);
#endif
    }
}

template <>
void DistMatrix<float>::syrk(
    char uplo, char trans, float alpha, DistMatrix<float>& a, float beta)
{
    assert(ictxt_ == a.ictxt());
    assert(n_ == m_); // *this must be a square matrix

    if (active_)
    {
        int n, k;
        if (trans == 'N' || trans == 'n')
        {
            n = m_;
            k = a.n();
        }
        else
        {
            n = m_;
            k = a.m();
        }

#ifdef MGMOL_USE_SCALAPACK
        int ione = 1;
        pssyrk(&uplo, &trans, &n, &k, &alpha, &a.val_[0], &ione, &ione,
            a.desc(), &beta, &val_[0], &ione, &ione, desc_);
#else
        ssyrk(&uplo, &trans, &n, &k, &alpha, &a.val_[0], &a.m_, &beta, &val_[0],
            &m_);
#endif
    }
}
////////////////////////////////////////////////////////////////////////////////
// getsub: *this = sub(A)
// copy submatrix A(ia:ia+m, ja:ja+n) into *this;
// *this and A may live in different contexts,
// Assumes that the context of *this is a subcontext of the context of A
template <>
void DistMatrix<double>::getsub(
    const DistMatrix<double>& a, int m, int n, int ia, int ja)
{
#ifdef MGMOL_USE_SCALAPACK
    int iap = ia + 1;
    int jap = ja + 1;
    assert(n <= n_);
    assert(n <= a.n());
    assert(m <= m_);
    assert(m <= a.m());
    int ione = 1;
    BlacsContext bc(comm_global_);
    int gictxt = bc.ictxt();
    // int gictxt = ictxt_;
    pdgemr2d(&m, &n, a.val_.data(), &iap, &jap, a.desc_, val_.data(), &ione,
        &ione, desc_, &gictxt);
#else
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            val_[i + j * m_] = a.val_[(i + ia) + (j + ja) * a.m()];
#endif
}

template <>
void DistMatrix<float>::getsub(
    const DistMatrix<float>& a, int m, int n, int ia, int ja)
{
#ifdef MGMOL_USE_SCALAPACK
    int iap = ia + 1;
    int jap = ja + 1;
    assert(n <= n_);
    assert(n <= a.n());
    assert(m <= m_);
    assert(m <= a.m());
    int ione = 1;
    BlacsContext bc(comm_global_);
    int gictxt = bc.ictxt();
    // int gictxt = ictxt_;
    psgemr2d(&m, &n, a.val_.data(), &iap, &jap, a.desc_, val_.data(), &ione,
        &ione, desc_, &gictxt);
#else
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            val_[i + j * m_] = a.val_[(i + ia) + (j + ja) * a.m()];
#endif
}
////////////////////////////////////////////////////////////////////////////////
template <>
void DistMatrix<int>::getsub(
    const DistMatrix<int>& a, int m, int n, int ia, int ja)
{
#ifdef MGMOL_USE_SCALAPACK
    int iap = ia + 1;
    int jap = ja + 1;
    assert(n <= n_);
    assert(n <= a.n());
    assert(m <= m_);
    assert(m <= a.m());
    int ione = 1;
    BlacsContext bc(comm_global_);
    int gictxt = bc.ictxt();
    pigemr2d(&m, &n, a.val_.data(), &iap, &jap, a.desc(), val_.data(), &ione,
        &ione, desc_, &gictxt);
#else
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            val_[i + j * m_] = a.val_[(i + ia) + (j + ja) * a.m()];
#endif
}

////////////////////////////////////////////////////////////////////////////////
// matrix transpose
// this = alpha * transpose(A) + beta * this
////////////////////////////////////////////////////////////////////////////////
template <>
void DistMatrix<double>::transpose(
    const double alpha, const DistMatrix<double>& a, const double beta)
{
    assert(ictxt_ == a.ictxt());

    if (active_)
    {
        assert(a.m() == n_);
        assert(a.n() == m_);

#ifdef MGMOL_USE_SCALAPACK
        int ione = 1;
        pdtran(&m_, &n_, &alpha, &a.val_[0], &ione, &ione, a.desc(), &beta,
            &val_[0], &ione, &ione, desc_);
#else
        scal(beta);
        for (int i = 0; i < m_; i++)
            for (int j = 0; j < i; j++)
            {
                val_[i * m_ + j] += alpha * a.val_[j * m_ + i];
                val_[j * m_ + i] += alpha * a.val_[i * m_ + j];
            }
        for (int i = 0; i < m_; i++)
            val_[i * m_ + i] += alpha * a.val_[i * m_ + i];
#endif
    }
}

template <>
void DistMatrix<float>::transpose(
    const float alpha, const DistMatrix<float>& a, const float beta)
{
    assert(ictxt_ == a.ictxt());

    if (active_)
    {
        assert(a.m() == n_);
        assert(a.n() == m_);

#ifdef MGMOL_USE_SCALAPACK
        int ione = 1;
        pstran(&m_, &n_, &alpha, &a.val_[0], &ione, &ione, a.desc(), &beta,
            &val_[0], &ione, &ione, desc_);
#else
        scal(beta);
        for (int i = 0; i < m_; i++)
            for (int j = 0; j < i; j++)
            {
                val_[i * m_ + j] += alpha * a.val_[j * m_ + i];
                val_[j * m_ + i] += alpha * a.val_[i * m_ + j];
            }
        for (int i = 0; i < m_; i++)
            val_[i * m_ + i] += alpha * a.val_[i * m_ + i];
#endif
    }
}

////////////////////////////////////////////////////////////////////////////////
// set to zero upper or lower part of a lower or upper triangular matrix
////////////////////////////////////////////////////////////////////////////////
template <class T>
void DistMatrix<T>::trset(const char uplo)
{
    if (active_)
    {

        assert(m_ == n_);

        if (uplo == 'l' || uplo == 'L')
        {
            for (int li = 0; li < mblocks_; li++)
            {
                for (int lj = 0; lj < nblocks_; lj++)
                {
                    for (int ii = 0; ii < mbs(li); ii++)
                    {
                        for (int jj = 0; jj < nbs(lj); jj++)
                        {
                            if (i(li, ii) < j(lj, jj))
                                val_[(ii + li * mb_) + (jj + lj * nb_) * mloc_]
                                    = 0.0;
                        }
                    }
                }
            }
        }
        else
        {
            for (int li = 0; li < mblocks_; li++)
            {
                for (int lj = 0; lj < nblocks_; lj++)
                {
                    for (int ii = 0; ii < mbs(li); ii++)
                    {
                        for (int jj = 0; jj < nbs(lj); jj++)
                        {
                            if (i(li, ii) > j(lj, jj))
                                val_[(ii + li * mb_) + (jj + lj * nb_) * mloc_]
                                    = 0.0;
                        }
                    }
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
template <class T>
void DistMatrix<T>::init(const T* const a, const int lda)
{
    if (active_)
    {
        for (int li = 0; li < mblocks_; li++)
        {
            for (int lj = 0; lj < nblocks_; lj++)
            {
                for (int ii = 0; ii < mbs(li); ii++)
                {
                    for (int jj = 0; jj < nbs(lj); jj++)
                    {
                        val_[(ii + li * mb_) + (jj + lj * nb_) * mloc_]
                            = a[i(li, ii) + j(lj, jj) * lda];
                    }
                }
            }
        }
    }
}

template <class T>
void DistMatrix<T>::add(const T* const a, const int lda)
{
    if (active_)
    {
        for (int li = 0; li < mblocks_; li++)
        {
            for (int lj = 0; lj < nblocks_; lj++)
            {
                for (int ii = 0; ii < mbs(li); ii++)
                {
                    for (int jj = 0; jj < nbs(lj); jj++)
                    {
                        val_[(ii + li * mb_) + (jj + lj * nb_) * mloc_]
                            += a[i(li, ii) + j(lj, jj) * lda];
                    }
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
template <class T>
double DistMatrix<T>::sumProdElements(const DistMatrix<T>& a) const
{
    assert(mblocks_ == a.mblocks_);
    assert(nblocks_ == a.nblocks_);
    assert(mb_ == a.mb_);
    assert(nb_ == a.nb_);

    double tsum = 0.;

    if (active_)
    {
        for (int li = 0; li < mblocks_; li++)
        {
            for (int lj = 0; lj < nblocks_; lj++)
            {
                for (int ii = 0; ii < mbs(li); ii++)
                {
                    for (int jj = 0; jj < nbs(lj); jj++)
                    {
                        tsum += (double)val_[(ii + li * mb_)
                                             + (jj + lj * nb_) * mloc_]
                                * (double)a.val_[(ii + li * mb_)
                                                 + (jj + lj * nb_) * mloc_];
                    }
                }
            }
        }
    }

    double sum = 0.;
#ifdef MGMOL_USE_SCALAPACK
    MPI_Allreduce(&tsum, &sum, 1, MPI_DOUBLE, MPI_SUM, comm_global_);
#else
    sum = tsum;
#endif
    return sum;
}

////////////////////////////////////////////////////////////////////////////////
// Constructor for a distributed matrix
// from a duplicated matrix
////////////////////////////////////////////////////////////////////////////////
template <class T>
DistMatrix<T>::DistMatrix(const std::string& name, const BlacsContext& bc,
    const int lda, const T* const a, const int m, const int n, const int mb,
    const int nb)
    : object_name_(name), bc_(bc), comm_global_(bc.comm_global())
{
    resize(m, n, mb, nb);
    init(a, lda);
}

template <class T>
DistMatrix<T>::DistMatrix(const std::string& name, const BlacsContext& bc,
    const int lda, const T* const a, const int m, const int n)
    : object_name_(name), bc_(bc), comm_global_(bc.comm_global())
{
    resize(m, n, distmatrix_def_block_size_, distmatrix_def_block_size_);
    init(a, lda);
}

////////////////////////////////////////////////////////////////////////////////
// Generate a distributed matrix from a replicated one
//
template <>
void DistMatrix<double>::initFromReplicated(double* const src, const int lda)
{
#ifdef MGMOL_USE_SCALAPACK
    // all the nodes send data
    const int rev = 1;
    const int ii  = 1;
    const int jj  = 1;
    const int i   = 1;
    if (active_)
        pdlacp3(&m_, &i, val_.data(), desc_, src, &lda, &ii, &jj, &rev);
#else
    memcpy(&val_[0], src, lda * n_ * sizeof(double));
#endif
}

////////////////////////////////////////////////////////////////////////////////
//
// Generate a duplicated matrix from a distributed matrix
//
////////////////////////////////////////////////////////////////////////////////
template <>
void DistMatrix<double>::allgather(double* const a, const int lda) const
{
    allgather_tm_.start();

    assert(m_ == n_);

#ifdef MGMOL_USE_SCALAPACK
    // all the nodes receive the data
    const int ii  = -1;
    const int jj  = -1;
    const int rev = 0;
    const int i   = 1;
    if (active_) pdlacp3(&m_, &i, val_.data(), desc_, a, &lda, &ii, &jj, &rev);

    // int nrpocs=bc_.size();
    int nrpocs     = bc_.nprocs();
    int n_active   = bc_.nprow() * bc_.npcol();
    int n_inactive = nrpocs - n_active;
    int ns         = n_inactive / n_active;
    if (ns * n_active < n_inactive) ns++;
    int mpirc = MPI_SUCCESS;
    int size  = lda * n_;
    int mype  = bc_.mype();
    MPI_Status status;
    // send data from active PEs to inactive PEs
    for (int i = 0; i < ns; i++)
    {
        if (!active_ && mype >= (i + 1) * n_active && mype < (i + 2) * n_active)
        {
            int rpe = mype % n_active; // remote PE
            mpirc
                = MPI_Recv(a, size, MPI_DOUBLE, rpe, 0, comm_global_, &status);
        }
        if (active_ && (mype + (i + 1) * n_active) < nrpocs)
            mpirc = MPI_Send(a, size, MPI_DOUBLE, mype + (i + 1) * n_active, 0,
                comm_global_);
        if (mpirc != MPI_SUCCESS)
            std::cerr
                << "DistMatrix<T>::allgather: MPI Send/Recv failed for PE "
                << mype << std::endl;
    }
#else
    memcpy(a, &val_[0], lda * n_ * sizeof(double));

#endif
    allgather_tm_.stop();
}

template <>
void DistMatrix<float>::allgather(float* const a, const int lda) const
{
    allgather_tm_.start();

    assert(m_ == n_);

#ifdef MGMOL_USE_SCALAPACK
    const int ii  = -1;
    const int jj  = -1;
    const int rev = 0;
    const int i   = 1;
    if (active_) pslacp3(&m_, &i, val_.data(), desc_, a, &lda, &ii, &jj, &rev);

    // int nrpocs=bc_.size();
    int nrpocs     = bc_.nprocs();
    int n_active   = bc_.nprow() * bc_.npcol();
    int n_inactive = nrpocs - n_active;
    int ns         = n_inactive / n_active;
    if (ns * n_active < n_inactive) ns++;
    int mpirc = MPI_SUCCESS;
    int size  = lda * n_;
    int mype  = bc_.mype();
    MPI_Status status;
    // send data from active PEs to inactive PEs
    for (int i = 0; i < ns; i++)
    {
        if (!active_ && mype >= (i + 1) * n_active && mype < (i + 2) * n_active)
        {
            int rpe = mype % n_active; // remote PE
            mpirc = MPI_Recv(a, size, MPI_FLOAT, rpe, 0, comm_global_, &status);
        }
        if (active_ && (mype + (i + 1) * n_active) < nrpocs)
            mpirc = MPI_Send(
                a, size, MPI_FLOAT, mype + (i + 1) * n_active, 0, comm_global_);
        if (mpirc != MPI_SUCCESS)
            std::cerr
                << "DistMatrix<T>::allgather: MPI Send/Recv failed for PE "
                << mype << std::endl;
    }
#else
    memcpy(a, &val_[0], lda * n_ * sizeof(float));

#endif
    allgather_tm_.stop();
}
////////////////////////////////////////////////////////////////////////////////
// Constructor for a diagonal distributed matrix
// from the diagonal elements
////////////////////////////////////////////////////////////////////////////////
template <class T>
DistMatrix<T>::DistMatrix(const std::string& name, const BlacsContext& bc,
    const T* const dmat, const int m, const int n, const int mb, const int nb)
    : object_name_(name), bc_(bc), comm_global_(bc.comm_global())
{
    assert(m == n);
    assert(mb == nb);

    resize(m, n, mb, nb);
    setDiagonalValues(dmat);
}

template <class T>
DistMatrix<T>::DistMatrix(const std::string& name, const BlacsContext& bc,
    const T* const dmat, const int m, const int n)
    : object_name_(name), bc_(bc), comm_global_(bc.comm_global())
{
    resize(m, n, distmatrix_def_block_size_, distmatrix_def_block_size_);
    setDiagonalValues(dmat);
}

template <class T>
DistMatrix<T>::DistMatrix(
    const std::string& name, const T* const dmat, const int m)
    : object_name_(name),
      bc_(*default_bc_),
      comm_global_(default_bc_->comm_global())
{
    resize(m, m, distmatrix_def_block_size_, distmatrix_def_block_size_);
    setDiagonalValues(dmat);
}

template <class T>
void DistMatrix<T>::setDiagonalValues(const T* const dmat)
{
    if (active_)
    {
        if (myrow_ == mycol_)
        {
            for (int li = 0; li < nblocks_; li++)
            {
                for (int ii = 0; ii < nbs(li); ii++)
                {
                    val_[(ii + li * nb_) * (mloc_ + 1)] = dmat[i(li, ii)];
                }
            }
        }
    }
}

template <class T>
void DistMatrix<T>::getDiagonalValues(T* const dmat) const
{
    memset(dmat, 0, m_ * sizeof(T));

    if (active_)
    {
        if (myrow_ == mycol_)
        {
            for (int li = 0; li < nblocks_; li++)
            {
                for (int ii = 0; ii < nbs(li); ii++)
                {
                    dmat[i(li, ii)] = val_[(ii + li * nb_) * (mloc_ + 1)];
                }
            }
        }
    }
    std::vector<T> recvbuf_vec(m_);
    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    T* recvbuf      = recvbuf_vec.data();
    mmpi.allreduce(dmat, recvbuf, m_, MPI_SUM);
    //  MPI_Allreduce ( dmat, recvbuf, m_,
    //                   MPI_DOUBLE, MPI_SUM, comm_global_ );
    memcpy(dmat, recvbuf, m_ * sizeof(T));
}

////////////////////////////////////////////////////////////////////////////////
// Compute the trace of the DistMatrix
////////////////////////////////////////////////////////////////////////////////
template <>
double DistMatrix<double>::trace(void) const
{
    assert(m_ == n_);
    double trace = 0.0;

    if (active_)
    {
#ifdef MGMOL_USE_SCALAPACK
        int ione = 1;
        trace    = pdlatra(&n_, &val_[0], &ione, &ione, desc_);
#else
        int inc = m_ + 1;
        for (int i = 0; i < n_; i++)
            trace += val_[i * inc];
#endif
    }

    MPI_Bcast(&trace, 1, MPI_DOUBLE, 0, comm_global_);
    return trace;
}

template <>
double DistMatrix<float>::trace(void) const
{
    assert(m_ == n_);
    double trace = 0.0;

    if (active_)
    {
#ifdef MGMOL_USE_SCALAPACK
        int ione = 1;
        trace    = (double)pslatra(&n_, &val_[0], &ione, &ione, desc_);
#else
        int inc = m_ + 1;
        for (int i = 0; i < n_; i++)
            trace += val_[i * inc];
#endif
    }

    MPI_Bcast(&trace, 1, MPI_DOUBLE, 0, comm_global_);
    return trace;
}
////////////////////////////////////////////////////////////////////////////////
// Reduces a real  symmetric-definite  generalized  eigenproblem  to  standard
// form.  If itype = 1, the problem is A*x = lambda*B*x,
// and A (=*this) is overwritten by inv(U**T)*A*inv(U) or inv(L)*A*inv(L**T)
// If itype = 2 or 3, the problem is A*B*x = lambda*x or
// B*A*x = lambda*x, and *this is overwritten by U*A*U**T or L**T*A*L.
// B must have been previously factorized as U**T*U or L*L**T by
// DistMatrix<double>::dpotrf.
////////////////////////////////////////////////////////////////////////////////
template <>
void DistMatrix<double>::sygst(
    int itype, char uplo, const DistMatrix<double>& b)
{
    int info;
    if (active_)
    {
        assert(m_ == n_);

#ifdef MGMOL_USE_SCALAPACK
        int ione = 1;
        double scale;
        pdsygst(&itype, &uplo, &m_, &val_[0], &ione, &ione, desc_, &b.val_[0],
            &ione, &ione, b.desc_, &scale, &info);
#else
        dsygst(&itype, &uplo, &m_, &val_[0], &m_, &b.val_[0], &b.m_, &info);
#endif
        if (info != 0)
        {
            MGMOL_DISTMATRIX_ERROR(
                "sygst, object " << object_name_ << ", info=" << info);
        }
    }
}

template <>
void DistMatrix<float>::sygst(int itype, char uplo, const DistMatrix<float>& b)
{
    int info;
    if (active_)
    {
        assert(m_ == n_);

#ifdef MGMOL_USE_SCALAPACK
        int ione = 1;
        float scale;
        pssygst(&itype, &uplo, &m_, &val_[0], &ione, &ione, desc_, &b.val_[0],
            &ione, &ione, b.desc_, &scale, &info);
#else
        ssygst(&itype, &uplo, &m_, &val_[0], &m_, &b.val_[0], &b.m_, &info);
#endif
        if (info != 0)
        {
            MGMOL_DISTMATRIX_ERROR("sygst, info=" << info);
        }
    }
}
////////////////////////////////////////////////////////////////////////////////
template <>
void DistMatrix<double>::syev(
    char jobz, char uplo, std::vector<double>& w, DistMatrix<double>& z)
{
    int info;
    if (active_)
    {
        assert(m_ == n_);

#ifdef MGMOL_USE_SCALAPACK
        assert(mb_ > 1);
        int ione = 1, izero = 0;
        int nn    = std::max(mb_, m_);
        int np0   = NUMROC(&nn, &mb_, &izero, &izero, &nprow_);
        int lwork = 5 * m_ + 3 * np0 + mb_ * nn + m_ * np0;
        lwork *= 2;
        std::vector<double> work(lwork);
        pdsyev(&jobz, &uplo, &m_, &val_[0], &ione, &ione, desc_, &w[0],
            &z.val_[0], &ione, &ione, z.desc_, work.data(), &lwork, &info);
#else
        int lwork = 3 * m_;
        std::vector<double> work(lwork);
        z = *this;
        dsyev(&jobz, &uplo, &m_, &z.val_[0], &m_, &w[0], work.data(), &lwork,
            &info);
#endif
        if (info != 0)
        {
            MGMOL_DISTMATRIX_ERROR(
                "syev, object " << object_name_ << ", info=" << info);
        }
    }

#ifdef MGMOL_USE_SCALAPACK
    MPI_Bcast(&w[0], m_, MPI_DOUBLE, 0, comm_global_);
#endif
}

template <>
void DistMatrix<float>::syev(
    char jobz, char uplo, std::vector<float>& w, DistMatrix<float>& z)
{
    int info;
    if (active_)
    {
        assert(m_ == n_);

#ifdef MGMOL_USE_SCALAPACK
        assert(mb_ > 1);
        int ione = 1, izero = 0;
        int nn    = std::max(mb_, m_);
        int np0   = NUMROC(&nn, &mb_, &izero, &izero, &nprow_);
        int lwork = 5 * m_ + 3 * np0 + mb_ * nn + m_ * np0;
        lwork *= 2;
        std::vector<float> work(lwork);
        pssyev(&jobz, &uplo, &m_, &val_[0], &ione, &ione, desc_, &w[0],
            &z.val_[0], &ione, &ione, z.desc_, work.data(), &lwork, &info);
#else
        int lwork = 3 * m_;
        std::vector<float> work(lwork);
        z = *this;
        ssyev(&jobz, &uplo, &m_, &z.val_[0], &m_, &w[0], work.data(), &lwork,
            &info);
#endif
        if (info != 0)
        {
            MGMOL_DISTMATRIX_ERROR(
                "syev, object " << object_name_ << ", info=" << info);
        }
    }

#ifdef MGMOL_USE_SCALAPACK
    MPI_Bcast(&w[0], m_, MPI_FLOAT, 0, comm_global_);
#endif
}
////////////////////////////////////////////////////////////////////////////////
template <>
void DistMatrix<double>::gesvd(char jobu, char jobvt, std::vector<double>& s,
    DistMatrix<double>& u, DistMatrix<double>& vt)
{
    int info;
    if (active_)
    {
        assert(m_ == n_);

        int lwork = -1;
        double qwork[1];
#ifdef MGMOL_USE_SCALAPACK
        assert(mb_ > 1);
        int ione = 1;
        pdgesvd(&jobu, &jobvt, &m_, &n_, &val_[0], &ione, &ione, desc_, &s[0],
            &u.val_[0], &ione, &ione, u.desc_, &vt.val_[0], &ione, &ione,
            vt.desc_, &qwork[0], &lwork, &info);
        lwork = qwork[0];
        std::vector<double> work(lwork);
        pdgesvd(&jobu, &jobvt, &m_, &n_, &val_[0], &ione, &ione, desc_, &s[0],
            &u.val_[0], &ione, &ione, u.desc_, &vt.val_[0], &ione, &ione,
            vt.desc_, work.data(), &lwork, &info);
#else
        dgesvd(&jobu, &jobvt, &m_, &n_, &val_[0], &m_, &s[0], &u.val_[0], &m_,
            &vt.val_[0], &n_, qwork, &lwork, &info);
        lwork = qwork[0];
        dgesvd(&jobu, &jobvt, &m_, &n_, &val_[0], &m_, &s[0], &u.val_[0], &m_,
            &vt.val_[0], &n_, qwork, &lwork, &info);
#endif
        if (info != 0)
        {
            MGMOL_DISTMATRIX_ERROR(
                "gesvd, object " << object_name_ << ", info=" << info);
        }
    }

#ifdef MGMOL_USE_SCALAPACK
    int m = std::min(m_, n_);
    MPI_Bcast(&s[0], m, MPI_DOUBLE, 0, comm_global_);
#endif
}

template <>
void DistMatrix<float>::gesvd(char jobu, char jobvt, std::vector<float>& s,
    DistMatrix<float>& u, DistMatrix<float>& vt)
{
    int info;
    if (active_)
    {
        assert(m_ == n_);

#ifdef MGMOL_USE_SCALAPACK
        assert(mb_ > 1);
        int ione  = 1;
        int lwork = -1;
        float qwork[1];
        psgesvd(&jobu, &jobvt, &m_, &n_, &val_[0], &ione, &ione, desc_, &s[0],
            &u.val_[0], &ione, &ione, u.desc_, &vt.val_[0], &ione, &ione,
            vt.desc_, &qwork[0], &lwork, &info);
        lwork = qwork[0];
        std::vector<float> work(lwork);
        psgesvd(&jobu, &jobvt, &m_, &n_, &val_[0], &ione, &ione, desc_, &s[0],
            &u.val_[0], &ione, &ione, u.desc_, &vt.val_[0], &ione, &ione,
            vt.desc_, work.data(), &lwork, &info);
#else
        int lwork = -1;
        float qwork[1];
        sgesvd(&jobu, &jobvt, &m_, &n_, &val_[0], &m_, &s[0], &u.val_[0], &m_,
            &vt.val_[0], &n_, qwork, &lwork, &info);
        lwork = qwork[0];
        sgesvd(&jobu, &jobvt, &m_, &n_, &val_[0], &m_, &s[0], &u.val_[0], &m_,
            &vt.val_[0], &n_, qwork, &lwork, &info);
#endif
        if (info != 0)
        {
            MGMOL_DISTMATRIX_ERROR(
                "gesvd, object " << object_name_ << ", info=" << info);
        }
    }

#ifdef MGMOL_USE_SCALAPACK
    int m = std::min(m_, n_);
    MPI_Bcast(&s[0], m, MPI_FLOAT, 0, comm_global_);
#endif
}

////////////////////////////////////////////////////////////////////////////////
template <class T>
void DistMatrix<T>::sygv(char /*jobz*/, char uplo, const char fac,
    DistMatrix<T>& b, std::vector<T>& w, DistMatrix<T>& z)
{
    // compute Cholesky factorization of b if necessary
    if (fac != 'c') b.potrf(uplo);

    // Transform the generalized eigenvalue problem to a standard form
    sygst(1, uplo, b);

    // solve a standard symmetric eigenvalue problem
    syev('v', uplo, w, z);

    // Get the eigenvectors Z of the generalized eigenvalue problem
    // Solve Z=L**(-T)*U
    b.trtrs(uplo, 't', 'n', z);
}

////////////////////////////////////////////////////////////////////////////////
// get max in absolute value of column j
template <>
int DistMatrix<double>::iamax(const int j, double& val)
{
    int indx = -1;
    int incx = 1;
#ifdef MGMOL_USE_SCALAPACK
    int ix          = 1;
    int jx          = y(j) + 1; // C to fortran
    int proc_col    = pc(j);
    int desc_col[9] = { desc_[0], desc_[1], desc_[2], 1, desc_[4], 1, desc_[6],
        proc_col, desc_[8] };

    if (mycol_ == proc_col)
        pdamax(&m_, &val, &indx, &val_[y(j) * desc_[8]], &ix, &jx, desc_col,
            &incx);
    indx--; // fortran to C

    int proot = proc_col;
    MPI_Bcast(&indx, 1, MPI_INT, proot, comm_global_);
    MPI_Bcast(&val, 1, MPI_DOUBLE, proot, comm_global_);
#else
    indx     = idamax(&m_, &val_[0] + m_ * j, &incx) - 1;
    val      = val_[m_ * j + indx];
#endif
    return indx;
}

template <>
int DistMatrix<float>::iamax(const int j, float& val)
{
    int indx = -1;
    int incx = 1;
#ifdef MGMOL_USE_SCALAPACK
    int ix          = 1;
    int jx          = y(j) + 1; // C to fortran
    int proc_col    = pc(j);
    int desc_col[9] = { desc_[0], desc_[1], desc_[2], 1, desc_[4], 1, desc_[6],
        proc_col, desc_[8] };

    if (mycol_ == proc_col)
        psamax(&m_, &val, &indx, &val_[y(j) * desc_[8]], &ix, &jx, desc_col,
            &incx);
    indx--; // fortran to C

    int proot = proc_col;
    MPI_Bcast(&indx, 1, MPI_INT, proot, comm_global_);
    MPI_Bcast(&val, 1, MPI_FLOAT, proot, comm_global_);
#else
    indx     = isamax(&m_, &val_[0] + m_ * j, &incx) - 1;
    val      = val_[m_ * j + indx];
#endif
    return indx;
}
////////////////////////////////////////////////////////////////////////////////
template <>
void DistMatrix<double>::swapColumns(const int j1, const int j2)
{
    if (j1 == j2) return;
#ifdef MGMOL_USE_SCALAPACK
    int jx   = j1 + 1; // C to Fortran
    int jy   = j2 + 1; // C to Fortran
    int ione = 1;
    int inc  = 1; // indicates swap columns
    if (active_)
        pdswap(&m_, &val_[0], &ione, &jx, desc_, &inc, &val_[0], &ione, &jy,
            desc_, &inc);
#else
    int ione = 1;
    dswap(&m_, &val_[0] + m_ * j1, &ione, &val_[0] + m_ * j2, &ione);
#endif
}

template <>
void DistMatrix<float>::swapColumns(const int j1, const int j2)
{
    if (j1 == j2) return;
#ifdef MGMOL_USE_SCALAPACK
    int jx   = j1 + 1; // C to Fortran
    int jy   = j2 + 1; // C to Fortran
    int ione = 1;
    int inc  = 1; // indicates swap columns
    if (active_)
        psswap(&m_, &val_[0], &ione, &jx, desc_, &inc, &val_[0], &ione, &jy,
            desc_, &inc);
#else
    int ione = 1;
    sswap(&m_, &val_[0] + m_ * j1, &ione, &val_[0] + m_ * j2, &ione);
#endif
}
////////////////////////////////////////////////////////////////////////////////
template <class T>
void DistMatrix<T>::print(std::ostream& os) const
{
    if (m_ == 0 || n_ == 0) return;
    BlacsContext bcl(comm_global_, 1, 1);
#if 0
  // First version: uses a complete copy of *this on process (0,0)
  DistMatrix<T> t(bcl,m_,n_,mb_,nb_);
  t = *this;
  if ( t.active() )
  {
    // this is done only on pe 0
    os << " print: (m,n)=" << "(" << m_ << "," << n_ << ")" << std::endl;
    for ( int ii = 0; ii < m_; ii++ )
    {
      for ( int jj = 0; jj < n_; jj++ )
      {
        os << "(" << ii << "," << jj << ")=" << t.val_[ii+t.mloc()*jj] << std::endl;
      }
    }
  }
#else
    // Copy blocks of <blocksize> columns and print them on process (0,0)
    const int blockmemsize = 32768; // maximum memory size of a block in bytes
    // compute maximum block size: must be at least 1
    int maxbs = std::max(1, (int)((blockmemsize / sizeof(T)) / m_));
    DistMatrix<T> t("DistMatrix<T>::print", bcl, m_, maxbs);
    int nblocks = n_ / maxbs + ((n_ % maxbs == 0) ? 0 : 1);
    int ia      = 0;
    int ja      = 0;
    for (int jb = 0; jb < nblocks; jb++)
    {
        int blocksize = ((jb + 1) * maxbs > n_) ? n_ % maxbs : maxbs;
        t.getsub(*this, t.m(), blocksize, ia, ja);
        ja += blocksize;
        if (t.active())
        {
            os << std::scientific;
            // this is done only on pe 0
            for (int jj = 0; jj < blocksize; jj++)
            {
                for (int ii = 0; ii < m_; ii++)
                {
                    os << "(" << ii << "," << jj + jb * maxbs
                       << ")=" << t.val_[ii + t.mloc() * jj] << std::endl;
                }
            }
        }
    }
#endif
}

////////////////////////////////////////////////////////////////////////////////
template <class T>
void DistMatrix<T>::printMM(std::ostream& os) const
{
    if (m_ == 0 || n_ == 0) return;
    BlacsContext bcl(comm_global_, 1, 1);
    // Copy blocks of <blocksize> columns and print them on process (0,0)
    const int blockmemsize = 32768; // maximum memory size of a block in bytes
    // compute maximum block size: must be at least 1
    const int maxbs = std::max(1, (int)((blockmemsize / sizeof(T)) / m_));
    DistMatrix<T> t("DistMatrix<T>::print", bcl, m_, maxbs);
    const int nblocks = n_ / maxbs + ((n_ % maxbs == 0) ? 0 : 1);
    int ia            = 0;
    int ja            = 0;
    if (t.active())
    {
        os << "%%MatrixMarket matrix array real general" << std::endl;
        os << m_ << " " << n_ << std::endl;
    }
    for (int jb = 0; jb < nblocks; jb++)
    {
        const int blocksize = ((jb + 1) * maxbs > n_) ? n_ % maxbs : maxbs;
        t.getsub(*this, t.m(), blocksize, ia, ja);
        ja += blocksize;
        if (t.active()) // this is done only on pe 0
        {
            for (int jj = 0; jj < blocksize; jj++)
            {
                // print one column
                for (int ii = 0; ii < m_; ii++)
                {
                    os << t.val_[ii + t.mloc() * jj] << std::endl;
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
template <class T>
void DistMatrix<T>::axpyColumn(
    const int icol, const double alpha, const DistMatrix<T>& x)
{
    const int ni = mloc_ * icol;
    LinearAlgebraUtils<MemorySpace::Host>::MPaxpy(
        mloc_, alpha, &x.val_[ni], &val_[ni]);
}

////////////////////////////////////////////////////////////////////////////////
template <class T>
void DistMatrix<T>::scalColumn(const int icol, const double alpha)
{
    const int ni = mloc_ * icol;
    LinearAlgebraUtils<MemorySpace::Host>::MPscal(mloc_, alpha, &val_[ni]);
}

////////////////////////////////////////////////////////////////////////////////
template <class T>
void DistMatrix<T>::print(std::ostream& os, const int ia, const int ja,
    const int ma, const int na) const
{
    const int m = std::min(ma, std::max(m_ - ia, 0));
    const int n = std::min(na, std::max(n_ - ja, 0));
    BlacsContext bcl(comm_global_, 1, 1);
    DistMatrix<T> t("DistMatrix<T>::print", bcl, m, n);

    MPI_Barrier(comm_global_);
    t.getsub(*this, m, n, ia, ja);
    if (t.active())
    {
        os << std::setprecision(5);
        for (int ii = 0; ii < m; ii++)
        {
            // this is done only on pe 0
            for (int jj = 0; jj < n; jj++)
            {
                os << t.val_[ii + m * jj] << "\t";
            }
            os << "\n";
        }
    }
}
////////////////////////////////////////////////////////////////////////////////
template <>
float DistMatrix<float>::getVal(const int i, const int j) const
{
    char scope = 'A';
    char top   = ' '; // default topology
    // C -> Fortran
    int ia = i + 1;
    int ja = j + 1;

    float val;
    if (active_) pselget(&scope, &top, &val, val_.data(), &ia, &ja, desc_);
    MPI_Bcast(&val, 1, MPI_FLOAT, 0, comm_global_);
    return val;
}

template <>
double DistMatrix<double>::getVal(const int i, const int j) const
{
    char scope = 'A';
    char top   = ' '; // default topology
    // C -> Fortran
    int ia = i + 1;
    int ja = j + 1;

    double val;
    if (active_) pdelget(&scope, &top, &val, val_.data(), &ia, &ja, desc_);
    MPI_Bcast(&val, 1, MPI_DOUBLE, 0, comm_global_);
    return val;
}

// get value for global index (i,j)
// assuming we are on the right processor to get it!
template <class T>
T DistMatrix<T>::getValOnPE(const int i, const int j) const
{
    assert(i < m_);
    assert(j < n_);
    assert(i >= 0);
    assert(j >= 0);

    const int ib = i / (nprow_ * mb_);
    const int jb = j / (npcol_ * nb_);
    const int x  = i % mb_;
    const int y  = j % nb_;

    assert(ib * mb_ + x + (jb * nb_ + y) * mloc_ < size_);
    return val_[ib * mb_ + x + (jb * nb_ + y) * mloc_];
}

////////////////////////////////////////////////////////////////////////////////
template <class T>
MPI_Comm DistMatrix<T>::comm_global() const
{
    return comm_global_;
}

////////////////////////////////////////////////////////////////////////////////
template <class T>
std::ostream& operator<<(std::ostream& os, const DistMatrix<T>& a)
{
    a.print(os);
    return os;
}

template <>
void DistMatrix<double>::setVal(const int i, const int j, const double val)
{
    // C -> Fortran
    int ii = i + 1;
    int jj = j + 1;

    pdelset(val_.data(), &ii, &jj, desc_, &val);
}

template <>
void DistMatrix<float>::setVal(const int i, const int j, const float val)
{
    // C -> Fortran
    int ia = i + 1;
    int ja = j + 1;

    pselset(val_.data(), &ia, &ja, desc_, &val);
}

template std::ostream& operator<<(std::ostream& os, const DistMatrix<double>&);
template std::ostream& operator<<(std::ostream& os, const DistMatrix<float>&);

template class DistMatrix<double>;
template class DistMatrix<float>;

} // namespace

// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "AndersonMix.h"
#include "ExtendedGridOrbitals.h"
#include "LocGridOrbitals.h"
#include "MPIdata.h"
#include "global.h"
#include "lapack_c.h"

#ifdef TESTING
#include "Solution.h"
#endif

#include "MGmol_abort.h"
#include <iomanip>
#include <iostream>
using namespace std;

static const double min_det_mat = 0.01;
static const double max_theta   = 0.5;
static const double min_theta   = -3.;

// #define DEBUG 0

template <class T>
AndersonMix<T>::AndersonMix(const int m, const double beta, T& x) : m_(m), x_(x)
{
    mm_   = -1;
    beta_ = beta;
    if (m_ > 0)
    {
        xi_.resize(m_);
        fi_.resize(m_);
        for (int i = 0; i < m_; i++)
        {
            xi_[i] = new T("xi", x);
            fi_[i] = new T("fi", x);
        }
        mat_.resize(m_ * m_);
        rhs_.resize(m_);
        theta_.resize(m_);
    }
}

template <class T>
AndersonMix<T>::~AndersonMix()
{
    for (int i = 0; i < m_; i++)
        assert(xi_[i] != nullptr);
    for (int i = 0; i < m_; i++)
        assert(fi_[i] != nullptr);

    for (int i = 0; i < m_; i++)
        delete xi_[i];
    for (int i = 0; i < m_; i++)
        delete fi_[i];
}

template <class T>
void AndersonMix<T>::restart(void)
{
    mm_ = -1;
}

template <class T>
void AndersonMix<T>::update(T& f, T& work, ostream& os, const bool verbose)
{
    update_tm_.start();
#ifdef DEBUG
    os << "AndersonMix::update()" << endl;
#endif
    int ione = 1;

    if (mm_ < m_) mm_++;

    // os<<"mm_="<<mm_<<endl;

    if (mm_ > 0)
    {
        // compute mat_ and rhs_
        for (int i = 0; i < mm_; i++)
        {
            work.assign(f);
            assert(fi_[i] != nullptr);
            work -= (*fi_[i]);

            mat_[i * m_ + i] = work.dotProduct(work);
            rhs_[i]          = work.dotProduct(f);

            // non-diagonal terms
            if (i > 0)
            {
                T* tmp1 = new T("AndersonMix_tmp", f);
                for (int j = 0; j < i; j++)
                {
                    if (j > 0) tmp1->assign(f);

                    (*tmp1) -= (*fi_[j]);

                    mat_[j * m_ + i] = work.dotProduct(*tmp1);
                }
                delete tmp1;
            }
        }

        int info;
        char uplo = 'l';
        bool flag = true;
        while (flag)
        {
            flag = false;

            for (int i = 0; i < mm_; i++)
            {
                theta_[i] = rhs_[i];
            }

            // solve linear system
            int n2 = m_ * mm_;

            // Reduce mm_ until problem well conditioned:
            // Build Gramian to check linear dependence between fi_ vectors
            double* tmp_mem = new double[3 * m_ + mm_ + n2];
            double* work2   = &tmp_mem[0];
            double* w       = &tmp_mem[3 * m_];
            double* tmp_mat = &tmp_mem[3 * m_ + mm_];

            char jobz  = 'N';
            int lwork2 = 3 * mm_ - 1;

            while (mm_ > 1)
            {
                DCOPY(&n2, &mat_[0], &ione, &tmp_mat[0], &ione);
                // rescale matrix to have 1. as diagonal entries
                // (equivalent to having normalized vectors)
                for (int i = 0; i < mm_; i++)
                {
                    double alpha = 1. / sqrt(tmp_mat[(m_ + 1) * i]);
                    // row i
                    for (int j = 0; j <= i; j++)
                    {
                        tmp_mat[j * m_ + i] *= alpha;
                    }
                    // column i
                    for (int j = i; j < mm_; j++)
                    {
                        tmp_mat[i * m_ + j] *= alpha;
                    }
                }
                DSYEV(&jobz, &uplo, &mm_, &tmp_mat[0], &m_, w, work2, &lwork2,
                    &info);

                double det = 1;
                for (int i = 0; i < mm_; i++)
                    det *= w[i];

                // "volume" = sqrt( determinant )
                if (det < min_det_mat)
                {
                    os << "Det. Anderson matrix=" << det << ", set m=" << mm_
                       << endl;
                    mm_--;
                }
                else
                    break;
            }

#if 0
            if( onpe0 )
            for(int i=0;i<mm_;i++)
            {
                os<<"rhs["<<i<<"]="<<rhs_[i]<<endl;
                os<<"mat["<<i<<"]="<<mat_[i+m_*i]<<endl;
            }
#endif
            DCOPY(&n2, &mat_[0], &ione, &tmp_mat[0], &ione);

            // solve linear system to find Anderson coefficients
            DPOTRF(&uplo, &mm_, &tmp_mat[0], &m_, &info);
            DPOTRS(
                &uplo, &mm_, &ione, &tmp_mat[0], &m_, &theta_[0], &m_, &info);

            delete[] tmp_mem;
            if (info != 0)
            {
                os << "AndersonMix, dpotrs: info=" << info << endl;
                mgmol_abort();
            }

            for (int j = 0; j < mm_; j++)
            {
                if (theta_[j] > max_theta)
                {
                    if (mm_ > 1)
                    {
                        mm_--;
                        os << "Warning: theta[" << j << "]=" << theta_[j]
                           << " > " << max_theta << ", set m=" << mm_ << endl;
                        flag = true;
                        break;
                    }
                    double alpha;
                    if (theta_[j] > 1.)
                    {
                        alpha = -0.5;
                    }
                    else
                    {
                        alpha = 0.; // simple SD
                    }
                    os << "Warning: theta[" << j << "]=" << theta_[j]
                       << " --> reset theta to " << alpha << endl;
                    theta_[j] = alpha;
                }
                else if (theta_[j] < min_theta)
                {
                    if (mm_ > 1)
                    {
                        mm_--;
                        os << "Warning: theta[" << j << "]=" << theta_[j]
                           << " < " << min_theta << ", set m=" << mm_ << endl;
                        flag = true;
                        break;
                    }
                    double alpha = min_theta;
                    os << "Warning: theta[" << j << "]=" << theta_[j]
                       << " --> reset theta to " << alpha << endl;
                    theta_[j] = alpha;
                }
            }
        }

        // #ifdef DEBUG
        if (os.good() && mm_ > 0 && verbose)
        {
            os << "Anderson extrapolation:";
            for (int j = 0; j < mm_; j++)
                os << "  theta[" << j << "]=" << theta_[j];
            os << endl;
        }
        // #endif
    }

    // update x_
    if (m_ > 0)
    {
        // save current x_
        work.assign(x_);

        // compute x bar and save it into x_
        double factor = 1.;
        for (int j = 0; j < mm_; j++)
            factor -= theta_[j];
        if (mm_ > 0) x_.scal(factor);

        for (int j = 0; j < mm_; j++)
        {
            x_.axpy((ORBDTYPE)theta_[j], *xi_[j]);
        }
        // update xi_ for next step
        // restart
        T* tx = xi_[m_ - 1];
        for (int j = m_ - 1; j > 0; j--)
        {
            xi_[j] = xi_[j - 1];
        }
        xi_[0] = tx;
        // keep old x_ in memory
        xi_[0]->assign(work);
    }

    // compute f bar
    if (m_ > 0)
    {
        // save current f
        work.assign(f);

        double factor = 1.;
        for (int j = 0; j < mm_; j++)
            factor -= theta_[j];
        if (mm_ > 0) f.scal(factor);

        for (int j = 0; j < mm_; j++)
        {
            f.axpy((ORBDTYPE)theta_[j], *fi_[j]);
        }

        // update fi_ for next step
        assert(fi_[m_ - 1] != nullptr);
        T* tf = fi_[m_ - 1];
        for (int j = m_ - 1; j > 0; j--)
        {
            assert(fi_[j - 1] != nullptr);
            fi_[j] = fi_[j - 1];
        }
        fi_[0] = tf;
        // keep current f in memory for next call
        fi_[0]->assign(work);
    }

#ifdef DEBUG
    os << "AndersonMix: update x with beta=" << beta_ << endl;
#endif
    // update x_
    if (mm_ > 0)
        x_.axpy((ORBDTYPE)beta_, f);
    else
        x_.axpy((ORBDTYPE)1., f);

    postprocessUpdate();

    update_tm_.stop();
}

#ifdef TESTING
template class AndersonMix<Solution>;
#else
template class AndersonMix<LocGridOrbitals<ORBDTYPE>>;
template class AndersonMix<ExtendedGridOrbitals<ORBDTYPE>>;
#endif

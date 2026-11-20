// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include <iostream>
#include <vector>

#include <mpi.h>

#include "LinearSolverMatrix.h"
#include "MGmol_abort.h"
#include "MPIdata.h"
#include "PreconILU.h"

#define TEMP_DECL template <>

template <class T>
Timer PreconILU<T>::pcilu_setup_tm_("PreconILU::setup");
template <class T>
Timer PreconILU<T>::pcilu_solve_tm_("PreconILU::solve");

template <class T>
PreconILU<T>::PreconILU(LinearSolverMatrix<lsdatatype>& csmat,
    const float droptol, const int maxfill, const int lof)
{
    n_ = csmat.n();
    if (n_ > 0)
    {
        const int nzmax = csmat.nnz();
        L_              = new LinearSolverMatrix<T>(n_, nzmax);
        U_              = new LinearSolverMatrix<T>(n_, nzmax);
        D_.reserve(n_);

        setParams(droptol, maxfill, lof);
    }
    else
    {
        L_ = nullptr;
        U_ = nullptr;
    }
}

template <class T>
PreconILU<T>::~PreconILU()
{
    if (n_ > 0)
    {
        delete L_;
        delete U_;
    }
}
template <class T>
void PreconILU<T>::setup(
    LinearSolverMatrix<lsdatatype>& csmat_, const PCILUTYPE type_)
{
    pcilu_setup_tm_.start();
    if (type_ == PCILUK)
    {
        if (lof_ == 0)
        {
            ilu0(csmat_);
            pctype_ = PCILU0;
        }
        else
        {
            ilukC(csmat_);
            pctype_ = PCILUK;
        }
    }
    else if (type_ == PCMILUT)
    {
        milut(csmat_);
        pctype_ = PCMILUT;
    }
    else if (type_ == PCILUT)
    {
        ilut(csmat_);
        pctype_ = PCILUT;
    }
    else
    {
        ilu0(csmat_);
        pctype_ = PCILU0;
    }
    pcilu_setup_tm_.stop();
}
template <class T>
int PreconILU<T>::nnz_ilu()
{
    int nnz, n1, n2;
    n1  = (*L_).nnz();
    n2  = (*U_).nnz();
    nnz = n1 + n2 + n_;
    return nnz;
}

template <class T>
void PreconILU<T>::LUsolve(double* const y, double* const x) const
{
    /*----------------------------------------------------------------------
     *    performs a forward followed by a backward solve
     *    for LU matrix as produced by ilu factorization
     *    y  = right-hand-side
     *    x  = solution on return
     *--------------------------------------------------------------------*/

    /* -------------begin------------*/
    pcilu_solve_tm_.start();

    const int n = n_;
    memcpy(x, y, n * sizeof(double));

    /*-------------------- L solve */
    (*L_).Lsolve(x);
    /*-------------------- U solve */
    (*U_).Usolve(x, D_);
    pcilu_solve_tm_.stop();
}

template <class T>
void PreconILU<T>::LUsolve(float* const y, float* const x) const
{
    /*----------------------------------------------------------------------
     *    performs a forward followed by a backward solve
     *    for LU matrix as produced by ilu factorization
     *    y  = right-hand-side
     *    x  = solution on return
     *--------------------------------------------------------------------*/

    /* -------------begin------------*/
    pcilu_solve_tm_.start();

    const int n = n_;
    double* z   = new double[n];
    /* copy rhs to before in-place solve */
    for (int i = 0; i < n; i++)
        z[i] = (double)y[i];

    //    memcpy(x, y, n*sizeof(float));

    /*-------------------- L solve */
    (*L_).Lsolve(z);
    /*-------------------- U solve */
    (*U_).Usolve(z, D_);

    /* copy solution */
    for (int i = 0; i < n; i++)
        x[i] = (float)z[i];
    delete[] z;
    pcilu_solve_tm_.stop();
}

template <class T>
void PreconILU<T>::qsplitC(T* a, int* ind, const int n, const int Ncut)
{
    /*----------------------------------------------------------------------
    |     does a quick-sort split of a complex array.
    |     on input a[0 : (n-1)] is a complex array
    |     on output is permuted such that its elements satisfy:
    |
    |     abs(a[i]) >= abs(a[Ncut-1]) for i < Ncut-1 and
    |     abs(a[i]) <= abs(a[Ncut-1]) for i > Ncut-1
    |
    |     ind[0 : (n-1)] is an integer array permuted in the same way as a.
    |---------------------------------------------------------------------*/
    T abskey, tmp;
    int mid, itmp;
    int ncut  = Ncut - 1;
    int first = 0;
    int last  = n - 1;
    if (ncut < first || ncut > last) return;
/* outer loop -- while mid != ncut */
label1:
    mid    = first;
    abskey = std::abs(a[mid]);
    for (int j = first + 1; j <= last; j++)
    {
        if (std::abs(a[j]) > abskey)
        {
            tmp      = a[++mid];
            itmp     = ind[mid];
            a[mid]   = a[j];
            ind[mid] = ind[j];
            a[j]     = tmp;
            ind[j]   = itmp;
        }
    }
    /*-------------------- interchange */
    tmp        = a[mid];
    a[mid]     = a[first];
    a[first]   = tmp;
    itmp       = ind[mid];
    ind[mid]   = ind[first];
    ind[first] = itmp;
    /*-------------------- test for while loop */
    if (mid == ncut) return;
    if (mid > ncut)
        last = mid - 1;
    else
        first = mid + 1;
    goto label1;
}

template <class T>
void PreconILU<T>::qsplitC(
    std::vector<T>& a, std::vector<int>& ind, const int n, const int Ncut)
{
    /*----------------------------------------------------------------------
    |     does a quick-sort split of a complex array.
    |     on input a[0 : (n-1)] is a complex array
    |     on output is permuted such that its elements satisfy:
    |
    |     abs(a[i]) >= abs(a[Ncut-1]) for i < Ncut-1 and
    |     abs(a[i]) <= abs(a[Ncut-1]) for i > Ncut-1
    |
    |     ind[0 : (n-1)] is an integer array permuted in the same way as a.
    |---------------------------------------------------------------------*/
    T abskey, tmp;
    int mid, itmp;
    int ncut  = Ncut - 1;
    int first = 0;
    int last  = n - 1;
    if (ncut < first || ncut > last) return;
/* outer loop -- while mid != ncut */
label1:
    mid    = first;
    abskey = std::abs(a[mid]);
    for (int j = first + 1; j <= last; j++)
    {
        if (std::abs(a[j]) > abskey)
        {
            tmp      = a[++mid];
            itmp     = ind[mid];
            a[mid]   = a[j];
            ind[mid] = ind[j];
            a[j]     = tmp;
            ind[j]   = itmp;
        }
    }
    /*-------------------- interchange */
    tmp        = a[mid];
    a[mid]     = a[first];
    a[first]   = tmp;
    itmp       = ind[mid];
    ind[mid]   = ind[first];
    ind[first] = itmp;
    /*-------------------- test for while loop */
    if (mid == ncut) return;
    if (mid > ncut)
        last = mid - 1;
    else
        first = mid + 1;
    goto label1;
}

template <class T>
void PreconILU<T>::setParams(
    const float droptol, const int maxfill, const int lof)
{
    droptol_ = droptol;
    if (maxfill < 0)
        MaxFil_ = 10000;
    else
        MaxFil_ = maxfill;
    /* use lof_ >= 0 */
    if (lof < 0)
        lof_ = 0;
    else
        lof_ = lof;
}
template <class T>
int PreconILU<T>::lofC(LinearSolverMatrix<lsdatatype>& csmat_)
{
    /*--------------------------------------------------------------------
     * symbolic ilu factorization to calculate structure of ilu matrix
     * for specified level of fill.
     * Returns the pattern of L and U
     *------------------------------------------------------------------*/
    int nzmax = csmat_.nnz();
    /*-------------------- local work arrays */
    std::vector<int> iw(n_, -1);
    std::vector<int> iU;
    iU.resize(n_);
    std::vector<int> levU;
    levU.resize(n_);
    /* define matrix of fill levels for L-part of factorization */
    std::vector<int> llvl;
    llvl.reserve(nzmax);

    /* define iterators */
    std::vector<int>::const_iterator start, end, lstart, lend, row;
    std::vector<int>::iterator ustart, uend, jpiv, uptr, upstart;

    /*-----------------------Beginning of main loop ---------*/
    for (int i = 0; i < n_; i++)
    {
        /* begin ... */
        int k1 = csmat_.nzptrval(i);
        int k2 = csmat_.nzptrval((i + 1));
        start  = csmat_.getColumnIterator(k1);
        end    = csmat_.getColumnIterator(k2);
        /*-------------------- assign lof = 0 for matrix elements */
        std::vector<int>::iterator iL   = iU.begin() + i;
        std::vector<int>::iterator levL = levU.begin() + i;
        int lenl                        = 0;
        int lenu                        = 0;
        for (row = start; row != end; ++row)
        {
            if (*row < i)
            {
                /*-------------------- U-part  */
                iU[lenu]   = *row; //.push_back(row);
                levU[lenu] = 0; //.push_back(0);
                iw[*row]   = lenu++; // iU.size();
            }
            else if (*row > i)
            {
                /*-------------------- L-part  */
                *(iL + lenl)   = *row; //.push_back(row);
                *(levL + lenl) = 0; //.push_back(0);
                iw[*row]       = lenl++; // iL.size();
            }
        }
        /*-------------------- symbolic k,i,j Gaussian elimination  */
        ustart  = iU.begin();
        uend    = ustart + lenu;
        upstart = levU.begin();
        for (jpiv = ustart, uptr = upstart; jpiv != uend; ++jpiv, ++uptr)
        {
            /* find the smallest pivot index.
             * NOTE: This step may be skipped if column indices are
             * already sorted in ascending order
             */
            std::vector<int>::iterator jmin = min_element(jpiv, uend);

            /* now swap current column with smallest column index */
            if (jmin != jpiv)
            {
                std::swap(*jpiv, *jmin); // swap in iU
                std::swap(iw[*jpiv], iw[*jmin]); // swap in iw
                std::swap(*uptr, levU[(jmin - iU.begin())]); // swap in levU
            }
            /* now perform symbolic linear combination of rows */
            int l_k1 = (*L_).nzptrval(*jpiv);
            int l_k2 = (*L_).nzptrval((*jpiv + 1));
            lstart   = (*L_).getColumnIterator(l_k1);
            lend     = (*L_).getColumnIterator(l_k2);
            for (row = lstart; row != lend; ++row)
            {
                int pos = row
                          - (*L_).getColumnIterator(
                              0); /* get the position of row */
                int it = llvl[pos] + *uptr + 1;
                if (it > lof_) continue;
                int ip = iw[*row];
                if (ip == -1)
                {
                    if (*row < i)
                    {
                        iU[lenu]   = *row; //.push_back(row);
                        levU[lenu] = it; //.push_back(it);
                        iw[*row]   = lenu++; // iU.size();
                        /* Fill-in in U-part:- update end of iterator jpiv */
                        uend = ustart + lenu;
                    }
                    else if (*row > i)
                    {
                        *(iL + lenl)   = *row; //.push_back(row);
                        *(levL + lenl) = it; //.push_back(it);
                        iw[*row]       = lenl++; // iL.size();
                    }
                }
                else
                {
                    if (*row < i)
                        levU[ip] = std::min(levU[ip], it);
                    else
                        *(levL + ip) = std::min(*(levL + ip), it);
                }
            }
        }
        /* reset the iw vector */
        uend = iU.begin() + lenu;
        for (std::vector<int>::iterator j = iU.begin(); j != uend; ++j)
            iw[*j] = -1;
        lend = iL + lenl;
        for (std::vector<int>::iterator j = iL; j != lend; ++j)
            iw[*j] = -1;

        /* store the U-part */
        (*U_).initRowNNZPattern(lenu, iU);
        /* store the L-part */
        (*L_).initRowNNZPattern(lenl, iL);
        /* update L-level matrix llvl */
        if (lenl > 0) llvl.insert(llvl.end(), levL, (levL + lenl));
    }
    return 0;
}
template <class T>
int PreconILU<T>::ilukC(LinearSolverMatrix<lsdatatype>& csmat_)
{
    /*----------------------------------------------------------------------------
     * ILUK preconditioner
     * incomplete LU factorization with level of fill dropping
     *----------------------------------------------------------------------------
     * Notes:
     * ======
     * All the diagonals of the input matrix must be non-zero
     *--------------------------------------------------------------------------*/
    int j, k;
    /* symbolic factorization to calculate level of fill index arrays */
    if ((lofC(csmat_)))
    {
        mgmol_abort("ILUK Error: lofC");
    }
    /* initialize non-zero pattern to zero */
    (*U_).initializeNonZeroPattern(0.0);
    (*L_).initializeNonZeroPattern(0.0);
    /* Initialize indicator array jw to -1 */
    std::vector<int> jw(n_, -1);
    /* define iterators over matrix data */
    std::vector<int>::const_iterator jcol, istart, iend, start, end, row;

    /* beginning of main loop */
    const int n = n_;
    for (int i = 0; i < n; i++)
    {
        /* set up the i-th column according to the nonzero information from
           symbolic factorization */

        /* setup array jw[] for i-th row */
        int u_k1 = (*U_).nzptrval(i);
        int u_k2 = (*U_).nzptrval((i + 1));
        start    = (*U_).getColumnIterator(u_k1);
        end      = (*U_).getColumnIterator(u_k2);
        for (row = start, j = u_k1; row != end; ++row, j++)
        { /* initialize U part   */
            jw[*row] = j;
        }
        jw[i] = i; /* diagonal entry */
        //        D_.push_back(MAT_TOL); /* initialize diagonal */
        D_.push_back(1.0e-6); /* initialize diagonal */

        int l_k1 = (*L_).nzptrval(i);
        int l_k2 = (*L_).nzptrval((i + 1));
        start    = (*L_).getColumnIterator(l_k1);
        end      = (*L_).getColumnIterator(l_k2);
        for (row = start, j = l_k1; row != end; ++row, j++)
        { /* initialize L part   */
            jw[*row] = j;
        }

        /* copy columns from csmat into lu */
        int k1 = csmat_.nzptrval(i);
        int k2 = csmat_.nzptrval((i + 1));
        start  = csmat_.getColumnIterator(k1);
        end    = csmat_.getColumnIterator(k2);
        for (row = start, j = k1; row != end; ++row, j++)
        {
            int jpos    = jw[*row];
            const T val = (T)csmat_.getRowEntry(j);
            if (*row < i)
                (*U_).insertMatrixEntry(jpos, *row, val);
            else if (*row == i)
                /*------ diagonal -----*/
                D_[i] = val;
            else
                (*L_).insertMatrixEntry(jpos, *row, val);
        }

        /* eliminate previous cols */
        start = (*U_).getColumnIterator(u_k1);
        end   = (*U_).getColumnIterator(u_k2);
        for (jcol = start, j = u_k1; jcol != end; ++jcol, j++)
        {
            /* get the multiplier for col. to be eliminated (jcol) */
            T fact = (*U_).getRowEntry(j);

            /* combine current col. and col. jcol */
            int i_k1 = (*L_).nzptrval(*jcol);
            int i_k2 = (*L_).nzptrval((*jcol + 1));
            istart   = (*L_).getColumnIterator(i_k1);
            iend     = (*L_).getColumnIterator(i_k2);
            for (row = istart, k = i_k1; row != iend; ++row, k++)
            {
                int jpos = jw[*row];
                if (jpos == -1) continue;
                const T val = -(fact * ((*L_).getRowEntry(k)));
                if (*row < i)
                    (*U_).addToMatrixEntry(jpos, val);
                else if (*row == i)
                    D_[i] += val;
                else
                    (*L_).addToMatrixEntry(jpos, val);
            }
        }

        /* reset double-pointer to -1  */
        start = (*U_).getColumnIterator(u_k1);
        end   = (*U_).getColumnIterator(u_k2);
        for (row = start; row != end; ++row)
        {
            jw[*row] = -1;
        }
        jw[i] = -1;
        start = (*L_).getColumnIterator(l_k1);
        end   = (*L_).getColumnIterator(l_k2);
        for (row = start; row != end; ++row)
        {
            jw[*row] = -1;
        }

        /* update diagonal */
        D_[i] = 1.0 / D_[i];

        /************ Done with diagonal. Now scale column of L by the diagonal
         * *****/
        const T scal  = D_[i];
        T* dptr       = (*L_).getPtrToData(l_k1);
        const int len = l_k2 - l_k1;
        Tscal(len, scal, dptr);
    }
    return 0;
}

template <class T>
template <typename T2>
int PreconILU<T>::ilu0(LinearSolverMatrix<T2>& csmat_)
{
    /*
     * Column-based ILU0
     *--------------------------------------------------------------------------*/
    typedef typename std::vector<T2>::iterator LUvecIterator;
    /*-------------------- local work arrays */
    std::vector<int> iw(n_, -1);
    std::vector<int> iU;
    iU.resize(n_);
    std::vector<T> wU(n_);

    std::vector<int>::const_iterator start, end, lstart, lend, row;
    std::vector<int>::iterator ustart, uend, jpiv;
    TvecIterator ldstart, udstart, ldptr, udptr;
    LUvecIterator dstart, dptr;

    /*-------------------- beginning of main loop */
    /*-----------------------------------------------------------------------*/
    for (int i = 0; i < n_; i++)
    {

        int k1 = csmat_.nzptrval(i);
        int k2 = csmat_.nzptrval((i + 1));
        dstart = csmat_.getDataIterator(k1);

        /*-------------------- unpack L & U-parts of column of A */
        std::vector<int>::iterator iL = iU.begin() + i;
        TvecIterator wL               = wU.begin() + i;
        T dd     = 1.0e-6; // MAT_TOL; /* initialize diagonal entry */
        int lenl = 0;
        int lenu = 0;
        iw[i]    = i;
        /*-------------------- scan & unwrap column */
        start = csmat_.getColumnIterator(k1);
        end   = csmat_.getColumnIterator(k2);
        for (row = start, dptr = dstart; row != end; ++row, ++dptr)
        {
            T t = (T)*dptr;
            if (*row < i)
            {
                iU[lenu] = *row;
                wU[lenu] = t;
                iw[*row] = lenu++;
            }
            else if (*row > i)
            {
                *(iL + lenl) = *row;
                *(wL + lenl) = t;
                iw[*row]     = lenl++;
            }
            else
                dd = t;
        }
        /*-------------------- eliminate column */
        ustart  = iU.begin();
        uend    = ustart + lenu;
        udstart = wU.begin();
        for (jpiv = ustart, udptr = udstart; jpiv != uend; ++jpiv, ++udptr)
        {
            /*-------------------------------------------------------------------------
             *  in order to do the elimination in the correct order we must
             *select the smallest row index among iU[k], k = j, j+1, ..., lenu-1
             *-----------------------------------------------------------------------*/
            std::vector<int>::iterator jmin = min_element(jpiv, uend);
            /* now swap current column with smallest column index */
            if (jmin != jpiv)
            {
                std::swap(*jpiv, *jmin); // swap in iU
                std::swap(iw[*jpiv], iw[*jmin]); // swap in iw
                std::swap(*udptr, wU[(jmin - iU.begin())]); // swap in wU
            }
            /* zero out element - reset pivot */
            iw[*jpiv] = -1;
            /* combine current column and pivot column */
            int l_k1 = (*L_).nzptrval(*jpiv);
            int l_k2 = (*L_).nzptrval((*jpiv + 1));
            lstart   = (*L_).getColumnIterator(l_k1);
            lend     = (*L_).getColumnIterator(l_k2);
            ldstart  = (*L_).getDataIterator(l_k1);
            for (row = lstart, ldptr = ldstart; row != lend; ++row, ++ldptr)
            {
                int ipos = iw[*row];
                if (ipos == -1) continue;

                T lxu = -((*ldptr) * (*udptr));

                if (*row < i)
                {
                    /* dealing with upper part */
                    wU[ipos] += lxu;
                }
                else if (*row > i)
                {
                    /* dealing with lower part */
                    *(wL + ipos) += lxu;
                }
                else
                {
                    dd += lxu;
                }
            }
        }
        /* restore all iw to -1   */
        //       uend = iU.begin()+lenu;
        iw[i] = -1;
        //       for(std::vector<int>::iterator j = iU.begin(); j != uend; ++j )
        //          iw[*j] = -1;
        lend = iL + lenl;
        for (std::vector<int>::iterator j = iL; j != lend; ++j)
            iw[*j] = -1;

        /*-------------------- update U-matrix      */
        (*U_).initRow(lenu, iU, wU);

        /*----- update diagonal with modification ------ */
        if (std::abs(dd) < MAT_TOL) dd = 1.0e-6;
        T dval = 1.0 / dd;
        D_.push_back(dval);
        /* perform scaling of L-part of column and update L matrix */
        Tscal(lenl, dval, &(*wL));
        (*L_).initRow(lenl, iL, wL);
    }

    return (0);
}
template <class T>
int PreconILU<T>::milut(LinearSolverMatrix<lsdatatype>& csmat_)
{
    /*
     * This is a modified ILUT routine based on the new column ILUT version
     * ILUT3
     *
     * It computes the diagonally modified ILUT factorization, so that Mv := Av
     * where v = vector of ones.
     *
     * D.O-K. Aug. 2010 -- Jan. 2011 -- May 2012
     */
    /*-------------------- The new MILUT in C
     *------------------------------------- COLUMN-BASED MILUT (MILUTC)
     *preconditioner Modified Incomplete LU factorization with dual truncation
     *mechanism NOTE : no pivoting implemented as yet in GE for diagonal
     *elements
     *----------------------------------------------------------------------------
     * Notes:
     * ======
     * All the diagonals of the input matrix must not be zero
     *--------------------------------------------------------------------------*/

    /*-------------------- local work arrays */
    std::vector<int> iw(n_, -1);
    std::vector<int> iU;
    iU.resize(n_);
    std::vector<T> dnormL(n_);
    std::vector<T> wU(n_);

    std::vector<int>::const_iterator start, end, lstart, lend, row;
    std::vector<int>::iterator ustart, uend, jpiv;
    TvecIterator ldstart, udstart, ldend, ldptr, udptr;
    LSvecIterator dstart, dend, dptr;

    /*-------------------- beginning of main loop */
    /*-----------------------------------------------------------------------*/
    for (int i = 0; i < n_; i++)
    {
        T tnorm = 0.0;
        int k1  = csmat_.nzptrval(i);
        int k2  = csmat_.nzptrval((i + 1));
        dstart  = csmat_.getDataIterator(k1);
        dend    = csmat_.getDataIterator(k2);
        for (dptr = dstart; dptr != dend; ++dptr)
            tnorm += (T)std::abs(*dptr);
        if (tnorm == 0.0)
        {
            printf("Zero row encountered in MILUT - row %d \n", i);
            mgmol_abort();
        }
        T tnorm0 = tnorm;
        tnorm /= (T)(k2 - k1 + 1);
        /*-------------------- unpack L & U-parts of column of A */
        std::vector<int>::iterator iL = iU.begin() + i;
        TvecIterator wL               = wU.begin() + i;
        T dd     = 1.0e-6; // MAT_TOL; /* initialize diagonal entry */
        int lenl = 0;
        int lenu = 0;
        iw[i]    = i;

        /*-------------------- scan & unwrap column */
        start = csmat_.getColumnIterator(k1);
        end   = csmat_.getColumnIterator(k2);
        for (row = start, dptr = dstart; row != end; ++row, ++dptr)
        {
            T t = (T)*dptr;
            if (*row < i)
            {
                iU[lenu] = *row;
                wU[lenu] = t;
                iw[*row] = lenu++;
            }
            else if (*row > i)
            {
                *(iL + lenl) = *row;
                *(wL + lenl) = t;
                iw[*row]     = lenl++;
            }
            else
                dd = t;
        }
        T dsum       = 0.0;
        int dropctrU = 0;
        int dropctrL = 0;
        T dd0        = std::abs(dd);
        T ddr        = dd0 / tnorm0;
        /*-------------------- eliminate column */
        ustart  = iU.begin();
        uend    = ustart + lenu;
        udstart = wU.begin();
        for (jpiv = ustart, udptr = udstart; jpiv != uend; ++jpiv, ++udptr)
        {
            /*-------------------------------------------------------------------------
             *  in order to do the elimination in the correct order we must
             *select the smallest row index among iU[k], k = j, j+1, ..., lenu-1
             *-----------------------------------------------------------------------*/
            std::vector<int>::iterator jmin = min_element(jpiv, uend);
            /* now swap current column with smallest column index */
            if (jmin != jpiv)
            {
                std::swap(*jpiv, *jmin); // swap in iU
                std::swap(iw[*jpiv], iw[*jmin]); // swap in iw
                std::swap(*udptr, wU[(jmin - iU.begin())]); // swap in wU
            }
            /* zero out element - reset pivot */
            iw[*jpiv] = -1;
            /* dropping in U */
            if (std::abs(*udptr) <= dnormL[*jpiv] * droptol_)
            {
                dsum += (*udptr);
                *udptr = 0.0;
                dropctrU++;
                continue;
            }
            /* combine current column and pivot column */
            int l_k1 = (*L_).nzptrval(*jpiv);
            int l_k2 = (*L_).nzptrval((*jpiv + 1));
            lstart   = (*L_).getColumnIterator(l_k1);
            lend     = (*L_).getColumnIterator(l_k2);
            ldstart  = (*L_).getDataIterator(l_k1);
            for (row = lstart, ldptr = ldstart; row != lend; ++row, ++ldptr)
            {
                int ipos = iw[*row];
                T lxu    = -((*ldptr) * (*udptr));

                if (*row < i)
                {
                    /* dealing with upper part */
                    if (ipos == -1)
                    {
                        /* this is a fill-in element */
                        iU[lenu] = *row;
                        wU[lenu] = lxu;
                        iw[*row] = lenu++;
                        /* Fill-in in U-part:- update end of iterator jpiv */
                    }
                    else
                    {
                        wU[ipos] += lxu;
                    }
                }
                else if (*row > i)
                {
                    /* dealing with lower part */
                    if (ipos == -1)
                    {
                        /* this is a fill-in element */
                        *(iL + lenl) = *row;
                        *(wL + lenl) = lxu;
                        iw[*row]     = lenl++;
                    }
                    else
                    {
                        *(wL + ipos) += lxu;
                    }
                }
                else
                {
                    dd += lxu;
                }
            }
            /* update end of iterator jpiv */
            uend = ustart + lenu;
        }
        /* restore all iw to -1   */
        //       uend = iU.begin()+lenu;
        iw[i] = -1;
        //       for(std::vector<int>::iterator j = iU.begin(); j != uend; ++j )
        //          iw[*j] = -1;
        lend = iL + lenl;
        for (std::vector<int>::iterator j = iL; j != lend; ++j)
            iw[*j] = -1;

        /*-------------------- update U-matrix      */
        /*-------------------- first prune col. of U to remove zeros ---*/
        int len = std::min(lenu, (lenu - dropctrU));
        qsplitC(wU, iU, lenu, len);
        (*U_).initRow(len, iU, wU);

        /*-------------------- update l-matrix */
        /*-------------------- first prune col. of L to remove small terms */
        for (row = iL, ldptr = wL; row != lend; ++row, ++ldptr)
        {
            if (std::abs(*ldptr) < droptol_)
            {
                dsum += (*ldptr);
                *ldptr = 0.0;
                dropctrL++;
            }
        }

        len = std::min((lenl - dropctrL), MaxFil_);
        /*-------------------- scale column of L by t - then copy all*/
        qsplitC(&(*wL), &(*iL), lenl, len);
        /*---------- update dsum -----*/
        ldstart = wL + len;
        ldend   = wL + lenl;
        for (ldptr = ldstart; ldptr != ldend; ++ldptr)
            dsum += (*ldptr);

        /*-------------------- update diagonal ------*/
        dsum += MAT_TOL;

        /* ------ select diagonal modification using optimal spreading ----*/
        T alpha        = 0.0;
        T beta         = 0.0;
        const T gamma  = std::abs(dsum);
        const short sr = dsum < 0.0 ? -1 : 1;
        const short sd = dd < 0.0 ? -1 : 1;

        tnorm = Tnrm2(len, &(*wL));

        if (tnorm)
        {
            /* relaxed compensation with optimal spreading */
            T tval = (gamma * gamma) - (tnorm * tnorm);
            if (tval > 0.0)
            {
                if (sr * sd == 1)
                {
                    alpha
                        = sr
                          * (std::sqrt(tval) + ddr * (gamma - std::sqrt(tval)));
                }
                else
                {
                    alpha = sr
                            * (std::sqrt(tval)
                                + droptol_ * (gamma - std::sqrt(tval)));
                }
            }
            else
            {
                if (sr * sd == 1)
                {
                    alpha = sr * gamma * ddr;
                }
                else
                {
                    alpha = sr * gamma * droptol_;
                }
            }
            T bf = 1.0;
            T mu = (-1 + std::sqrt((tnorm) / (gamma * gamma - alpha * alpha)))
                   / std::pow(std::abs(dd + alpha), 2.0);
            beta = bf * (-1 / (1 + (mu * std::pow(std::abs(dd + alpha), 2.0))));
        }
        /*----- update diagonal with modification ------ */
        T dval = 1.0 / (dd + alpha);
        D_.push_back(dval);

        /*---- done with diagonal modification. Now update for lower part of
         * column ----*/
        /*----------- Update lower part of column before scaling ----*/
        dnormL[i] = 1.0;
        ldend     = wL + len;
        for (ldptr = wL; ldptr != ldend; ++ldptr)
        {
            (*ldptr) *= (1 + 1.0 * beta);
            dnormL[i] += std::abs((*ldptr));
        }
        /* perform scaling of L-part of column and update L matrix */
        const T t = D_[i];

        Tscal(len, t, &(*wL));
        (*L_).initRow(len, iL, wL);

        /*--- save the norm of the ith column of L ---*/
        dnormL[i] = dnormL[i] / (1 + len);
    }

    return (0);
}
template <class T>
int PreconILU<T>::ilut(LinearSolverMatrix<lsdatatype>& csmat_)
{
    /*
     * This is a standard ILUT routine based on the new column ILUT version
     * ILUT3
     *
     *
     */
    /*-------------------- The new MILUT in C
     *------------------------------------- COLUMN-BASED ILUT (ILUTC)
     *preconditioner Incomplete LU factorization with dual truncation mechanism
     * NOTE : no pivoting implemented as yet in GE for diagonal elements
     *----------------------------------------------------------------------------
     * Notes:
     * ======
     * All the diagonals of the input matrix must not be zero
     *--------------------------------------------------------------------------*/

    /*-------------------- local work arrays */
    std::vector<int> iw(n_, -1);
    std::vector<int> iU;
    iU.resize(n_);
    std::vector<T> dnormL(n_);
    std::vector<T> wU(n_);

    std::vector<int>::const_iterator start, end, lstart, lend, row;
    std::vector<int>::iterator ustart, uend, jpiv;
    TvecIterator ldstart, udstart, ldptr, udptr;
    LSvecIterator dstart, dend, dptr;

    /*-------------------- beginning of main loop */
    /*-----------------------------------------------------------------------*/
    for (int i = 0; i < n_; i++)
    {
        T tnorm = 0.0;
        int k1  = csmat_.nzptrval(i);
        int k2  = csmat_.nzptrval((i + 1));
        dstart  = csmat_.getDataIterator(k1);
        dend    = csmat_.getDataIterator(k2);
        for (dptr = dstart; dptr != dend; ++dptr)
            tnorm += (T)std::abs(*dptr);
        if (tnorm == 0.0)
        {
            printf("Zero row encountered in MILUT - row %d \n", i);
            mgmol_abort();
        }
        tnorm /= (T)(k2 - k1 + 1);
        T tolnorm = droptol_ * tnorm;
        /*-------------------- unpack L & U-parts of column of A */
        std::vector<int>::iterator iL = iU.begin() + i;
        TvecIterator wL               = wU.begin() + i;
        T dd     = 1.0e-6; // MAT_TOL; /* initialize diagonal entry */
        int lenl = 0;
        int lenu = 0;
        iw[i]    = i;

        /*-------------------- scan & unwrap column */
        start = csmat_.getColumnIterator(k1);
        end   = csmat_.getColumnIterator(k2);
        for (row = start, dptr = dstart; row != end; ++row, ++dptr)
        {
            T t = (T)*dptr;
            if (*row < i)
            {
                iU[lenu] = *row;
                wU[lenu] = t;
                iw[*row] = lenu++;
            }
            else if (*row > i)
            {
                *(iL + lenl) = *row;
                *(wL + lenl) = t;
                iw[*row]     = lenl++;
            }
            else
                dd = t;
        }
        /*-------------------- eliminate column */
        ustart  = iU.begin();
        uend    = ustart + lenu;
        udstart = wU.begin();
        for (jpiv = ustart, udptr = udstart; jpiv != uend; ++jpiv, ++udptr)
        {
            /*-------------------------------------------------------------------------
             *  in order to do the elimination in the correct order we must
             *select the smallest row index among iU[k], k = j, j+1, ..., lenu-1
             *-----------------------------------------------------------------------*/
            std::vector<int>::iterator jmin = min_element(jpiv, uend);
            /* now swap current column with smallest column index */
            if (jmin != jpiv)
            {
                std::swap(*jpiv, *jmin); // swap in iU
                std::swap(iw[*jpiv], iw[*jmin]); // swap in iw
                std::swap(*udptr, wU[(jmin - iU.begin())]); // swap in wU
            }
            /* zero out element - reset pivot */
            iw[*jpiv] = -1;
            /* combine current column and pivot column */
            int l_k1 = (*L_).nzptrval(*jpiv);
            int l_k2 = (*L_).nzptrval((*jpiv + 1));
            lstart   = (*L_).getColumnIterator(l_k1);
            lend     = (*L_).getColumnIterator(l_k2);
            ldstart  = (*L_).getDataIterator(l_k1);
            for (row = lstart, ldptr = ldstart; row != lend; ++row, ++ldptr)
            {
                int ipos = iw[*row];
                T lxu    = -((*ldptr) * (*udptr));
                /* drop small fill-in elements */
                if (std::abs(lxu) < tolnorm && ipos == -1) continue;

                if (*row < i)
                {
                    /* dealing with upper part */
                    if (ipos == -1)
                    {
                        /* this is a fill-in element */
                        iU[lenu] = *row;
                        wU[lenu] = lxu;
                        iw[*row] = lenu++;
                    }
                    else
                    {
                        wU[ipos] += lxu;
                    }
                }
                else if (*row > i)
                {
                    /* dealing with lower part */
                    if (ipos == -1)
                    {
                        /* this is a fill-in element */
                        *(iL + lenl) = *row;
                        *(wL + lenl) = lxu;
                        iw[*row]     = lenl++;
                    }
                    else
                    {
                        *(wL + ipos) += lxu;
                    }
                }
                else
                {
                    dd += lxu;
                }
            }
            /* update end of iterator jpiv */
            uend = ustart + lenu;
        }
        /* restore all iw to -1   */
        //       uend = iU.begin()+lenu;
        iw[i] = -1;
        //       for(std::vector<int>::iterator j = iU.begin(); j != uend; ++j )
        //          iw[*j] = -1;
        lend = iL + lenl;
        for (std::vector<int>::iterator j = iL; j != lend; ++j)
            iw[*j] = -1;

        /*-------------------- update U-matrix      */
        /*-------------------- first prune col. of U to remove zeros ---*/
        int len = std::min(lenu, MaxFil_);
        qsplitC(wU, iU, lenu, len);
        (*U_).initRow(len, iU, wU);

        /*-------------------- update l-matrix */
        len = std::min(lenl, MaxFil_);
        /*-------------------- scale column of L by t - then copy all*/
        qsplitC(&(*wL), &(*iL), lenl, len);

        /*----- update diagonal with modification ------ */
        if (std::abs(dd) < MAT_TOL) dd = tolnorm * 1.0e-6;
        T dval = 1.0 / dd;
        D_.push_back(dval);

        /* perform scaling of L-part of column and update L matrix */
        Tscal(len, dval, &(*wL));
        (*L_).initRow(len, iL, wL);
    }

    return (0);
}
template <class T>
int PreconILU<T>::diag(LinearSolverMatrix<lsdatatype>& csmat)
{
    for (int i = 0; i < n_; i++)
    {
        T tnorm = 1.0e-6; // MAT_TOL;
        int k1  = csmat.nzptrval(i);
        int k2  = csmat.nzptrval((i + 1));
        for (int j = k1; j < k2; j++)
        {
            T val = (T)csmat.getRowEntry(j);
            tnorm += std::abs(val);
        }
        T dval = 1.0 / tnorm;
        D_.push_back(dval);
    }

    return (0);
}
template <class T>
void PreconILU<T>::printTimers(std::ostream& os)
{
    pcilu_setup_tm_.print(os);
    pcilu_solve_tm_.print(os);
}

template class PreconILU<float>;
template class PreconILU<double>;

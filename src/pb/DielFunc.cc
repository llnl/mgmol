// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

// $Id: DielFunc.cc,v 1.13 2009/07/13 17:37:06 jeanluc Exp $
#include "DielFunc.h"
#include "Grid.h"
#include "MGmol_abort.h"

namespace pb
{

template <class T>
DielFunc<T>& DielFunc<T>::operator=(const T val)
{
    const int N = GridFunc<T>::grid_.sizeg();

    for (int i = 0; i < N; i++)
        GridFunc<T>::uu_[i] = val;

    GridFunc<T>::updated_boundaries_ = 1;
    return *this;
}
template <class T>
double DielFunc<T>::epsilon_rho(
    const T rho, const T e0, const T rho0, const T drho0)
{
    assert(e0 >= 1.);

    const double ratio = pow(rho / rho0, 2. * drho0);
    return (1. + 0.5 * (e0 - 1.) * ((1. - ratio) / (1. + ratio) + 1.));
}
template <class T>
double DielFunc<T>::epsilon_rho(const T rho)
{
    const double ratio = pow(rho / rho0_, 2. * drho0_);
    return (
        1. + 0.5 * (epsilon_max_ - 1.) * ((1. - ratio) / (1. + ratio) + 1.));
}
template <class T>
double DielFunc<T>::depsilon_rho(
    const T rho, const T e0, const T rho0, const T drho0)
{
    if (rho < 0.001 * rho0) return 0.;

    const double inv_rho0 = 1. / rho0;
    const double ratio    = rho * inv_rho0;
    const double pratio   = pow(ratio, 2. * drho0);
    return (2. * inv_rho0) * (1. - e0) * drho0 * pow(ratio, 2. * drho0 - 1.)
           / ((1. + pratio) * (1. + pratio));
}
template <class T>
double DielFunc<T>::depsilon_rho(const T rho)
{
    if (rho < 0.001 * rho0_) return 0.;

    const double inv_rho0 = 1. / rho0_;
    const double ratio    = rho * inv_rho0;
    const double pratio   = pow(ratio, 2. * drho0_);
    return (2. * inv_rho0) * (1. - epsilon_max_) * drho0_
           * pow(ratio, 2. * drho0_ - 1.) / ((1. + pratio) * (1. + pratio));
}
template <class T>
void DielFunc<T>::Gepsilon_rho(GridFunc<T>& rho)
{
    assert(GridFunc<T>::grid_.dim(0) > 1);

    //    GridFunc<T>::tag_ = rho.tag();
    GridFunc<T>::bc_[0]     = 1;
    GridFunc<T>::bc_[1]     = 1;
    GridFunc<T>::bc_[2]     = 1;
    const T* const rho_data = rho.uu();
    const int shift1        = GridFunc<T>::grid().ghost_pt();
    const int shift2        = rho.grid().ghost_pt();

    const int incx1 = GridFunc<T>::grid().inc(0);
    const int incx2 = rho.grid().inc(0);
    const int incy1 = GridFunc<T>::grid().inc(1);
    const int incy2 = rho.grid().inc(1);

    assert(GridFunc<T>::grid().inc(2) == 1);
    assert(rho.grid().inc(2) == 1);

    if (rho.uu() != nullptr)
    {

        const int dim0 = GridFunc<T>::dim(0);
        const int dim1 = GridFunc<T>::dim(1);
        const int dim2 = GridFunc<T>::dim(2);
        for (int ix = 0; ix < dim0; ix++)
        {

            int ix1 = (ix + shift1) * incx1 + shift1;
            int ix2 = (ix + shift2) * incx2 + shift2;

            for (int iy = 0; iy < dim1; iy++)
            {

                int iy1 = ix1 + (iy + shift1) * incy1;
                int iy2 = ix2 + (iy + shift2) * incy2;

                for (int iz = 0; iz < dim2; iz++)
                {

                    int iz1 = iy1 + iz;
                    int iz2 = iy2 + iz;

                    GridFunc<T>::uu_[iz1] = epsilon_rho(rho_data[iz2]);

                    assert(GridFunc<T>::uu_[iz1] <= epsilon_max_);
                    assert(GridFunc<T>::uu_[iz1] >= 1.);
                }
            }
        }

        GridFunc<T>::updated_boundaries_ = 0;
    }
    else
    {
        mgmol_abort(" Need a density to build dielectric function...");
    }
}
template <class T>
void DielFunc<T>::Gepsilon_rho(GridFunc<T>& rho, const T rho0, const T drho0)
{
    assert(GridFunc<T>::grid_.dim(0) > 1);

    //    GridFunc<T>::tag_ = rho.tag();
    GridFunc<T>::bc_[0]     = 1;
    GridFunc<T>::bc_[1]     = 1;
    GridFunc<T>::bc_[2]     = 1;
    const T* const rho_data = rho.uu();
    const int shift1        = GridFunc<T>::grid().ghost_pt();
    const int shift2        = rho.grid().ghost_pt();

    const int incx1 = GridFunc<T>::grid().inc(0);
    const int incx2 = rho.grid().inc(0);
    const int incy1 = GridFunc<T>::grid().inc(1);
    const int incy2 = rho.grid().inc(1);

    assert(GridFunc<T>::grid().inc(2) == 1);
    assert(rho.grid().inc(2) == 1);

    if (rho.uu() != nullptr)
    {

        const int dim0 = GridFunc<T>::dim(0);
        const int dim1 = GridFunc<T>::dim(1);
        const int dim2 = GridFunc<T>::dim(2);
        for (int ix = 0; ix < dim0; ix++)
        {

            int ix1 = (ix + shift1) * incx1 + shift1;
            int ix2 = (ix + shift2) * incx2 + shift2;

            for (int iy = 0; iy < dim1; iy++)
            {

                int iy1 = ix1 + (iy + shift1) * incy1;
                int iy2 = ix2 + (iy + shift2) * incy2;

                for (int iz = 0; iz < dim2; iz++)
                {

                    int iz1 = iy1 + iz;
                    int iz2 = iy2 + iz;

                    GridFunc<T>::uu_[iz1]
                        = epsilon_rho(rho_data[iz2], epsilon_max_, rho0, drho0);

                    assert(GridFunc<T>::uu_[iz1] <= epsilon_max_);
                    assert(GridFunc<T>::uu_[iz1] >= 1.);
                }
            }
        }

        GridFunc<T>::updated_boundaries_ = 0;
    }
    else
    {
        mgmol_abort(" Need a density to build dielectric function...");
    }
}
template <class T>
void DielFunc<T>::Gdepsilon_rho(
    GridFunc<T>& rho, GridFunc<T>& depsilon, const T rho0, const T drho0)
{
    assert(GridFunc<T>::grid_.dim(0) > 1);

    //    GridFunc<T>::tag_ = rho.tag();
    GridFunc<T>::bc_[0]     = 1;
    GridFunc<T>::bc_[1]     = 1;
    GridFunc<T>::bc_[2]     = 1;
    const T* const rho_data = rho.uu(0);
    T* de                   = depsilon.uu();
    const int shift1        = GridFunc<T>::grid().ghost_pt();
    const int shift2        = rho.grid().ghost_pt();

    const int incx1 = GridFunc<T>::grid().inc(0);
    const int incx2 = rho.grid().inc(0);
    const int incy1 = GridFunc<T>::grid().inc(1);
    const int incy2 = rho.grid().inc(1);

    assert(GridFunc<T>::grid().inc(2) == 1);
    assert(rho.grid().inc(2) == 1);

    if (rho.uu() != nullptr)
    {

        const int dim0 = GridFunc<T>::dim(0);
        const int dim1 = GridFunc<T>::dim(1);
        const int dim2 = GridFunc<T>::dim(2);
        for (int ix = 0; ix < dim0; ix++)
        {

            int ix1 = (ix + shift1) * incx1 + shift1;
            int ix2 = (ix + shift2) * incx2 + shift2;

            for (int iy = 0; iy < dim1; iy++)
            {

                int iy1 = ix1 + (iy + shift1) * incy1;
                int iy2 = ix2 + (iy + shift2) * incy2;

                for (int iz = 0; iz < dim2; iz++)
                {

                    int iz1 = iy1 + iz;
                    int iz2 = iy2 + iz;

                    de[iz1] = depsilon_rho(
                        rho_data[iz2], epsilon_max_, rho0, drho0);

                    assert(GridFunc<T>::uu_[iz1] <= epsilon_max_);
                    assert(GridFunc<T>::uu_[iz1] >= 1.);
                }
            }
        }

        GridFunc<T>::updated_boundaries_ = 0;
    }
    else
    {
        mgmol_abort(" Need a density to build dielectric function...");
    }
}

template <class T>
void DielFunc<T>::Gdepsilon_rho(GridFunc<T>& rho, GridFunc<T>& depsilon)
{
    assert(GridFunc<T>::grid_.dim(0) > 1);

    //    GridFunc<T>::tag_ = rho.tag();
    GridFunc<T>::bc_[0]     = 1;
    GridFunc<T>::bc_[1]     = 1;
    GridFunc<T>::bc_[2]     = 1;
    const T* const rho_data = rho.uu(0);
    T* de                   = depsilon.uu();
    const int shift1        = GridFunc<T>::grid().ghost_pt();
    const int shift2        = rho.grid().ghost_pt();

    const int incx1 = GridFunc<T>::grid().inc(0);
    const int incx2 = rho.grid().inc(0);
    const int incy1 = GridFunc<T>::grid().inc(1);
    const int incy2 = rho.grid().inc(1);

    assert(GridFunc<T>::grid().inc(2) == 1);
    assert(rho.grid().inc(2) == 1);

    if (rho.uu() != nullptr)
    {

        const int dim0 = GridFunc<T>::dim(0);
        const int dim1 = GridFunc<T>::dim(1);
        const int dim2 = GridFunc<T>::dim(2);
        for (int ix = 0; ix < dim0; ix++)
        {

            int ix1 = (ix + shift1) * incx1 + shift1;
            int ix2 = (ix + shift2) * incx2 + shift2;

            for (int iy = 0; iy < dim1; iy++)
            {

                int iy1 = ix1 + (iy + shift1) * incy1;
                int iy2 = ix2 + (iy + shift2) * incy2;

                for (int iz = 0; iz < dim2; iz++)
                {

                    int iz1 = iy1 + iz;
                    int iz2 = iy2 + iz;

                    de[iz1] = depsilon_rho(rho_data[iz2]);

                    assert(GridFunc<T>::uu_[iz1] <= epsilon_max_);
                    assert(GridFunc<T>::uu_[iz1] >= 1.);
                }
            }
        }

        GridFunc<T>::updated_boundaries_ = 0;
    }
    else
    {
        mgmol_abort(" Need a density to build dielectric function...");
    }
}
template class DielFunc<double>;
template class DielFunc<float>;
} // namespace pb

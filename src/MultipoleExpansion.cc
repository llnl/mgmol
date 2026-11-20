// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

// Convention: a positive charge generates a positive potential

#include "MultipoleExpansion.h"
#include "MGmol_MPI.h"

#include "MGmol_abort.h"
#include <iomanip>

using namespace std;

// template <typename T>
MultipoleExpansion::MultipoleExpansion(const pb::Grid& mygrid,
    const short bc[3], const Vector3D& origin_cell, const Vector3D& cell)
    : grid_(mygrid), origin_cell_(origin_cell), cell_(cell)
{
    space_dim_.clear();
    for (short i = 0; i < 3; i++)
    {
        origin_[i] = origin_cell_[i] + 0.5 * cell_[i];
        bc_[i]     = bc[i];
        if (bc[i] == 2) space_dim_.push_back(i);
    }

    order_ = 1; // default

    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    onpe0_          = mmpi.instancePE0();
}
// template <typename T>
void MultipoleExpansion::get_monopole(RHODTYPE* rho)
{
    const int dim0 = grid_.dim(0);
    const int dim1 = grid_.dim(1);
    const int dim2 = grid_.dim(2);

    const int incx = dim1 * dim2;
    const int incy = dim2;

    qtotal_ = 0.;

    for (int ix = 0; ix < dim0; ix++)
    {
        for (int iy = 0; iy < dim1; iy++)
        {
            for (int iz = 0; iz < dim2; iz++)
            {
                qtotal_ += (double)rho[ix * incx + iy + incy + iz];
            }
        }
    }

    const double h0 = grid_.hgrid(0);
    const double h1 = grid_.hgrid(1);
    const double h2 = grid_.hgrid(2);

    qtotal_ *= (h0 * h1 * h2);
    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    double tmp;
    mmpi.allreduce(&qtotal_, &tmp, 1, MPI_SUM);
    qtotal_ = tmp;
    if (onpe0_)
        (*MPIdata::sout) << scientific
                         << "MultipoleExpansion::get_monopole(double*), Charge="
                         << qtotal_ << endl;
}
// template <typename T>
void MultipoleExpansion::get_monopole(const pb::GridFunc<RHODTYPE>& rho)
{
    qtotal_ = rho.integral();

    if (onpe0_)
        (*MPIdata::sout) << scientific
                         << "MultipoleExpansion::get_monopole(), Charge="
                         << qtotal_ << endl;
}

// template <typename T>
void MultipoleExpansion::get_dipole(RHODTYPE* rho)
{
    const double start0 = grid_.start(0);
    const double start1 = grid_.start(1);
    const double start2 = grid_.start(2);

    const double h0 = grid_.hgrid(0);
    const double h1 = grid_.hgrid(1);
    const double h2 = grid_.hgrid(2);

    const int dim0 = grid_.dim(0);
    const int dim1 = grid_.dim(1);
    const int dim2 = grid_.dim(2);

    const int incx = dim1 * dim2;
    const int incy = dim2;

    Vector3D gpoint(start0, start1, start2);
    double dipole[3] = { 0., 0., 0. };

    for (int ix = 0; ix < dim0; ix++)
    {
        gpoint[0] = start0 + h0 * ix - origin_[0];
        for (int iy = 0; iy < dim1; iy++)
        {
            gpoint[1] = start1 + h1 * iy - origin_[1];
            for (int iz = 0; iz < dim2; iz++)
            {
                const double valrho = (double)rho[ix * incx + iy * incy + iz];

                dipole[0] += gpoint[0] * valrho;
                dipole[1] += gpoint[1] * valrho;

                gpoint[2] = start2 + h2 * iz - origin_[2];
                dipole[2] += gpoint[2] * valrho;
            }
        }
    }

    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    mmpi.allreduce(&dipole[0], &dipole_moment_[0], 3, MPI_SUM);
    for (short d = 0; d < 3; d++)
        dipole_moment_[d] *= (h0 * h1 * h2);

    if (onpe0_)
        (*MPIdata::sout) << "MultipoleExpansion::get_dipole(), Total Dipole=("
                         << dipole_moment_[0] << "," << dipole_moment_[1] << ","
                         << dipole_moment_[2] << ")" << endl;
}

// template <typename T>
void MultipoleExpansion::get_dipole(const pb::GridFunc<RHODTYPE>& rho)
{
    const bool mympi0 = (grid_.mype_env().my_mpi(0) == 0 && bc_[0] != 1);
    const bool mympi1 = (grid_.mype_env().my_mpi(1) == 0 && bc_[1] != 1);
    const bool mympi2 = (grid_.mype_env().my_mpi(2) == 0 && bc_[2] != 1);

    const short nghosts = rho.ghost_pt();
    assert(nghosts == grid_.ghost_pt());

    // increase ghost layer to avoid counting boundary
    const int istart[3] = { mympi0 ? nghosts + 1 : nghosts,
        mympi1 ? nghosts + 1 : nghosts, mympi2 ? nghosts + 1 : nghosts };

    const pb::Grid& grid = rho.grid();

    const double hspacing[3] = { grid.hgrid(0), grid.hgrid(1), grid.hgrid(2) };

    const double start[3] = { grid.start(0) - nghosts * hspacing[0],
        grid.start(1) - nghosts * hspacing[1],
        grid.start(2) - nghosts * hspacing[2] };

    const int incx        = grid.inc(0);
    const int incy        = grid.inc(1);
    const unsigned dim[3] = { grid.dim(0), grid.dim(1), grid.dim(2) };

    Vector3D gpoint(start[0], start[1], start[2]);
    double dipole[3] = { 0., 0., 0. };

    for (int ix = istart[0]; ix < static_cast<int>(nghosts + dim[0]); ix++)
    {

        const int iix = ix * incx;
        gpoint[0]     = start[0] + hspacing[0] * ix - origin_[0];

        for (int iy = istart[1]; iy < static_cast<int>(nghosts + dim[1]); iy++)
        {

            const int iiy = iy * incy + iix;
            gpoint[1]     = start[1] + hspacing[1] * iy - origin_[1];

            const RHODTYPE* const prho = rho.uu(iiy);
            for (int iz = istart[2]; iz < static_cast<int>(nghosts + dim[2]);
                 iz++)
            {
                gpoint[2] = start[2] + hspacing[2] * iz - origin_[2];

                const double valrho = (double)prho[iz];
                dipole[0] += gpoint[0] * valrho;
                dipole[1] += gpoint[1] * valrho;
                dipole[2] += gpoint[2] * valrho;
            }
        }
    }

    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    mmpi.allreduce(&dipole[0], &dipole_moment_[0], 3, MPI_SUM);
    const double vel = (hspacing[0] * hspacing[1] * hspacing[2]);
    for (short d = 0; d < 3; d++)
        dipole_moment_[d] *= vel;

    if (onpe0_)
        (*MPIdata::sout) << "MultipoleExpansion::get_dipole(), Dipole=("
                         << setprecision(5) << scientific << dipole_moment_[0]
                         << "," << dipole_moment_[1] << "," << dipole_moment_[2]
                         << ")" << endl;
}
// template <typename T>
void MultipoleExpansion::computeQuadrupole(const pb::GridFunc<RHODTYPE>& rho)
{
    const bool mympi0 = (grid_.mype_env().my_mpi(0) == 0 && bc_[0] != 1);
    const bool mympi1 = (grid_.mype_env().my_mpi(1) == 0 && bc_[1] != 1);
    const bool mympi2 = (grid_.mype_env().my_mpi(2) == 0 && bc_[2] != 1);

    const short nghosts = rho.ghost_pt();
    assert(nghosts == grid_.ghost_pt());

    // increase ghost layer to avoid counting boundary
    const int istart[3] = { mympi0 ? nghosts + 1 : nghosts,
        mympi1 ? nghosts + 1 : nghosts, mympi2 ? nghosts + 1 : nghosts };

    const pb::Grid& grid = rho.grid();

    const double hspacing[3] = { grid.hgrid(0), grid.hgrid(1), grid.hgrid(2) };

    const double start[3] = { grid.start(0) - nghosts * hspacing[0],
        grid.start(1) - nghosts * hspacing[1],
        grid.start(2) - nghosts * hspacing[2] };

    const int incx        = grid.inc(0);
    const int incy        = grid.inc(1);
    const unsigned dim[3] = { grid.dim(0), grid.dim(1), grid.dim(2) };

    Vector3D gpoint(start[0], start[1], start[2]);
    double quadrupole[6] = { 0., 0., 0., 0., 0., 0. };

    for (int ix = istart[0]; ix < static_cast<int>(nghosts + dim[0]); ix++)
    {

        const int iix = ix * incx;
        gpoint[0]     = start[0] + hspacing[0] * ix - origin_[0];

        for (int iy = istart[1]; iy < static_cast<int>(nghosts + dim[1]); iy++)
        {

            const int iiy = iy * incy + iix;
            gpoint[1]     = start[1] + hspacing[1] * iy - origin_[1];

            const RHODTYPE* const prho = rho.uu(iiy);
            for (int iz = istart[2]; iz < static_cast<int>(nghosts + dim[2]);
                 iz++)
            {
                gpoint[2] = start[2] + hspacing[2] * iz - origin_[2];
                const double ri2
                    = (gpoint[0] * gpoint[0] + gpoint[1] * gpoint[1]
                        + gpoint[2] * gpoint[2]);

                const double valrho = (double)prho[iz];
                quadrupole[0] += (2.0 * gpoint[0] * gpoint[0] - ri2) * valrho;
                quadrupole[1] += 2.0 * gpoint[0] * gpoint[1] * valrho;
                quadrupole[2] += 2.0 * gpoint[0] * gpoint[2] * valrho;
                quadrupole[3] += (2.0 * gpoint[1] * gpoint[1] - ri2) * valrho;
                quadrupole[4] += 2.0 * gpoint[1] * gpoint[2] * valrho;
                quadrupole[5] += (2.0 * gpoint[2] * gpoint[2] - ri2) * valrho;
            }
        }
    }

    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    mmpi.allreduce(&quadrupole[0], &quadrupole_moment_[0], 6, MPI_SUM);
    const double vel = (hspacing[0] * hspacing[1] * hspacing[2]);
    for (short d = 0; d < 6; d++)
        quadrupole_moment_[d] *= vel;

    if (onpe0_)
        (*MPIdata::sout)
            << "MultipoleExpansion::computeQuadrupole(), quadrupole=("
            << setprecision(5) << scientific << quadrupole_moment_[0] << ","
            << quadrupole_moment_[1] << "," << quadrupole_moment_[2] << ","
            << quadrupole_moment_[3] << "," << quadrupole_moment_[4] << ","
            << quadrupole_moment_[5] << ")" << endl;
}
// template <typename T>
void MultipoleExpansion::resetOriginToChargeCenter(
    const pb::GridFunc<RHODTYPE>& rho)
{
    const bool mympi0 = (grid_.mype_env().my_mpi(0) == 0 && bc_[0] != 1);
    const bool mympi1 = (grid_.mype_env().my_mpi(1) == 0 && bc_[1] != 1);
    const bool mympi2 = (grid_.mype_env().my_mpi(2) == 0 && bc_[2] != 1);

    const short nghosts  = rho.ghost_pt();
    const pb::Grid& grid = rho.grid();

    const double hspacing[3] = { grid.hgrid(0), grid.hgrid(1), grid.hgrid(2) };

    const double start[3] = { grid.start(0) - nghosts * hspacing[0],
        grid.start(1) - nghosts * hspacing[1],
        grid.start(2) - nghosts * hspacing[2] };

    // increase ghost layer to avoid counting boundary if non-periodic
    const int istart[3] = { mympi0 ? nghosts + 1 : nghosts,
        mympi1 ? nghosts + 1 : nghosts, mympi2 ? nghosts + 1 : nghosts };

    const int incx        = grid.inc(0);
    const int incy        = grid.inc(1);
    const unsigned dim[3] = { grid.dim(0), grid.dim(1), grid.dim(2) };

    Vector3D gpoint(start[0], start[1], start[2]);
    double charge_center[3] = { 0., 0., 0. };

    for (int ix = istart[0]; ix < static_cast<int>(nghosts + dim[0]); ix++)
    {

        const int iix = ix * incx;
        gpoint[0]     = start[0] + hspacing[0] * ix;

        for (int iy = istart[1]; iy < static_cast<int>(nghosts + dim[1]); iy++)
        {

            const int iiy = iy * incy + iix;
            gpoint[1]     = start[1] + hspacing[1] * iy;

            const RHODTYPE* const prho = rho.uu(iiy);
            for (int iz = istart[2]; iz < static_cast<int>(nghosts + dim[2]);
                 iz++)
            {

                gpoint[2] = start[2] + hspacing[2] * iz;

                const double valrho = (double)prho[iz];

                charge_center[0] += gpoint[0] * valrho;
                charge_center[1] += gpoint[1] * valrho;
                charge_center[2] += gpoint[2] * valrho;
            }
        }
    }

    for (short d = 0; d < 3; d++)
    {
        charge_center[d] *= grid.vel();
        charge_center[d] /= qtotal_;
    }

    // update origin_
    double tmp[3]   = { 0., 0., 0. };
    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    mmpi.allreduce(&charge_center[0], &tmp[0], 3, MPI_SUM);
    if (onpe0_)
        (*MPIdata::sout) << "MultipoleExpansion::resetOriginToChargeCenter(), "
                            "charge_center=("
                         << tmp[0] << "," << tmp[1] << "," << tmp[2] << ")"
                         << endl;
    for (short d = 0; d < 3; d++)
    {
        origin_[d] = tmp[d];
    }
    if (onpe0_)
        (*MPIdata::sout)
            << "MultipoleExpansion::resetOriginToChargeCenter(), new origin_=("
            << origin_[0] << "," << origin_[1] << "," << origin_[2] << ")"
            << endl;
}
// template <typename T>
void MultipoleExpansion::setup(RHODTYPE* rho)
{
    get_monopole(rho);

    if (!(fabs(qtotal_) > 0.1)) get_dipole(rho);
}
// template <typename T>
void MultipoleExpansion::setup(pb::GridFunc<RHODTYPE>& rho)
{
    get_monopole(rho);

    // if( fabs(qtotal_)>0.1 )
    //    resetOriginToChargeCenter(rho);

    if (order_ > 0) get_dipole(rho);

    if (order_ > 1) computeQuadrupole(rho);
}
template <typename T>
void MultipoleExpansion::expand(pb::GridFunc<T>& func)
{
    const short ndim = (short)space_dim_.size();

    if (ndim == 3)
    {
        expand3d(func);
    }
    else if (ndim == 2)
    {
        expand2d(func);
    }
    else
    {
        (*MPIdata::sout) << "MultipoleExpansion::expand() implemented only for "
                         << "2D and 3D cases" << endl;
    }
}

// uses only 2 non-periodic directions to generate multipole
template <typename T>
void MultipoleExpansion::expand2d(pb::GridFunc<T>& func)
{
    assert(space_dim_.size() == 2);

    const short index0 = space_dim_[0];
    const short index1 = space_dim_[1];

    short indexp;
    if (index0 == 1)
    {
        indexp = 0;
    }
    else
    { // index0==0
        if (index1 == 2)
        {
            indexp = 1;
        }
        else
        {
            indexp = 2;
        }
    }

    // dipole
    //(*MPIdata::sout)<<"dipole 0="<<dipole_moment_[index0]<<endl;
    //(*MPIdata::sout)<<"dipole 1="<<dipole_moment_[index1]<<endl;
    if (onpe0_)
    {
        (*MPIdata::sout) << "MultipoleExpansion::expand2d(), origin 0="
                         << origin_[index0] << ", origin 1=" << origin_[index1]
                         << ", order=" << order_ << endl;
    }

    const short nghosts  = func.ghost_pt();
    const pb::Grid& grid = func.grid();

    const double h0 = grid.hgrid(index0);
    const double h1 = grid.hgrid(index1);

    const double invlp
        = 2. / grid.ll(indexp); // factor 2 for Hartree units in 2D

    const double start0 = grid.start(index0) - nghosts * h0;
    const double start1 = grid.start(index1) - nghosts * h1;

    const int dim0g = grid.dim(index0) + 2 * nghosts;
    const int dim1g = grid.dim(index1) + 2 * nghosts;
    const int dimpg = grid.dim(indexp) + 2 * nghosts;

    const int inc0 = grid.inc(index0);
    const int inc1 = grid.inc(index1);
    const int incp = grid.inc(indexp);

    double gpoint[2] = { start0, start1 };

    T* const pfunc = func.uu();

    // monopole
    if (fabs(qtotal_) > 0.1)
    {
        const double tol     = 1.e-10;
        const double mlogtol = -log(tol);
        for (int i0 = 0; i0 < dim0g; i0++)
        {
            int is0         = i0 * inc0;
            gpoint[0]       = start0 + h0 * i0;
            const double d0 = gpoint[0] - origin_[index0];
            for (int i1 = 0; i1 < dim1g; i1++)
            {
                int is1         = is0 + i1 * inc1;
                gpoint[1]       = start1 + h1 * i1;
                const double d1 = gpoint[1] - origin_[index1];
                const double r  = sqrt(d0 * d0 + d1 * d1);
                double logr;
                if (r > tol)
                    logr = -1. * log(r);
                else
                    logr = mlogtol;

                for (int ip = 0; ip < dimpg; ip++)
                    pfunc[is1 + ip * incp] = (T)(qtotal_ * logr * invlp);
            }
        }
    }

    if (order_ >= 1)
    { // dipole
        const double tolr2    = 1.e-10;
        const double invtolr2 = 1. / tolr2;

        // loop over mesh points
        for (int i0 = 0; i0 < dim0g; i0++)
        {
            const int is0   = i0 * inc0;
            gpoint[0]       = start0 + h0 * i0;
            const double d0 = gpoint[0] - origin_[index0];
            for (int i1 = 0; i1 < dim1g; i1++)
            {
                const int is1   = is0 + i1 * inc1;
                gpoint[1]       = start1 + h1 * i1;
                const double d1 = gpoint[1] - origin_[index1];

                const double vr[2] = { d0, d1 };
                const double r2    = vr[0] * vr[0] + vr[1] * vr[1];
                const double rdotp = vr[0] * dipole_moment_[index0]
                                     + vr[1] * dipole_moment_[index1];
                double alpha;
                if (r2 > tolr2)
                    alpha = 1. / r2;
                else
                    alpha = invtolr2;
                for (int ip = 0; ip < dimpg; ip++)
                {
                    double addval = (double)pfunc[is1 + ip * incp]
                                    + rdotp * alpha * invlp;
                    pfunc[is1 + ip * incp] = (T)addval;
                }
            }
        }
    }

    if (order_ >= 2)
    { // quadrupole
        const double tolr2    = 1.e-5;
        const double invtolr4 = 1. / (tolr2 * tolr2);

        double qmoment[3];
        switch (indexp)
        {

            case 0:
                qmoment[0] = quadrupole_moment_[3];
                qmoment[1] = quadrupole_moment_[4];
                qmoment[2] = quadrupole_moment_[5];
                break;

            case 1:
                qmoment[0] = quadrupole_moment_[0];
                qmoment[1] = quadrupole_moment_[2];
                qmoment[2] = quadrupole_moment_[5];
                break;

            case 2:
                qmoment[0] = quadrupole_moment_[0];
                qmoment[1] = quadrupole_moment_[1];
                qmoment[2] = quadrupole_moment_[3];
                break;
            default:
                mgmol_abort("Invalid case for quadrupole moment");
        }

        // loop over mesh points
        for (int i0 = 0; i0 < dim0g; i0++)
        {
            const int is0   = i0 * inc0;
            gpoint[0]       = start0 + h0 * i0;
            const double d0 = gpoint[0] - origin_[index0];
            for (int i1 = 0; i1 < dim1g; i1++)
            {
                const int is1   = is0 + i1 * inc1;
                gpoint[1]       = start1 + h1 * i1;
                const double d1 = gpoint[1] - origin_[index1];

                const double vr[2] = { d0, d1 };
                const double r2    = vr[0] * vr[0] + vr[1] * vr[1];
                const double rdotp = vr[0] * vr[0] * qmoment[0]
                                     + vr[1] * vr[0] * qmoment[1]
                                     + vr[1] * vr[1] * qmoment[2];
                double alpha;
                if (r2 > tolr2)
                    alpha = 1.0 / (r2 * r2);
                else
                    alpha = 1.0 * invtolr4;
                for (int ip = 0; ip < dimpg; ip++)
                {
                    double addval = (double)pfunc[is1 + ip * incp]
                                    + rdotp * alpha * invlp;
                    pfunc[is1 + ip * incp] = (T)addval;
                }
            }
        }
    }
}
template <typename T>
void MultipoleExpansion::expand3d(pb::GridFunc<T>& func)
{
    if (onpe0_)
    {
        (*MPIdata::sout) << "MultipoleExpansion::expand3()" << endl;
    }
    const short nghosts  = func.ghost_pt();
    const pb::Grid& grid = func.grid();

    const double h0 = grid.hgrid(0);
    const double h1 = grid.hgrid(1);
    const double h2 = grid.hgrid(2);

    const double start0 = grid.start(0) - nghosts * h0;
    const double start1 = grid.start(1) - nghosts * h1;
    const double start2 = grid.start(2) - nghosts * h2;

    const int dim0g = grid.dim(0) + 2 * nghosts;
    const int dim1g = grid.dim(1) + 2 * nghosts;
    const int dim2g = grid.dim(2) + 2 * nghosts;

    const int incx = grid.inc(0);
    const int incy = grid.inc(1);

    Vector3D gpoint(start0, start1, start2);

    T* const pfunc = func.uu();

    if (fabs(qtotal_) > 0.1)
    {
        // monopole
        for (int ix = 0; ix < dim0g; ix++)
        {
            int ix1   = ix * incx;
            gpoint[0] = start0 + h0 * ix;
            for (int iy = 0; iy < dim1g; iy++)
            {
                int iy1   = ix1 + iy * incy;
                gpoint[1] = start1 + h1 * iy;
                for (int iz = 0; iz < dim2g; iz++)
                {
                    gpoint[2]      = start2 + h2 * iz;
                    const double r = gpoint.minimage(origin_, cell_, bc_);

                    if (r > 1.e-10)
                        pfunc[iy1 + iz] = qtotal_ / r;
                    else
                        pfunc[iy1 + iz] = 1.e10 * qtotal_;
                }
            }
        }
    }
    else
    {
        // dipole
        for (int ix = 0; ix < dim0g; ix++)
        {
            int ix1   = ix * incx;
            gpoint[0] = start0 + h0 * ix;
            for (int iy = 0; iy < dim1g; iy++)
            {
                int iy1   = ix1 + iy * incy;
                gpoint[1] = start1 + h1 * iy;
                for (int iz = 0; iz < dim2g; iz++)
                {
                    gpoint[2] = start2 + h2 * iz;
                    const Vector3D vr(gpoint.vminimage(origin_, cell_, bc_));
                    const double r2
                        = vr[0] * vr[0] + vr[1] * vr[1] + vr[2] * vr[2];
                    const double r3 = r2 * sqrt(r2);
                    if (r3 > 1.e-10)
                        pfunc[iy1 + iz] = (T)((vr * dipole_moment_) / r3);
                    else
                        pfunc[iy1 + iz] = (T)(1.e10 * (vr * dipole_moment_));
                }
            }
        }
    }
}
template void MultipoleExpansion::expand<double>(pb::GridFunc<double>& func);
template void MultipoleExpansion::expand<float>(pb::GridFunc<float>& func);
template void MultipoleExpansion::expand2d<double>(pb::GridFunc<double>& func);
template void MultipoleExpansion::expand2d<float>(pb::GridFunc<float>& func);
template void MultipoleExpansion::expand3d<double>(pb::GridFunc<double>& func);
template void MultipoleExpansion::expand3d<float>(pb::GridFunc<float>& func);
// template class MultipoleExpansion<double>;
// template class MultipoleExpansion<float>;

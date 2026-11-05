// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "RadialMeshFunction.h"
#include "MPIdata.h"

#include <cmath>
#include <iostream>
#include <stdlib.h>

const double inv48 = 1. / 48;

// Gaussian cutoff function
double RadialMeshFunction::gauss_cutoff(const double r, const double rcut)
{
    if (r < rcut) return 1.;

    double s = (r - rcut) / rcut;
    return exp(-40. * (s * s));
}

// smooth cutoff function
double RadialMeshFunction::gcutoff(const double g, const double gcut)
{
    if (g < gcut) return 1.;

    double t = (g - gcut) / gcut;
    return exp(-6. * (t * t));
}

void RadialMeshFunction::gauss_filter(const double rcut, const int j)
{
    int n = (int)y_[j].size();
    for (int i = 0; i < n; i++)
    {
        y_[j][i] *= gauss_cutoff(x_[i], rcut);
    }
}

void RadialMeshFunction::read(
    const int nrow, const int ncol, std::ifstream* tfile)
{
    assert(ncol > 1);
    assert(nrow > 1);

    x_.resize(nrow);
    y_.resize(ncol - 1);
    for (int i = 0; i < ncol - 1; i++)
        y_[i].resize(nrow);

    //(*MPIdata::sout)<<"read "<<nrow<<" rows and "<<ncol<<" columns"<<endl;
    for (int k = 0; k < nrow; k++)
    {
        (*tfile) >> x_[k];
        for (int i = 0; i < ncol - 1; i++)
            (*tfile) >> y_[i][k];
        while (tfile->get() != '\n')
            ;
    }

    drlin_ = x_[1] - x_[0];
    assert(drlin_ > 1.e-15);
    invdr_ = 1. / drlin_;
}

void RadialMeshFunction::bcast(MPI_Comm comm, const int root)
{
    int mpi_rank;
    MPI_Comm_rank(comm, &mpi_rank);

    MPI_Bcast(&drlin_, 1, MPI_DOUBLE, root, comm);
    assert(drlin_ > 1.e-15);
    invdr_ = 1. / drlin_;

    // Bcast dimensions
    int nn[2] = { (int)x_.size(), (int)y_.size() };
    MPI_Bcast(&nn[0], 2, MPI_INT, root, comm);

    // Resize vectors on receiving PEs
    if (mpi_rank != root)
    {
        x_.resize(nn[0]);
        y_.resize(nn[1]);
        for (int j = 0; j < nn[1]; j++)
            y_[j].resize(nn[0]);
    }
    assert(x_.size() > 0);

    int mpirc = MPI_Bcast(&x_[0], nn[0], MPI_DOUBLE, root, comm);
    if (mpirc != MPI_SUCCESS)
    {
        (*MPIdata::sout) << "RadialMeshFunction::bcast() failed!!!"
                         << std::endl;
        MPI_Abort(comm, EXIT_FAILURE);
    }

    for (int i = 0; i < nn[1]; i++)
    {
        //(*MPIdata::sout)<<"y["<<i<<"]:"<<y_[i].size()<<endl;
        assert(y_[i].size() > 0);
        mpirc = MPI_Bcast(&y_[i][0], nn[0], MPI_DOUBLE, root, comm);
        if (mpirc != MPI_SUCCESS)
        {
            (*MPIdata::sout)
                << "RadialMeshFunction::bcast() failed!!!" << std::endl;
            MPI_Abort(comm, EXIT_FAILURE);
        }
    }
}

// radial integratation of f
double RadialMeshFunction::radint(const int j)
{
    std::vector<double>& f = y_[j];
    const int n            = (int)x_.size();

    // Alternativ extended Simpson rule
    // Numerical Recipes, p. 108

    //(*MPIdata::sout)<<"Radial integration on regular grid\n";
    double rval = (f[0] * x_[0] * x_[0]) * 17.;
    rval += (f[1] * x_[1] * x_[1]) * 59.;
    rval += (f[2] * x_[2] * x_[2]) * 43.;
    rval += (f[3] * x_[3] * x_[3]) * 49.;
    rval += (f[n - 4] * x_[n - 4] * x_[n - 4]) * 49.;
    rval += (f[n - 3] * x_[n - 3] * x_[n - 3]) * 43.;
    rval += (f[n - 2] * x_[n - 2] * x_[n - 2]) * 59.;
    rval += (f[n - 1] * x_[n - 1] * x_[n - 1]) * 17.;
    rval *= inv48;

    for (int i = 4; i < n - 4; i++)
    {
        rval += f[i] * x_[i] * x_[i];
    }

    rval *= (x_[1] - x_[0]);

    if (f[n - 1] * x_[n - 1] * x_[n - 1] > 5.e-3)
    {
        (*MPIdata::sout) << "WARNING RadialMeshFunction" << std::endl;
        (*MPIdata::sout) << "WARNING radint(): f[n-1]*x_[n-1]*x_[n-1]="
                         << f[n - 1] * x_[n - 1] * x_[n - 1] << std::endl;
        (*MPIdata::sout) << "WARNING radint(): x_[n-1]=" << x_[n - 1]
                         << std::endl;
        (*MPIdata::sout) << "WARNING n=" << n << std::endl;
    }

    return rval;
}

// radial integratation of f^2
double RadialMeshFunction::radintf2(const int j)
{
    std::vector<double>& f = y_[j];
    const int n            = (int)x_.size();

    // Alternativ extended Simpson rule
    // Numerical Recipes, p. 108

    //(*MPIdata::sout)<<"Radial integration on regular grid\n";
    double rval = (f[0] * f[0] * x_[0] * x_[0]) * 17.;
    rval += (f[1] * f[1] * x_[1] * x_[1]) * 59.;
    rval += (f[2] * f[2] * x_[2] * x_[2]) * 43.;
    rval += (f[3] * f[3] * x_[3] * x_[3]) * 49.;
    rval += (f[n - 4] * f[n - 4] * x_[n - 4] * x_[n - 4]) * 49.;
    rval += (f[n - 3] * f[n - 3] * x_[n - 3] * x_[n - 3]) * 43.;
    rval += (f[n - 2] * f[n - 2] * x_[n - 2] * x_[n - 2]) * 59.;
    rval += (f[n - 1] * f[n - 1] * x_[n - 1] * x_[n - 1]) * 17.;
    rval *= inv48;

    for (int i = 4; i < n - 4; i++)
    {
        rval += (f[i] * f[i] * x_[i] * x_[i]);
    }

    rval *= (x_[1] - x_[0]);

    if (f[n - 1] * f[n - 1] * x_[n - 1] * x_[n - 1] > 5.e-3)
    {
        (*MPIdata::sout) << "WARNING RadialMeshFunction" << std::endl;
        (*MPIdata::sout) << "WARNING radint(): f[n-1]*f[n-1]*x_[n-1]*x_[n-1]="
                         << f[n - 1] * f[n - 1] * x_[n - 1] * x_[n - 1]
                         << std::endl;
        (*MPIdata::sout) << "WARNING radint(): x_[n-1]=" << x_[n - 1]
                         << std::endl;
        (*MPIdata::sout) << "WARNING n=" << n << std::endl;
    }

    return rval;
}

// compute the inverse FT
void RadialMeshFunction::inv_ft(const std::vector<double>& rgrid,
    std::vector<double>& rfunc, const int lval, const int j)
{
    assert(rgrid.size() == rfunc.size());

    std::vector<double>& gfunc = y_[j];
    const double dr            = rgrid[1] - rgrid[0];

    const int irmax = (int)rgrid.size();
    const int ikmax = (int)x_.size();

    memset(&rfunc[0], 0, rfunc.size() * sizeof(double));

#ifdef DEBUG
    if (onpe0) (*MPIdata::sout) << " Inv. FT for l=" << lval << std::endl;
#endif

    int irmin = 0;
    double ri = 0.;
    RadialMeshFunction rwork(x_);
    std::vector<double>& work = rwork.y();

    switch (lval)
    {

        case 0:

            if (ri < 1.e-10)
            {
                irmin++;
                ri += dr;
                rfunc[0] = radint();
            }

            for (int ir = irmin; ir < irmax; ir++)
            {

                // loop over k
                work[0] = gfunc[0];
                for (int ik = 1; ik < ikmax; ik++)
                {

                    const double t1 = ri * x_[ik];
                    assert(fabs(t1) > 1.e-10);

                    work[ik] = gfunc[ik] * sin(t1) / t1;
                }

                // Integrate work
                rfunc[ir] = rwork.radint();

                ri += dr;
            }

            break;

        case 1:

            if (ri < 1.e-10)
            {
                irmin++;
                ri += dr;
                rfunc[0] = 0.;
            }

            for (int ir = irmin; ir < irmax; ir++)
            {

                work[0] = 0.;
                for (int ik = 1; ik < ikmax; ik++)
                {

                    const double t1 = ri * x_[ik];
                    assert(t1 > 0.);
                    const double it1 = (1. / t1);
                    const double t2  = (cos(t1) - sin(t1) * it1) * it1;

                    work[ik] = gfunc[ik] * t2;
                }

                // Integrate work
                rfunc[ir] = rwork.radint();

                ri += dr;
            }

            break;

        case 2:

            if (ri < 1.e-10)
            {
                irmin++;
                ri += dr;
                rfunc[0] = 0.;
            }

            for (int ir = irmin; ir < irmax; ir++)
            {

                work[0] = 0.;
                for (int ik = 1; ik < ikmax; ik++)
                {

                    const double t1 = ri * x_[ik];
                    assert(t1 > 1.e-10);
                    const double it1 = (1. / t1);
                    double t2        = (3. * it1 * it1 - 1.) * sin(t1);
                    t2 -= 3. * cos(t1) * it1;
                    t2 *= it1;

                    work[ik] = gfunc[ik] * t2;
                }

                // Integrate work
                rfunc[ir] = rwork.radint();

                ri += dr;
            }

            break;

        case 3:

            if (ri < 1.e-10)
            {
                irmin++;
                ri += dr;
                rfunc[0] = 0.;
            }

            for (int ir = irmin; ir < irmax; ir++)
            {
                work[0] = 0.;
                for (int ik = 1; ik < ikmax; ik++)
                {
                    const double t1 = ri * x_[ik];
                    assert(t1 > 1.e-10);
                    const double it1 = (1. / t1);
                    double t2        = (15. * it1 * it1 - 6.) * it1 * sin(t1);
                    t2 -= (15. * it1 * it1 - 1.) * cos(t1);
                    t2 *= it1;

                    work[ik] = gfunc[ik] * t2;
                }

                // Integrate work
                rfunc[ir] = rwork.radint();

                ri += dr;
            }

            break;

        default:

            (*MPIdata::serr)
                << "invft: angular momentum state > 2 not implemented"
                << std::endl;
            exit(1);
    }

    double alpha = M_2_PI;
    assert(irmax > 10);
    int ione = 1;
    DSCAL(&irmax, &alpha, &rfunc[0], &ione);
}

void RadialMeshFunction::compute_ft(const std::vector<double>& ggrid,
    std::vector<double>& gcof, const int lval, std::ofstream* tfile,
    const int j)
{
    std::vector<double>& func = y_[j];

    assert(x_.size() == func.size());

    const int irmax = (int)func.size();
    const int ikmax = (int)ggrid.size();

    memset(&gcof[0], 0, ikmax * sizeof(double));

    RadialMeshFunction rwork(x_);
    std::vector<double>& work = rwork.y();

    // Integrate work to get coeff. 0
    switch (lval)
    {
        case 0:
            gcof[0] = radint(j);
            break;
        default:
            gcof[0] = 0.;
            break;
    }

    int irmin;
    // Loop over frequency components higher than 0
    for (int ik = 1; ik < ikmax; ik++)
    {

        if (x_[0] > 1.e-10)
        {
            irmin = 0;
        }
        else
        {
            irmin   = 1;
            work[0] = 0.;
        }

        const double vk = ggrid[ik];

        switch (lval)
        {
            case 0:

                for (int ir = irmin; ir < irmax; ir++)
                {
                    double t1 = x_[ir] * vk;
                    assert(t1 > 1.e-10);
                    work[ir] = func[ir] * sin(t1) / t1;
                }
                break;

            case 1:

                for (int ir = irmin; ir < irmax; ir++)
                {
                    double t1 = x_[ir] * vk;
                    assert(t1 > 1.e-10);
                    double t2 = (cos(t1) * t1 - sin(t1)) / (t1 * t1);
                    work[ir]  = func[ir] * t2;
                }
                break;

            case 2:

                for (int ir = irmin; ir < irmax; ir++)
                {
                    double t1        = x_[ir] * vk;
                    const double it1 = (1. / t1);
                    assert(t1 > 1.e-10);
                    double t2 = (3. * it1 * it1 - 1.) * sin(t1) * it1;
                    t2 -= 3. * cos(t1) * it1 * it1;
                    work[ir] = func[ir] * t2;
                }
                break;

            case 3:

                for (int ir = irmin; ir < irmax; ir++)
                {
                    double t1        = x_[ir] * vk;
                    const double it1 = (1. / t1);
                    assert(t1 > 1.e-10);
                    double t2 = (15. * it1 * it1 - 6.) * it1 * sin(t1) * it1;
                    t2 -= (15. * it1 * it1 - 1.) * cos(t1) * it1;
                    work[ir] = func[ir] * t2;
                }
                break;

            default:

                (*MPIdata::serr)
                    << "rft: angular momentum " << lval << " not implemented\n";
                exit(1);
        }

        // Integrate work to get coeff. ik
        gcof[ik] = rwork.radint();
    }

    if (tfile != nullptr)
        for (int ik = 0; ik < ikmax; ik++)
        {
            (*tfile) << ggrid[ik] << "\t" << gcof[ik] << std::endl;
        }
}

void RadialMeshFunction::rft(RadialMeshFunction& filt_func, const int lval,
    const double hmax, const bool print, const int j, const bool printFlag)
{
    assert(hmax > 0.);
    assert(y_[j].size() == x_.size());

    const double gmax = 1.5 * M_PI / hmax;
    const double dg   = 0.005;
    const int ikmax   = (int)(gmax / dg);
#ifdef DEBUG
    (*MPIdata::sout) << "Uses " << ikmax << " points for FT" << std::endl;
#endif

    // Arrays for FT of func and its grid
    std::vector<double> ggrid(ikmax);

    // Generate ggrid
    for (int i = 0; i < ikmax; i++)
        ggrid[i] = (double)(i)*dg;

    RadialMeshFunction gfunc(ggrid);
    std::vector<double>& gcof = gfunc.y();

    // compute FT of func
    std::ofstream tfile;
    if (print)
    {
        std::string filename("Freq_init.dat");
        tfile.open(filename.data(), std::ios::out);
    }
    compute_ft(ggrid, gcof, lval, &tfile, j);

    // Fourier Filter the transform gcof
    const double gcut = 0.75 * M_PI / hmax;

    if (printFlag)
        (*MPIdata::sout) << "RadialMeshFunction::rft(),"
                         << " Cutoff for pseudopotential: " << gcut * gcut
                         << "[Ry]" << std::endl;

    for (int ik = 0; ik < ikmax; ik++)
    {
        gcof[ik] *= gcutoff(ggrid[ik], gcut);
    }
    if (print)
    {
        std::string filename("Freq_filt.dat");
        std::ofstream tfile(filename.data());
        for (int ik = 0; ik < ikmax; ik++)
        {
            tfile << ggrid[ik] << "\t" << gcof[ik] << std::endl;
        }
    }

    // compute inverse FT of gcof defined on ggrid to get filt_func
    gfunc.inv_ft(filt_func.x(), filt_func.y(0), lval, 0);
}

void RadialMeshFunction::printLocalPot(
    const std::string& name, const short ll, std::ofstream* tfile)
{
    assert(tfile != nullptr);

    const std::vector<double>& potloc = y_[0];

    (*tfile) << "\"" << name << " local pseudopotential, l=" << ll << "\""
             << std::endl;

    for (unsigned short idx = 0; idx < x_.size(); idx += 3)
    {
        (*tfile) << x_[idx] << "\t" << potloc[idx] << std::endl;
    }
    (*tfile) << std::endl;
    (*tfile) << std::endl;
}

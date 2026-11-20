// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "fermi.h"

#include "MGmol_abort.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

// fill occupations according to the fermi-dirac distribution:
//   f(x) = 1 / (1 + Exp[kB*(E-mu)])
// states 0 to max_occ-1, set to 0 states above max_occ-1
// returns the number of electrons
template <typename ScalarType>
double fermi_distribution(const double mu, const int max_occ, const double kBT,
    const std::vector<ScalarType>& energies, std::vector<ScalarType>& occ)
{
    assert(kBT >= 0.);
    assert(occ.size() == energies.size());

    const int dim = occ.size();

    for (int i = max_occ; i < dim; i++)
        occ[i] = 0.;

    // if(onpe0)std::cout<<"fermi_distribution() with mu="
    //             <<mu<<" and kBT "<<kBT<<endl;
    double sum_occ = 0.;

    if (kBT > 1.e-10)
    {
        const double beta = 1. / kBT;
        for (int st = 0; st < max_occ; st++)
        {
            const double t1 = (energies[st] - mu) * beta;

            if (t1 > 0.)
            {
                const double t2 = exp(-t1);
                occ[st]         = t2 / (1. + t2);
            }
            else
            {
                const double t2 = exp(t1);
                occ[st]         = (ScalarType)(1. / (1. + t2));
            }

            sum_occ += occ[st];
        }
    }
    else // zero temperature
    {
        for (int st = 0; st < max_occ; st++)
        {
            const double t1 = (energies[st] - mu);

            if (t1 > 1.e-6)
            {
                occ[st] = 0.;
            }
            else
            {
                occ[st] = 1.;
            }
            sum_occ += occ[st];
        }
    }
    return sum_occ;
}

// find the Fermi level by a bisection
// algorithm adapted from numerical recipes, 2nd edition
// and fill orbitals accordingly (in fermi_distribution)
// Output: return value of mu and fill occ with occupations
//         between 0 and 1
template <typename ScalarType>
double compute_chemical_potential_and_occupations(
    const std::vector<ScalarType>& energies, const double width,
    const double nel, const int max_numst, const bool onpe0,
    std::vector<ScalarType>& occ)
{
    assert(energies.size() > 0);
    assert(energies.size() == occ.size());
    assert(nel >= 0.);

    const unsigned int dim  = energies.size();
    const int maxit         = 100;
    const double charge_tol = 1.0e-12;

    const double emin = *std::min_element(energies.begin(), energies.end());
    const double emax = *std::max_element(energies.begin(), energies.end());

    double mu1 = emin - 0.001;
    double mu2 = emax + 10. * width;
    assert(mu1 < mu2);
    bool done = false;

    if (nel <= 0.)
    {
        mu1 = -10000.;
        mu2 = 10000.;
    }

    double mu;

    // if number of electrons larger or equal to twice
    // the number of slots, all the slots are fully occupied
    if (static_cast<double>(dim) <= nel)
    {
        done = true;
        mu   = mu2;
        for (unsigned int i = 0; i < dim; i++)
            occ[i] = 1.;
    }

    double f2 = 0.;
    if (!done)
    {
        f2 = fermi_distribution(mu2, max_numst, width, energies, occ)
             - static_cast<double>(nel);
        // no unoccupied states
        if (fabs(f2) < charge_tol)
        {
            done = true;
            mu   = mu2;
        }
    }
    double f1 = 0.;
    if (!done)
    {
        f1 = fermi_distribution(mu1, max_numst, width, energies, occ)
             - static_cast<double>(nel);
        if (fabs(f1) < charge_tol)
        {
            if (onpe0) std::cout << "only unoccupied states" << std::endl;
            done = true;
            mu   = mu1;
        }
    }

    if (!done)
    {
        if (f1 * f2 > 0.)
        {
            std::cerr << "ERROR: mu1=" << mu1 << ", mu2=" << mu2 << std::endl;
            std::cerr << "ERROR: f1=" << f1 << ", f2=" << f2 << std::endl;
            std::cerr << "nel=" << nel << ", width=" << width << std::endl;
            mgmol_abort();
        }

        double dmu;
        if (f1 < 0.)
        {
            mu  = mu1;
            dmu = mu2 - mu1;
        }
        else
        {
            mu  = mu2;
            dmu = mu1 - mu2;
        }

        // main loop
        int iter = 0;
        double f = 0.;
        do
        {
            iter++;

            dmu *= 0.5;
            mu1 = mu + dmu;
            f = fermi_distribution(mu1, max_numst, width, energies, occ) - nel;

            if (f <= 0.)
            {
                mu = mu1;
                f  = -f;
            }

        } while ((iter < maxit) && (f > charge_tol));

        if (f > charge_tol)
        {
            if (onpe0)
            {
                std::cout << "WARNING: "
                             "ProjectedMatrices<MatrixType>::"
                             "computeChemicalPotentialAndOccupations()"
                          << std::endl;
                std::cout << "Iterations did not converge to tolerance "
                          << std::scientific << charge_tol << std::endl;
                std::cout << "f= " << f << ", nel=" << nel
                          << ", max_numst=" << max_numst << std::endl;
            }
        }
    }

    return mu;
}

template double fermi_distribution(const double mu, const int max_occ,
    const double kBT, const std::vector<double>& energies,
    std::vector<double>& occ);
template double compute_chemical_potential_and_occupations(
    const std::vector<double>& energies, const double width, const double nel,
    const int max_numst, const bool onpe0, std::vector<double>& occ);

// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "HamiltonianMVPSolver.h"
#include "Control.h"
#include "Electrostatic.h"
#include "Energy.h"
#include "Ions.h"
#include "KBPsiMatrixSparse.h"
#include "LocGridOrbitals.h"
#include "MGmol.h"
#include "Potentials.h"
#include "ProjectedMatrices.h"
#include "ProjectedMatricesSparse.h"
#include "ReplicatedMatrix.h"
#include "tools.h"

#include <iomanip>
double evalEntropyMVP(ProjectedMatricesInterface* projmatrices,
    const bool print_flag, std::ostream& os);

template <class MatrixType, class ProjMatrixType, class OrbitalsType>
Timer HamiltonianMVPSolver<MatrixType, ProjMatrixType, OrbitalsType>::solve_tm_(
    "HamiltonianMVPSolver::solve");
template <class MatrixType, class ProjMatrixType, class OrbitalsType>
Timer
    HamiltonianMVPSolver<MatrixType, ProjMatrixType, OrbitalsType>::target_tm_(
        "HamiltonianMVPSolver::target");

template <class MatrixType, class ProjMatrixType, class OrbitalsType>
HamiltonianMVPSolver<MatrixType, ProjMatrixType,
    OrbitalsType>::HamiltonianMVPSolver(std::ostream& os, Ions& ions,
    Rho<OrbitalsType>* rho, Energy<OrbitalsType>* energy,
    Electrostatic* electrostat, Hamiltonian<OrbitalsType>* hamiltonian,
    MGmol<OrbitalsType>* mgmol_strategy, const int numst,
    const short n_inner_steps, const MatrixType& hinit, const double mixing,
    const bool try_shorter_intervals)
    : os_(os),
      n_inner_steps_(n_inner_steps),
      ions_(ions),
      mixing_(mixing),
      try_shorter_intervals_(try_shorter_intervals)
{
    assert(n_inner_steps > 0);

    rho_            = rho;
    energy_         = energy;
    electrostat_    = electrostat;
    hamiltonian_    = hamiltonian;
    mgmol_strategy_ = mgmol_strategy;

    numst_ = numst;

    hmatrix_         = new MatrixType(hinit);
    initial_hmatrix_ = new MatrixType(hinit);
}

template <class MatrixType, class ProjMatrixType, class OrbitalsType>
HamiltonianMVPSolver<MatrixType, ProjMatrixType,
    OrbitalsType>::~HamiltonianMVPSolver()
{
    delete hmatrix_;
    delete initial_hmatrix_;
}

template <class MatrixType, class ProjMatrixType, class OrbitalsType>
void HamiltonianMVPSolver<MatrixType, ProjMatrixType, OrbitalsType>::reset()
{
    (*hmatrix_) = (*initial_hmatrix_);
}

// update density matrix in N x N space
template <class MatrixType, class ProjMatrixType, class OrbitalsType>
int HamiltonianMVPSolver<MatrixType, ProjMatrixType, OrbitalsType>::solve(
    OrbitalsType& orbitals)
{
    Control& ct     = *(Control::instance());
    MGmol_MPI& mmpi = *(MGmol_MPI::instance());

    assert(numst_ == (int)orbitals.numst());
    assert(n_inner_steps_ > 0);

    solve_tm_.start();

    if (mmpi.PE0() && ct.verbose > 1)
    {
        os_ << "---------------------------------------------------"
            << std::endl;
        os_ << "Update DM functions using Hamiltonian MVP Solver..."
            << std::endl;
        os_ << "---------------------------------------------------"
            << std::endl;
    }

    // save initial matrix to enable reset
    (*initial_hmatrix_) = (*hmatrix_);

    KBPsiMatrixSparse kbpsi(nullptr);
    kbpsi.setup(ions_);

    ProjMatrixType* projmatrices
        = dynamic_cast<ProjMatrixType*>(orbitals.getProjMatrices());

    // save computed vh for a fair energy "comparison" with vh computed
    // in close neigborhood
    const pb::GridFunc<POTDTYPE> vh_init(electrostat_->getVh());

    orbitals.setDataWithGhosts();

    // compute linear component of H
    MatrixType h11nl("h11nl", numst_);

    kbpsi.computeAll(ions_, orbitals);

    kbpsi.computeHvnlMatrix(&kbpsi, ions_, h11nl);

    MatrixType h11("h11", numst_);

    OrbitalsType hphi("HMVP_hphi", orbitals);

    for (int inner_it = 0; inner_it < n_inner_steps_; inner_it++)
    {
        if (mmpi.PE0() && ct.verbose > 1)
        {
            os_ << "---------------------------" << std::endl;
            os_ << "Inner iteration " << inner_it << std::endl;
            os_ << "---------------------------" << std::endl;
        }

        //
        // evaluate energy at origin
        //
        projmatrices->assignH(*hmatrix_);
        projmatrices->setHB2H();

        // update DM and compute entropy
        projmatrices->updateDM();
        double ts0 = evalEntropyMVP(projmatrices, true, os_);
        // Update density
        rho_->update(orbitals);

        // Update potential
        mgmol_strategy_->update_pot(vh_init, ions_);

        energy_->saveVofRho();

        // compute new h11 for the current potential by adding local part to
        // nonlocal components
        h11 = h11nl;
        hamiltonian_->applyLocal(numst_, orbitals, hphi);
        orbitals.addDotWithNcol2Matrix(hphi, h11);

        projmatrices->assignH(h11);
        projmatrices->setHB2H();

        // compute energy at origin
        const int printE = (ct.verbose > 1) ? 1 : 0;
        double e0        = energy_->evaluateTotal(
            ts0, projmatrices, ions_, orbitals, printE, os_);

        // set target H to H computed with last DM
        MatrixType htarget(projmatrices->getH());
        MatrixType delta_h(htarget);
        delta_h -= *hmatrix_;

        double beta = 0.;
        if (mixing_ > 0.)
        {
            beta = mixing_;
            if (mmpi.PE0() && ct.verbose > 1)
                os_ << "HMVP with beta = " << beta << std::endl;
        }
        else
        {
            // update DM and compute entropy
            projmatrices->updateDM();
            double ts1 = evalEntropyMVP(projmatrices, true, os_);
            // Update density
            rho_->update(orbitals);

            // Update potential
            mgmol_strategy_->update_pot(vh_init, ions_);

            energy_->saveVofRho();

            // update H and compute energy at midpoint
            h11 = h11nl;
            hamiltonian_->applyLocal(numst_, orbitals, hphi);
            orbitals.addDotWithNcol2Matrix(hphi, h11);

            projmatrices->assignH(h11);
            projmatrices->setHB2H();

            // compute energy at end (beta=1.)
            double e1 = energy_->evaluateTotal(
                ts1, projmatrices, ions_, orbitals, printE, os_);

            //
            // evaluate energy at mid-point
            //
            h11 = *hmatrix_;
            h11.axpy(0.5, delta_h);

            projmatrices->assignH(h11);
            projmatrices->setHB2H();

            // update DM and entropy
            projmatrices->updateDM();
            double tsi = evalEntropyMVP(projmatrices, true, os_);

            // Update density
            rho_->update(orbitals);

            // Update potential
            mgmol_strategy_->update_pot(vh_init, ions_);

            energy_->saveVofRho();

            // update H with new potential
            h11 = h11nl;
            hamiltonian_->applyLocal(numst_, orbitals, hphi);
            orbitals.addDotWithNcol2Matrix(hphi, h11);

            projmatrices->assignH(h11);
            projmatrices->setHB2H();

            // compute energy at midpoint
            double ei = energy_->evaluateTotal(
                tsi, projmatrices, ions_, orbitals, printE, os_);

            // line minimization
            beta = minQuadPolynomialFrom3values(
                e0, e1, ei, (ct.verbose > 2), os_);

            if (mmpi.PE0() && ct.verbose > 0)
            {
                os_ << std::setprecision(12);
                os_ << std::fixed << "Inner iteration " << inner_it
                    << ", E0=" << e0 << ", E(1/2)=" << ei << ", E1=" << e1;
                os_ << std::scientific << " -> beta=" << beta;
                os_ << std::endl;
            }

            if (try_shorter_intervals_)
            {
                double factor = 0.5;
                while (
                    beta
                    < 0.) // try with a shorter interval if line search failed
                {
                    if (mmpi.PE0() && ct.verbose > 1)
                    {
                        os_ << "HMVP: Reduce interval by factor " << factor
                            << " ..." << std::endl;
                    }
                    ts1 = tsi;
                    e1  = ei;

                    h11 = *hmatrix_;
                    h11.axpy(0.5 * factor, delta_h);

                    projmatrices->assignH(h11);
                    projmatrices->setHB2H();

                    // update DM and entropy
                    projmatrices->updateDM();
                    tsi = evalEntropyMVP(projmatrices, true, os_);

                    // Update density
                    rho_->update(orbitals);

                    // Update potential
                    mgmol_strategy_->update_pot(vh_init, ions_);

                    energy_->saveVofRho();

                    // update H
                    h11 = h11nl;
                    hamiltonian_->applyLocal(numst_, orbitals, hphi);
                    orbitals.addDotWithNcol2Matrix(hphi, h11);

                    projmatrices->assignH(h11);
                    projmatrices->setHB2H();

                    // compute energy at end (beta=1.)
                    ei = energy_->evaluateTotal(
                        tsi, projmatrices, ions_, orbitals, printE, os_);

                    // line minimization
                    beta = minQuadPolynomialFrom3values(
                        e0, e1, ei, (ct.verbose > 2), os_);

                    if (mmpi.PE0() && ct.verbose > 0)
                    {
                        os_ << std::setprecision(12);
                        os_ << std::fixed << "Inner iteration " << inner_it
                            << ", E0=" << e0 << ", E(1/2)=" << ei
                            << ", E1=" << e1;
                        os_ << std::scientific << " -> beta=" << beta;
                        os_ << std::endl;
                    }

                    beta *= factor;

                    factor *= 0.5;
                }
            } // try_shorter_intervals_
            else
            {
                if (beta < 0.)
                {
                    if (mmpi.PE0())
                        os_ << "!!! HMVP iteration failed: beta<0 !!!"
                            << std::endl;
                    projmatrices->assignH(*hmatrix_);
                    projmatrices->setHB2H();

                    return -1;
                }
            }
        }

        hmatrix_->axpy(beta, delta_h);

    } // inner iterations

    projmatrices->assignH(*hmatrix_);
    projmatrices->setHB2H();

    projmatrices->updateDM();

    if (mmpi.PE0() && ct.verbose > 1)
    {
        os_ << "-----------------------------" << std::endl;
        os_ << "End Hamiltonian MVP Solver..." << std::endl;
        os_ << "-----------------------------" << std::endl;
    }
    solve_tm_.stop();

    return 0;
}

template <class MatrixType, class ProjMatrixType, class OrbitalsType>
void HamiltonianMVPSolver<MatrixType, ProjMatrixType,
    OrbitalsType>::printTimers(std::ostream& os)
{
    if (onpe0)
    {
        os << std::setprecision(2) << std::fixed << std::endl;
        solve_tm_.print(os);
        target_tm_.print(os);
    }
}

// explicit instantiation of class
#ifdef MGMOL_USE_SCALAPACK
template class HamiltonianMVPSolver<dist_matrix::DistMatrix<DISTMATDTYPE>,
    ProjectedMatrices<dist_matrix::DistMatrix<DISTMATDTYPE>>,
    LocGridOrbitals<ORBDTYPE>>;
template class HamiltonianMVPSolver<dist_matrix::DistMatrix<DISTMATDTYPE>,
    ProjectedMatrices<dist_matrix::DistMatrix<DISTMATDTYPE>>,
    ExtendedGridOrbitals<ORBDTYPE>>;
#endif
template class HamiltonianMVPSolver<ReplicatedMatrix,
    ProjectedMatrices<ReplicatedMatrix>, ExtendedGridOrbitals<ORBDTYPE>>;

// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "MVPSolver.h"

#include "Control.h"
#include "Electrostatic.h"
#include "Energy.h"
#include "Ions.h"
#include "KBPsiMatrixSparse.h"
#include "LocGridOrbitals.h"
#include "MGmol.h"
#include "Potentials.h"
#include "ProjectedMatrices.h"
#include "ReplicatedMatrix.h"
#include "Rho.h"
#include "tools.h"

#ifdef MGMOL_USE_SCALAPACK
#include "DistMatrix.h"
#endif

#include <iomanip>

template <class OrbitalsType, class MatrixType>
Timer MVPSolver<OrbitalsType, MatrixType>::solve_tm_("MVPSolver::solve");
template <class OrbitalsType, class MatrixType>
Timer MVPSolver<OrbitalsType, MatrixType>::target_tm_("MVPSolver::target");

double evalEntropyMVP(ProjectedMatricesInterface* projmatrices,
    const bool print_flag, std::ostream& os)
{
    const double ts = 0.5 * projmatrices->computeEntropy(); // in [Ha]
    if (onpe0 && print_flag)
        os << std::setprecision(8) << "compute Entropy: TS=" << ts << "[Ha]"
           << std::endl;

    return ts;
}

template <class OrbitalsType, class MatrixType>
MVPSolver<OrbitalsType, MatrixType>::MVPSolver(MPI_Comm comm, std::ostream& os,
    Ions& ions, Rho<OrbitalsType>* rho, Energy<OrbitalsType>* energy,
    Electrostatic* electrostat, Hamiltonian<OrbitalsType>* hamiltonian,
    MGmol<OrbitalsType>* mgmol_strategy, const int numst, const double kbT,
    const std::vector<std::vector<int>>& global_indexes,
    const short n_inner_steps, const double mixing, const double tol_de0,
    const bool use_old_dm)
    : comm_(comm),
      os_(os),
      n_inner_steps_(n_inner_steps),
      use_old_dm_(use_old_dm),
      ions_(ions),
      numst_(numst),
      mixing_(mixing),
      tol_de0_(tol_de0)
{
    Control& ct = *(Control::instance());
    if (onpe0 && ct.verbose > 0)
    {
        os_ << "MVPSolver..." << std::endl;
        if (use_old_dm_ && ct.verbose > 1)
            os_ << "MVPSolver uses old DM..." << std::endl;
    }

    rho_            = rho;
    energy_         = energy;
    electrostat_    = electrostat;
    hamiltonian_    = hamiltonian;
    mgmol_strategy_ = mgmol_strategy;

    work_ = new MatrixType("workMVP", numst_, numst_);

    proj_mat_work_ = new ProjectedMatrices<MatrixType>(numst_, false, kbT);
    proj_mat_work_->setup(global_indexes);
}

template <class OrbitalsType, class MatrixType>
MVPSolver<OrbitalsType, MatrixType>::~MVPSolver()
{
    delete work_;
    delete proj_mat_work_;
}

template <class OrbitalsType, class MatrixType>
double MVPSolver<OrbitalsType, MatrixType>::evaluateDerivative(
    MatrixType& dmInit, MatrixType& delta_dm, const double ts0)
{
    work_->symm('l', 'l', 1., proj_mat_work_->getMatHB(), delta_dm, 0.);

    // factor 0.5 to get value in Hartree (HB is in Rydberg)
    double de0  = 0.5 * work_->trace();
    Control& ct = *(Control::instance());
    if (onpe0 && ct.verbose > 2)
        os_ << "Derivative of U in 0 = " << de0 << std::endl;

    //
    // evaluate numerical derivative of entropy in beta=0
    //
    // if( onpe0 ) os_<<"evaluate numerical derivative of entropy in
    // beta=0"<<endl;
    const double dbeta = 0.001;
    *work_             = dmInit;
    work_->axpy(dbeta, delta_dm);

    proj_mat_work_->setDM(*work_);
    proj_mat_work_->computeOccupationsFromDM();

    const double tsd0e = evalEntropyMVP(proj_mat_work_, false, os_);
    const double dts0e = (tsd0e - ts0) / dbeta;

    // if( onpe0 )os_<<"ts0="<<scientific<<ts0<<endl;
    // if( onpe0 )os_<<"tsd0e with 1.*beta="<<scientific<<tsd0e<<endl;

    if (onpe0 && ct.verbose > 2)
    {
        os_ << std::setprecision(8);
        os_ << "Numerical derivative of -TS in 0 = " << std::scientific
            << -dts0e << std::endl;
    }
    de0 -= dts0e;

    return de0;
}

template <class OrbitalsType, class MatrixType>
void MVPSolver<OrbitalsType, MatrixType>::buildTarget_MVP(
    MatrixType& h11, MatrixType& s11, MatrixType& target)
{
    target_tm_.start();

    Control& ct = *(Control::instance());
    if (onpe0 && ct.verbose > 1)
        std::cout << "buildTarget_MVP()..." << std::endl;

    proj_mat_work_->assignH(h11);

    int orbitals_index = 0;
    proj_mat_work_->setGramMatrix(s11, orbitals_index);

    //
    // compute target density matrix, with occupations>0 in numst only
    //
    proj_mat_work_->setHB2H();

    proj_mat_work_->updateDM();
    target = proj_mat_work_->dm();

    if (ct.verbose > 2)
    {
        double pnel = proj_mat_work_->getNel();
        if (onpe0) os_ << "Number of electrons = " << pnel << std::endl;
    }

    target_tm_.stop();
}

// update density matrix in N x N space
template <class OrbitalsType, class MatrixType>
int MVPSolver<OrbitalsType, MatrixType>::solve(OrbitalsType& orbitals)
{
    Control& ct = *(Control::instance());

    assert(numst_ == (int)orbitals.numst());
    assert(n_inner_steps_ > 0);

    solve_tm_.start();

    if (onpe0 && ct.verbose > 1)
    {
        os_ << "---------------------------------------------------------------"
               "-"
            << std::endl;
        os_ << "Update DM using MVP Solver..." << std::endl;
        os_ << "---------------------------------------------------------------"
               "-"
            << std::endl;
    }

    KBPsiMatrixSparse kbpsi(nullptr);
    kbpsi.setup(ions_);

    MatrixType s11("s11", numst_, numst_);
    {
        ProjectedMatrices<MatrixType>* projmatrices
            = dynamic_cast<ProjectedMatrices<MatrixType>*>(
                orbitals.getProjMatrices());
        assert(projmatrices);
        s11 = projmatrices->getGramMatrix();
    }

    {
        MatrixType dmInit("dm", numst_, numst_);

        // save computed vh for a fair energy "comparison" with vh computed
        // in close neigborhood
        const pb::GridFunc<POTDTYPE> vh_init(electrostat_->getVh());

        orbitals.setDataWithGhosts();

        // Update density
        if (use_old_dm_) rho_->update(orbitals);

        // compute linear component of H (nonlocal potential)
        MatrixType h11_nl("h11_nl", numst_, numst_);

        kbpsi.computeAll(ions_, orbitals);

        kbpsi.computeHvnlMatrix(&kbpsi, ions_, h11_nl);

        OrbitalsType hphi("MVP_hphi", orbitals);

        MatrixType h11(h11_nl);

        for (int inner_it = 0; inner_it < n_inner_steps_; inner_it++)
        {
            if (onpe0 && ct.verbose > 1)
            {
                os_ << "---------------------------" << std::endl;
                os_ << "Inner iteration " << inner_it << std::endl;
                os_ << "---------------------------" << std::endl;
            }

            // Update potential
            if (use_old_dm_ || inner_it > 0)
            {
                mgmol_strategy_->update_pot(vh_init, ions_);

                energy_->saveVofRho();
            }

            ProjectedMatrices<MatrixType>* current_proj_mat
                = (inner_it == 0)
                      ? dynamic_cast<ProjectedMatrices<MatrixType>*>(
                          orbitals.getProjMatrices())
                      : proj_mat_work_;

            const int printE = (ct.verbose > 1) ? 1 : 0;

            // compute h11 for the current potential by adding local part to
            // nonlocal components
            if (inner_it == 0)
            {
                hamiltonian_->applyLocal(numst_, orbitals, hphi);
            }
            else
            {
                hamiltonian_->applyDeltaPot(orbitals, hphi);
            }
            orbitals.addDotWithNcol2Matrix(hphi, h11);

            current_proj_mat->assignH(h11);
            current_proj_mat->setHB2H();

            if (inner_it == 0)
            {
                if (use_old_dm_)
                {
                    dmInit = current_proj_mat->dm();
                }
            }
            else
            {
                dmInit = proj_mat_work_->dm();
            }

            const double ts0
                = evalEntropyMVP(current_proj_mat, (ct.verbose > 1), os_);
            const double e0 = energy_->evaluateTotal(
                ts0, current_proj_mat, ions_, orbitals, printE, os_);

            MatrixType target("target", numst_, numst_);

            buildTarget_MVP(h11, s11, target);

            if (use_old_dm_ || inner_it > 0)
            {
                MatrixType delta_dm("delta_dm", numst_, numst_);
                delta_dm = target;
                delta_dm -= dmInit;

                double de0 = evaluateDerivative(dmInit, delta_dm, ts0);

                // check for convergence
                if (std::abs(de0) < tol_de0_ && inner_it > 0)
                {
                    if (onpe0 && ct.verbose > 0)
                        std::cout << "MVP: de0 = " << de0
                                  << ", convergence achieved" << std::endl;
                    break;
                }

                double beta = 0.;
                if (mixing_ > 0.)
                {
                    beta = mixing_;
                    if (onpe0 && ct.verbose > 0)
                    {
                        if (ct.verbose > 1)
                            os_ << "MVP with beta = " << beta << std::endl;
                        os_ << std::setprecision(12);
                        os_ << std::fixed << "MVP inner iteration " << inner_it
                            << ", E0=" << e0 << std::endl;
                    }
                }
                else
                {
                    //
                    // evaluate free energy at beta=1
                    //
                    if (onpe0 && ct.verbose > 2)
                        std::cout << "MVP --- Target energy..." << std::endl;
                    proj_mat_work_->setDM(target);
                    proj_mat_work_->computeOccupationsFromDM();
                    if (ct.verbose > 2) proj_mat_work_->printOccupations(os_);
                    const double nel = proj_mat_work_->getNel();
                    if (onpe0 && ct.verbose > 1)
                        os_ << "MVP --- Number of electrons at beta=1 : " << nel
                            << std::endl;

                    rho_->computeRho(orbitals, target);

                    mgmol_strategy_->update_pot(vh_init, ions_);

                    energy_->saveVofRho();

                    // update h11
                    hamiltonian_->applyDeltaPot(orbitals, hphi);
                    orbitals.addDotWithNcol2Matrix(hphi, h11);

                    proj_mat_work_->assignH(h11);
                    proj_mat_work_->setHB2H();

                    const double ts1
                        = evalEntropyMVP(proj_mat_work_, (ct.verbose > 2), os_);
                    const double e1 = energy_->evaluateTotal(ts1,
                        proj_mat_work_, ions_, orbitals, ct.verbose - 1, os_);

                    // line minimization
                    beta
                        = minQuadPolynomial(e0, e1, de0, (ct.verbose > 2), os_);
                    assert(!std::isnan(beta));

                    if (onpe0 && ct.verbose > 0)
                    {
                        os_ << std::setprecision(12);
                        os_ << std::fixed << "MVP inner iteration " << inner_it
                            << ", E0=" << e0 << ", E1=" << e1;
                        os_ << std::scientific << ", E0'=" << de0
                            << " -> beta=" << beta;
                        os_ << std::endl;
                    }
                }
                // update DM
                *work_ = dmInit;
                work_->axpy(beta, delta_dm);
            }
            else // no old DM to use
            {
                *work_ = target;
            }
            proj_mat_work_->setDM(*work_);

            if (inner_it < n_inner_steps_ - 1)
            {
                if (ct.verbose > 2)
                {
                    double pnel = proj_mat_work_->getNel();
                    if (onpe0)
                        os_ << "Number of electrons for interpolated DM = "
                            << pnel << std::endl;
                }
                rho_->computeRho(orbitals, *work_);
            }

        } // inner iterations

        // set DM to latest iteration value
        ProjectedMatrices<MatrixType>* projmatrices
            = dynamic_cast<ProjectedMatrices<MatrixType>*>(
                orbitals.getProjMatrices());
        projmatrices->setDM(*work_);
        projmatrices->setEigenvalues(proj_mat_work_->getEigenvalues());
        projmatrices->assignH(proj_mat_work_->getH());
        projmatrices->setHB2H();
    }

    if (onpe0 && ct.verbose > 1)
    {
        os_ << "---------------------------------------------------------------"
            << std::endl;
        os_ << "End MVP Solver..." << std::endl;
        os_ << "---------------------------------------------------------------"
            << std::endl;
    }
    solve_tm_.stop();

    return 0;
}

template <class OrbitalsType, class MatrixType>
void MVPSolver<OrbitalsType, MatrixType>::printTimers(std::ostream& os)
{
    solve_tm_.print(os);
    target_tm_.print(os);
}

#ifdef MGMOL_USE_SCALAPACK
template class MVPSolver<LocGridOrbitals<ORBDTYPE>,
    dist_matrix::DistMatrix<DISTMATDTYPE>>;
template class MVPSolver<ExtendedGridOrbitals<ORBDTYPE>,
    dist_matrix::DistMatrix<DISTMATDTYPE>>;
#endif
template class MVPSolver<LocGridOrbitals<ORBDTYPE>, ReplicatedMatrix>;
template class MVPSolver<ExtendedGridOrbitals<ORBDTYPE>, ReplicatedMatrix>;

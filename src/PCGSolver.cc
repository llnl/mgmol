// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE
#include "PCGSolver.h"

#include "LapFactory.h"

#include <cassert>
#include <iomanip>
#include <iostream>

template <class OperatorType, typename ScalarDataType, typename PrecondDataType>
void PCGSolver<OperatorType, ScalarDataType, PrecondDataType>::clear()
{
    for (short i = 0; i < (short)precond_oper_.size(); i++)
    {
        delete precond_oper_[i];
    }
    for (short i = 0; i < (short)gf_work_.size(); i++)
    {
        assert(gf_work_[i] != nullptr);
        delete gf_work_[i];
    }
    for (short i = 0; i < (short)gf_rcoarse_.size(); i++)
    {
        assert(gf_rcoarse_[i] != nullptr);
        delete gf_rcoarse_[i];
    }
    for (short i = 0; i < (short)gf_newv_.size(); i++)
    {
        assert(gf_newv_[i] != nullptr);
        delete gf_newv_[i];
    }
    // delete grids after pb::GridFunc<ScalarDataType> objects since those
    // have data members references to grids
    for (short i = 0; i < (short)grid_.size(); i++)
    {
        delete grid_[i];
    }
    precond_oper_.clear();
    grid_.clear();
    gf_work_.clear();
    gf_rcoarse_.clear();
    gf_newv_.clear();
}

template <class OperatorType, typename ScalarDataType, typename PrecondDataType>
void PCGSolver<OperatorType, ScalarDataType, PrecondDataType>::setupPrecon()
{
    // check if preconditioner is already setup
    // Assumes operator does not change, hence
    // a single setup is sufficient
    if (is_precond_setup_) return;

    // fine level
    pb::Grid* mygrid = new pb::Grid(oper_.grid());
    grid_.push_back(mygrid);
    const short nghosts = mygrid->ghost_pt();

    pb::Lap<PrecondDataType>* myoper
        = LapFactory<PrecondDataType>::createLap(*grid_[0], precond_lap_type_);
    precond_oper_.push_back(myoper);

    pb::GridFunc<PrecondDataType>* gf_work
        = new pb::GridFunc<PrecondDataType>(*grid_[0], bc_[0], bc_[1], bc_[2]);
    gf_work_.push_back(gf_work);

    // coarse levels
    nlevels_ = max_nlevels_;
    for (short ln = 1; ln <= max_nlevels_; ln++)
    {
        const bool flag_coarsen
            = ((!(mygrid->dim(0)
                   & 1)) // cannot coarsen if mesh not divisible by 2
                && (!(mygrid->dim(1) & 1)) && (!(mygrid->dim(2) & 1))
                && (static_cast<int>(mygrid->dim(0)) >= 2 * nghosts)
                && (static_cast<int>(mygrid->dim(1)) >= 2 * nghosts)
                && (static_cast<int>(mygrid->dim(2)) >= 2 * nghosts));

        if (!flag_coarsen)
        {
            nlevels_ = ln - 1;
            break;
        }
        pb::Grid* coarse_grid = new pb::Grid(mygrid->coarse_grid());
        grid_.push_back(coarse_grid);

        pb::Lap<PrecondDataType>* myoper
            = LapFactory<PrecondDataType>::createLap(
                *coarse_grid, pb::LapType::FD2nd);
        precond_oper_.push_back(myoper);

        gf_work = new pb::GridFunc<PrecondDataType>(
            *coarse_grid, bc_[0], bc_[1], bc_[2]);
        gf_work_.push_back(gf_work);

        pb::GridFunc<PrecondDataType>* gf_rcoarse
            = new pb::GridFunc<PrecondDataType>(
                *coarse_grid, bc_[0], bc_[1], bc_[2]);
        gf_rcoarse_.push_back(gf_rcoarse);
        pb::GridFunc<PrecondDataType>* gf_newv
            = new pb::GridFunc<PrecondDataType>(
                *coarse_grid, bc_[0], bc_[1], bc_[2]);
        gf_newv_.push_back(gf_newv);

        mygrid = coarse_grid;
    }
    is_precond_setup_ = true;
}

// MG V-cycle with no mask
template <class OperatorType, typename ScalarDataType, typename PrecondDataType>
void PCGSolver<OperatorType, ScalarDataType, PrecondDataType>::preconSolve(
    pb::GridFunc<PrecondDataType>& gf_v,
    const pb::GridFunc<PrecondDataType>& gf_f, const short level)
{
    assert(is_precond_setup_);

    //(*MPIdata::sout)<<"Preconditioning::mg() at level "<<level<<endl;
    short ncycl = nu1_;
    if (level == nlevels_)
    {
        ncycl = 4 > (nu1_ + nu2_) ? 4 : (nu1_ + nu2_);
    }

    pb::Lap<PrecondDataType>* myoper = precond_oper_[level];

    // pre-smoothing
    for (short it = 0; it < ncycl; it++)
    {
        myoper->jacobi(gf_v, gf_f, *gf_work_[level]);
    }

    if (level == nlevels_) return;

    // COARSE GRID CORRECTION

    // restrictions
    pb::GridFunc<PrecondDataType>* rcoarse = gf_rcoarse_[level];
    gf_work_[level]->restrict3D(*rcoarse);

    // storage functions for coarse grid
    pb::GridFunc<PrecondDataType>* newv = gf_newv_[level];

    // call mgrid solver on a coarser level
    newv->resetData();
    preconSolve(*newv, *rcoarse, level + 1);

    gf_work_[level]->extend3D(*newv);

    gf_v -= (*gf_work_[level]);

    // post-smoothing
    for (short it = 0; it < nu2_; it++)
    {
        myoper->jacobi(gf_v, gf_f, *gf_work_[level]);
    }

    if (bc_[0] != 1 || bc_[2] != 1 || bc_[2] != 1) gf_v.trade_boundaries();
}

// Left Preconditioned CG
template <class OperatorType, typename ScalarDataType, typename PrecondDataType>
bool PCGSolver<OperatorType, ScalarDataType, PrecondDataType>::solve(
    pb::GridFunc<ScalarDataType>& gf_phi,
    const pb::GridFunc<ScalarDataType>& gf_rhs)
{
    bool converged           = false;
    const pb::Grid& finegrid = gf_phi.grid();

    // initial data and residual - We assume a nonzero initial guess
    pb::GridFunc<ScalarDataType> lhs(finegrid, bc_[0], bc_[1], bc_[2]);
    // scale initial guess with epsilon
    oper_.inv_transform(gf_phi);
    // compute initial residual: r := b - Ax
    /* compute Ax */
    oper_.apply(gf_phi, lhs);
    // set r = b
    pb::GridFunc<ScalarDataType> rhs(gf_rhs);

    // transform r.h.s. to account for dielectric model
    oper_.transform(rhs);

    // apply Mehrstelllen r.h.s. if appropriate
    pb::GridFunc<ScalarDataType> res(finegrid, bc_[0], bc_[1], bc_[2]);
    oper_.rhs(rhs, res);

    // save rhs norm to compute relative norms
    const double rhs_norm = res.norm2();
    assert(!std::isnan(rhs_norm));

    // compute r = r - Ax
    res -= lhs;

    const double init_rnorm = res.norm2();
    assert(!std::isnan(init_rnorm));
    // cout<<"init_rnorm="<<init_rnorm<<endl;

    // Early return if already converged
    if (init_rnorm < tol_ * rhs_norm) return true;

    double rnorm = init_rnorm;

    /* preconditioned residual as type PrecondDataType */
    pb::GridFunc<PrecondDataType> prec_z(finegrid, bc_[0], bc_[1], bc_[2]);
    pb::GridFunc<PrecondDataType> prec_res(res);
    /* preconditioning step */
    prec_z.setZero();
    preconSolve(prec_z, prec_res, 0);
    pb::GridFunc<ScalarDataType> z(prec_z);

    // conjugate vectors
    pb::GridFunc<ScalarDataType> p(prec_z);
    pb::GridFunc<ScalarDataType> ap(p.grid(), bc_[0], bc_[1], bc_[2]);

    double rtz = res.gdot(z);

    // main loop
    for (int k = 0; k < maxiters_; k++)
    {
        // matvec: ap = A*p
        oper_.apply(p, ap);
        double ptap = p.gdot(ap);
        double alp  = rtz / ptap;

        assert(!std::isnan(alp));

        // update solution
        gf_phi.axpy(alp, p);
        // gf_phi += p*alp;
        double m_alp = -alp;
        res.axpy(m_alp, ap);
        // res -= ap*alp;

        // check for convergence
        rnorm = res.norm2();
        if (rnorm < tol_ * rhs_norm)
        {
            converged = true;
            break;
        }
        prec_z.setZero();
        prec_res.setValues(res);
        preconSolve(prec_z, prec_res, 0);
        z.setValues(prec_z);
        double rtz_new = res.gdot(z);
        double bet     = rtz_new / rtz;
        p *= bet;
        p += z;
        rtz = rtz_new;
    }
    oper_.transform(gf_phi);
    final_residual_     = rnorm;
    residual_reduction_ = rnorm / init_rnorm;

    if (fully_periodic_) gf_phi.average0();

    return converged;
}

template <class OperatorType, typename ScalarDataType, typename PrecondDataType>
bool PCGSolver<OperatorType, ScalarDataType, PrecondDataType>::solve(
    ScalarDataType* phi, ScalarDataType* rhs, const char dis)
{
    pb::GridFunc<ScalarDataType> gf_phi(oper_.grid(), bc_[0], bc_[1], bc_[2]);
    gf_phi.assign(phi, dis);

    pb::GridFunc<ScalarDataType> gf_work(oper_.grid(), bc_[0], bc_[1], bc_[2]);
    gf_work.assign(rhs, dis);

    bool converged = solve(gf_phi, gf_work);

    gf_phi.init_vect(phi, dis);

    return converged;
}

template class PCGSolver<pb::Laph4MP<double>, double, double>;
template class PCGSolver<pb::Laph4MP<double>, double, float>;
template class PCGSolver<pb::Laph4MP<float>, float, float>;
template class PCGSolver<pb::Laph4M<double>, double, double>;
template class PCGSolver<pb::Laph4M<double>, double, float>;
template class PCGSolver<pb::Laph4M<float>, float, float>;
template class PCGSolver<pb::Laph4<double>, double, double>;
template class PCGSolver<pb::Laph4<double>, double, float>;
template class PCGSolver<pb::Laph4<float>, float, float>;
template class PCGSolver<pb::Laph2<double>, double, double>;
template class PCGSolver<pb::Laph2<double>, double, float>;
template class PCGSolver<pb::Laph2<float>, float, float>;
template class PCGSolver<pb::Laph6<double>, double, double>;
template class PCGSolver<pb::Laph6<double>, double, float>;
template class PCGSolver<pb::Laph6<float>, float, float>;
template class PCGSolver<pb::Laph8<double>, double, double>;
template class PCGSolver<pb::Laph8<double>, double, float>;
template class PCGSolver<pb::Laph8<float>, float, float>;

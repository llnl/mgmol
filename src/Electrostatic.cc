// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "Electrostatic.h"
#include "Control.h"
#include "ExtendedGridOrbitals.h"
#include "GridFactory.h"
#include "GridFunc.h"
#include "Hartree.h"
#include "Hartree_CG.h"
#include "Ions.h"
#include "LocGridOrbitals.h"
#include "Mesh.h"
#include "PBdiel.h"
#include "PBdiel_CG.h"
#include "Potentials.h"
#include "Rho.h"
#include "ShiftedHartree.h"
#include "mputils.h"

Timer Electrostatic::solve_tm_("Electrostatic::solve");

Electrostatic::Electrostatic(PoissonFDtype lap_type, const short bcPoisson[3],
    const double screening_const)
    : laptype_(lap_type), poisson_solver_(nullptr)
{
    assert(bcPoisson[0] >= 0);
    assert(bcPoisson[1] >= 0);
    assert(bcPoisson[2] >= 0);

    bc_[0] = bcPoisson[0];
    bc_[1] = bcPoisson[1];
    bc_[2] = bcPoisson[2];

    Mesh* mymesh           = Mesh::instance();
    const pb::Grid& myGrid = mymesh->grid();

    // create Poisson solver
    poisson_solver_ = PoissonSolverFactory::create(
        myGrid, lap_type, bcPoisson, screening_const);

    grhoc_     = nullptr;
    diel_flag_ = false;
    grhod_     = nullptr;

    Evh_rho_         = 0.;
    Evh_rhoc_        = 0.;
    Evhold_rho_      = NAN;
    eepsilon_        = 0.;
    iterative_index_ = -1;
}

Electrostatic::~Electrostatic()
{
    assert(poisson_solver_ != nullptr);

    delete poisson_solver_;
    if (grhod_ != nullptr) delete grhod_;
    if (grhoc_ != nullptr) delete grhoc_;
}

void Electrostatic::setupInitialVh(const std::vector<POTDTYPE>& vh_init)
{
    poisson_solver_->set_vh(vh_init);

    if (iterative_index_ == -1) iterative_index_ = 0;
}

void Electrostatic::setupInitialVh(const pb::GridFunc<POTDTYPE>& vh_init)
{
    poisson_solver_->set_vh(vh_init);

    if (iterative_index_ == -1) iterative_index_ = 0;
}

template <class T>
void Electrostatic::computeVhRho(Rho<T>& rho)
{
    assert(grhoc_ != nullptr);

    Mesh* mymesh = Mesh::instance();

    RHODTYPE* work_rho;
    std::vector<std::vector<RHODTYPE>>& vrho = rho.rho_;
    if (vrho.size() > 1)
    {
        work_rho = new RHODTYPE[vrho[0].size()];
        for (int i = 0; i < (int)vrho[0].size(); i++)
            work_rho[i] = vrho[0][i] + vrho[1][i];
    }
    else
        work_rho = &vrho[0][0];

    const pb::Grid& grid = diel_flag_ ? *pbGrid_ : mymesh->grid();
    pb::GridFunc<RHODTYPE> grho(grid, bc_[0], bc_[1], bc_[2]);
    grho.assign(work_rho);

    poisson_solver_->computeVhRho(grho, *grhoc_);

    Evh_rho_  = poisson_solver_->IntVhRho();
    Evh_rhoc_ = poisson_solver_->IntVhRhoc();

    if (vrho.size() > 1) delete[] work_rho;

    iterative_index_ = rho.getIterativeIndex();
}

void Electrostatic::setupPB(
    const double e0, const double rho0, const double drho0, Potentials& pot)
{
    assert(rho0 > 0.);
    assert(grhod_ == nullptr);

    Mesh* mymesh = Mesh::instance();

    const pb::PEenv& myPEenv = mymesh->peenv();
    const pb::Grid& myGrid   = mymesh->grid();

    if (onpe0)
        (*MPIdata::sout) << "Setup PB solver with rho0=" << rho0
                         << " and beta=" << drho0 << std::endl;

    diel_flag_ = true;

    unsigned ngpts[3] = { myGrid.gdim(0), myGrid.gdim(1), myGrid.gdim(2) };
    double origin[3] = { myGrid.origin(0), myGrid.origin(1), myGrid.origin(2) };
    double cell[3]   = { myGrid.ll(0), myGrid.ll(1), myGrid.ll(2) };

    pbGrid_
        = GridFactory::createGrid(ngpts, origin, cell, laptype_, true, myPEenv);
    if (poisson_solver_ != nullptr) delete poisson_solver_;

    poisson_solver_ = PoissonSolverFactory::createDiel(
        *pbGrid_, laptype_, bc_, e0, rho0, drho0);

    if (grhoc_ != nullptr)
    {
        RHODTYPE* rhoc = new RHODTYPE[pbGrid_->size()];
        grhoc_->init_vect(rhoc, 'd');
        delete grhoc_;
        grhoc_ = nullptr;

        setupRhoc(rhoc);
        delete[] rhoc;
    }
    grhod_ = new pb::GridFunc<RHODTYPE>(*pbGrid_, bc_[0], bc_[1], bc_[2]);

    // initialize vh with last trial solution
    pb::GridFunc<POTDTYPE> gf_vh(*pbGrid_, bc_[0], bc_[1], bc_[2]);
    gf_vh.assign((pot.vh_rho()).data());
    poisson_solver_->set_vh(gf_vh);
}

// This function is only useful for Hartree problem with dielectric continuum
void Electrostatic::fillFuncAroundIons(const Ions& ions)
{
    assert(grhod_ != nullptr);
    assert(diel_flag_);

    const short shift = pbGrid_->ghost_pt();
    const int incx    = pbGrid_->inc(0);
    const int incy    = pbGrid_->inc(1);

    RHODTYPE* vv = grhod_->uu();

    const double lattice[3]
        = { pbGrid_->ll(0), pbGrid_->ll(1), pbGrid_->ll(2) };

    // here we are assuming the radius of the local potential is larger than
    // the species parameter rc
    // (otherwise we would need to track another list of ions...)
    const std::vector<Ion*>& rc_ions(ions.overlappingVL_ions());

    std::vector<Ion*>::const_iterator ion = rc_ions.begin();
    while (ion != rc_ions.end())
    {
        double rc = (*ion)->getRC();
        // Special case: silicon
        if ((*ion)->isMass28()) rc = 2.0;

        const double pi_rc = M_PI / rc;

        double xc[3];
        xc[0] = pbGrid_->start(0);

        if ((*ion)->isMassLargerThan1())
        {
#ifndef NDEBUG
            if (pbGrid_->mype_env().mytask() == 0)
            {
                (*MPIdata::sout) << " Fill func. around ion " << (*ion)->name()
                                 << " in a radius " << rc << std::endl;
            }
#endif
            for (unsigned int ix = 0; ix < pbGrid_->dim(0); ix++)
            {
                xc[1]         = pbGrid_->start(1);
                const int ix1 = (ix + shift) * incx;

                for (unsigned int iy = 0; iy < pbGrid_->dim(1); iy++)
                {
                    xc[2]         = pbGrid_->start(2);
                    const int iy1 = ix1 + (iy + shift) * incy;

                    for (unsigned int iz = 0; iz < pbGrid_->dim(2); iz++)
                    {
                        const double r = (*ion)->minimage(xc, lattice, bc_);

                        if (r < rc)
                        {
                            const double alpha = 0.2 * (1. + cos(r * pi_rc));
                            const int iz1      = iy1 + iz + shift;
                            vv[iz1] += alpha;
                        }
                        xc[2] += pbGrid_->hgrid(2);
                    }
                    xc[1] += pbGrid_->hgrid(1);
                } // end for iy
                xc[0] += pbGrid_->hgrid(0);
            } // end for ix
        }
        ion++;
    } // end loop on list of ions

    return;
}

void Electrostatic::setupRhoc(RHODTYPE* rhoc)
{
    Mesh* mymesh = Mesh::instance();

    const pb::Grid& grid = diel_flag_ ? *pbGrid_ : mymesh->grid();
    if (grhoc_ == nullptr)
    {
        grhoc_ = new pb::GridFunc<RHODTYPE>(grid, bc_[0], bc_[1], bc_[2]);
        grhoc_->assign(rhoc);
    }
    else
        grhoc_->assign(rhoc);
}

const pb::GridFunc<POTDTYPE>& Electrostatic::getVh() const
{
    return poisson_solver_->vh();
}

void Electrostatic::setup(const short max_sweeps)
{
    Control& ct           = *(Control::instance());
    const short nu1       = ct.poisson_pc_nu1;
    const short nu2       = ct.poisson_pc_nu2;
    const short max_nlevs = ct.poisson_pc_nlev;
    const float conv_tol  = ct.poisson_conv_tol;
    poisson_solver_->setup(nu1, nu2, max_sweeps, conv_tol, max_nlevs);
}

template <class T>
void Electrostatic::computeVh(const pb::GridFunc<POTDTYPE>& vh_init,
    const Ions& ions, Rho<T>& rho, Potentials& pot)
{
    poisson_solver_->set_vh(vh_init);

    computeVh(ions, rho, pot);
}

template <class T>
void Electrostatic::computeVh(const Ions& ions, Rho<T>& rho, Potentials& pot)
{
    solve_tm_.start();
#ifdef PRINT_OPERATIONS
    if (onpe0) (*MPIdata::sout) << "Electrostatic::computeVh()" << endl;
#endif

    assert(grhoc_ != nullptr);

    Mesh* mymesh             = Mesh::instance();
    const pb::PEenv& myPEenv = mymesh->peenv();

    std::vector<std::vector<RHODTYPE>>& vrho = rho.rho_;
    const int ngridpts                       = (int)vrho[0].size();

    RHODTYPE* work;
    if (vrho.size() > 1)
    {
        work = new RHODTYPE[ngridpts];
        for (int i = 0; i < ngridpts; i++)
            work[i] = vrho[0][i] + vrho[1][i];
    }
    else
        work = &vrho[0][0];

    const pb::Grid& grid = diel_flag_ ? *pbGrid_ : mymesh->grid();
    pb::GridFunc<RHODTYPE> grho(grid, bc_[0], bc_[1], bc_[2]);
    grho.assign(work);

    if (diel_flag_)
    {
        grhod_->copyFrom(&grho);
        fillFuncAroundIons(ions);

        poisson_solver_->set_rhod(grhod_);

        poisson_solver_->solve(grho, *grhoc_);

        //        int ione=1;
        int n     = ngridpts;
        eepsilon_ = LinearAlgebraUtils<MemorySpace::Host>::MPdot(
            n, work, pot.vepsilon());
        eepsilon_ = pbGrid_->vel() * myPEenv.double_sum_all(eepsilon_);
    }
    else
    {
        poisson_solver_->solve(grho, *grhoc_);
        eepsilon_ = 0.;
    }

    iterative_index_ = rho.getIterativeIndex();
    pot.setVh(poisson_solver_->vh(), iterative_index_);

    if (diel_flag_)
    {
        poisson_solver_->getVepsilon(pot.vepsilon());
    }

    Evh_rho_    = poisson_solver_->IntVhRho();
    Evh_rhoc_   = poisson_solver_->IntVhRhoc();
    Evhold_rho_ = poisson_solver_->IntVhRho_old();

    if (vrho.size() > 1) delete[] work;

    solve_tm_.stop();
}

template void Electrostatic::computeVhRho(Rho<LocGridOrbitals<ORBDTYPE>>& rho);
template void Electrostatic::computeVh(
    const Ions& ions, Rho<LocGridOrbitals<ORBDTYPE>>& rho, Potentials& pot);
template void Electrostatic::computeVh(const pb::GridFunc<POTDTYPE>& vhinit,
    const Ions& ions, Rho<LocGridOrbitals<ORBDTYPE>>& rho, Potentials& pot);

template void Electrostatic::computeVhRho(
    Rho<ExtendedGridOrbitals<ORBDTYPE>>& rho);
template void Electrostatic::computeVh(const Ions& ions,
    Rho<ExtendedGridOrbitals<ORBDTYPE>>& rho, Potentials& pot);
template void Electrostatic::computeVh(const pb::GridFunc<POTDTYPE>& vhinit,
    const Ions& ions, Rho<ExtendedGridOrbitals<ORBDTYPE>>& rho,
    Potentials& pot);

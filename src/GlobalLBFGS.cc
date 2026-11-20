// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "GlobalLBFGS.h"
#include "Control.h"
#include "Electrostatic.h"
#include "KBPsiMatrixInterface.h"
#include "MGmol.h"
#include "MasksSet.h"
#include "Mesh.h"

#include "my_prototypes.h"

GlobalLBFGS::GlobalLBFGS(LocGridOrbitals** orbitals, Ions& ions, Rho& rho,
    ConstraintSet& constraints, KBPsiMatrixInterface* g_kbpsi, Energy& energy,
    std::shared_ptr<LocalizationRegions> lrs, MasksSet& masks,
    Electrostatic& electrostat, const double dt, MGmol& strategy,
    const int local_image, const int nimages)
    : orbitals_(orbitals),
      ions_(ions),
      rho_(rho),
      constraints_(constraints),
      g_kbpsi_(g_kbpsi),
      energy_(energy),
      lrs_(lrs),
      masks_(masks),
      electrostat_(electrostat),
      ref_lrs_(lrs),
      mgmol_strategy_(strategy),
      local_image_(local_image)
{
    Mesh* mymesh           = Mesh::instance();
    const pb::Grid& mygrid = mymesh->grid();
    Control& ct            = *(Control::instance());

    const int na = (int)ions.list_ions().size();

    tau0_.resize(3 * na, NAN); // tau0[3*ia+j]
    taup_.resize(3 * na, NAN); // taup[3*ia+j]
    fion_.resize(3 * na, NAN); // fion[3*ia+j]
    pmass_.resize(na, NAN); // pmass[ia]
    atmove_.resize(na); // atmove[ia]

    gtau0_.resize(3 * na * nimages, NAN); // tau0[3*ia+j]
    gtaup_.resize(3 * na * nimages, NAN); // taup[3*ia+j]
    gfion_.resize(3 * na * nimages, NAN); // fion[3*ia+j]
    gatmove_.resize(na * nimages); // atmove[ia]

    etot_i_[0] = 10000.;
    etot_i_[1] = 10000.;
    etot_i_[2] = 10000.;

    int ia                     = 0;
    vector<Ion*>::iterator ion = ions_.list_ions().begin();
    while (ion != ions.list_ions().end())
    {
        atmove_[ia] = !(*ion)->locked();
        pmass_[ia]  = (*ion)->getMass();
        ions_names_.push_back((*ion)->name());

        ion++;
        ia++;
    }

    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    mmpi.allGatherImages(atmove_, gatmove_);

    stepper_ = new LBFGS_IonicStepper(
        dt, gatmove_, gtau0_, gtaup_, gfion_, 20, &etot_i_[0]);

    constraints_.setup(ions_);

    if (onpe0) constraints_.printConstraints((*MPIdata::sout));

    ref_masks_ = new MasksSet(lrs, ct.getMGlevels());
    vh_init_   = new pb::GridFunc<POTDTYPE>(electrostat_.getVh());

    ref_orbitals_ = new LocGridOrbitals(mygrid, mymesh->subdivx(), ct.numst,
        ct.bc, (*orbitals_)->getProjMatrices(), ref_masks_, &ref_lrs_);

    ref_orbitals_->reset(ct.getMGlevels(), ct.lap_type);
    ref_orbitals_->assign(**orbitals_);
}

GlobalLBFGS::~GlobalLBFGS()
{
    delete vh_init_;
    delete ref_masks_;
    delete stepper_;
}

void GlobalLBFGS::init(HDFrestart* h5f_file)
{
    Control& ct = *(Control::instance());

    bool flag_init = false;
    if (ct.restart_info > 0)
    {
        int status = stepper_->init(*h5f_file);
        if (status < 0) mgmol_abort();

        // if restart data for lbfgs found
        if (status == 0)
        {
            if (onpe0) (*MPIdata::sout) << "use restart info for LBFGS" << endl;
            ions_.setPositions(tau0_);
            ions_.setup();

            ions_.printPositions((*MPIdata::sout));

            // Update items that change when the ionic coordinates change
            mgmol_strategy_.moveVnuc(ions_);

            g_kbpsi_->setup(ions_, **orbitals_);

            mgmol_strategy_.updateHmatrix(**orbitals_, ions_);

            // theta = invB * Hij
            // projmatrices->updateTheta();

            flag_init = true;
        }
    }

    if (!flag_init)
    {
        // fill tau0, taup with values in ions
        ions_.getPositions(tau0_);
        taup_ = tau0_;

        // enforce constraints before 1st step
        constraints_.enforceConstraints(20);

        tau0_ = taup_;
        ions_.setPositions(taup_);
        ions_.setup();

        MGmol_MPI& mmpi = *(MGmol_MPI::instance());
        mmpi.allGatherImages(taup_, gtaup_);
        gtau0_ = gtaup_;
    }
}

int GlobalLBFGS::run1step()
{
    // compute taup
    int conv = stepper_->run();

    memcpy(&taup_[0], &gtaup_[local_image_ * taup_.size()],
        taup_.size() * sizeof(double));

    constraints_.enforceConstraints(20);

    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    mmpi.allGatherImages(taup_, gtaup_);

    // Move ions
    tau0_  = taup_;
    gtau0_ = gtaup_;

    // set ions_
    ions_.setPositions(tau0_);
    ions_.setup();

    return conv;
}

void GlobalLBFGS::computeForces()
{
    mgmol_strategy_.force(**orbitals_, ions_);

    ions_.getForces(fion_);

    if (onpe0) constraints_.printConstraintsForces((*MPIdata::sout));

    constraints_.projectOutForces(20);

    // Print forces on image (prints from PE0 by default)
    printForces((*MPIdata::sout));

    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    mmpi.allGatherImages(fion_, gfion_);
}

void GlobalLBFGS::setForces(const vector<vector<double>>& f)
{
    assert(3 * f.size() == fion_.size());

    int i = 0;
    for (int ia = 0; ia < (int)f.size(); ia++)
    {
        fion_[3 * i]     = f[ia][0];
        fion_[3 * i + 1] = f[ia][1];
        fion_[3 * i + 2] = f[ia][2];
        i++;
    }

    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    mmpi.allGatherImages(fion_, gfion_);
}

void GlobalLBFGS::printForces(ostream& os, const int root)
{
    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    if (mmpi.mype() == root)
    {
        os << "Stepper Forces:" << endl;
        double maxf  = 0.;
        const int na = (int)ions_names_.size();
        for (int i = 0; i < na; i++)
        {
            os << "## ";
            os << " ";
            os << setw(4) << ions_names_[i] << setiosflags(ios::right)
               << setw(10) << setprecision(4) << fixed << tau0_[3 * i + 0]
               << setw(10) << tau0_[3 * i + 1] << setw(10) << tau0_[3 * i + 2]
               << setprecision(3) << scientific << setw(12) << fion_[3 * i + 0]
               << setw(12) << fion_[3 * i + 1] << setw(12) << fion_[3 * i + 2]
               << endl;

            double fi = sqrt(fion_[3 * i + 0] * fion_[3 * i + 0]
                             + fion_[3 * i + 0] * fion_[3 * i + 1]
                             + fion_[3 * i + 0] * fion_[3 * i + 2]);
            maxf      = (fi > maxf) ? fi : maxf;
        }
        os << "global_lbfgs: Max. Force on image: " << maxf << endl;
    }
}

bool GlobalLBFGS::checkTolForces(const double tol)
{
    assert(3 * gatmove_.size() == gfion_.size());

    const int na = (int)gatmove_.size();
    double f2    = 0.;
    for (int i = 0; i < na; i++)
    {
        if (atmove_[i])
            f2 = max(f2, gfion_[3 * i + 0] * gfion_[3 * i + 0]
                             + gfion_[3 * i + 1] * gfion_[3 * i + 1]
                             + gfion_[3 * i + 2] * gfion_[3 * i + 2]);
    }
    if (onpe0)
        (*MPIdata::sout) << setprecision(3) << scientific
                         << "global_lbfgs: Max. |force| for image = "
                         << sqrt(f2) << endl;
    int flag_convF           = (f2 < tol * tol);
    Mesh* mymesh             = Mesh::instance();
    const pb::PEenv& myPEenv = mymesh->peenv();
    // make sure flag_convF is the same on all pes
    MPI_Bcast(&flag_convF, 1, MPI_INT, 0, myPEenv.comm());

    return (flag_convF);
}

int GlobalLBFGS::quenchElectrons(const int itmax, double& etot)
{
    etot_i_[0] = etot_i_[1];
    etot_i_[1] = etot_i_[2];

    int ret = mgmol_strategy_.quench(*orbitals_, ions_, itmax, 0, etot_i_[2]);

    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    mmpi.allreduceImages(&etot_i_[2], 1, MPI_SUM);

    etot = etot_i_[2];
    return ret;
}

void GlobalLBFGS::addImageInteractionEnergy(const double eii)
{
    etot_i_[2] += eii;
    if (onpe0)
    {
        (*MPIdata::sout) << "global_lbfgs: add NEB energy. Total energy = "
                         << etot_i_[2] << endl;
    }
}

void GlobalLBFGS::updateRefs()
{
    Control& ct = *(Control::instance());
    if (!stepper_->check_last_step_accepted())
    {
        if (onpe0)
            (*MPIdata::sout)
                << "global_lbfgs: Reset orbitals to reference orbitals "
                << endl;
        (*orbitals_)->assign(*ref_orbitals_);
        electrostat_.setupInitialVh(*vh_init_);
    }
    else
    {
        // update references
        if (ct.lr_update)
        {
            ref_lrs_    = lrs_;
            *ref_masks_ = masks_;
        }
        ref_orbitals_->reset(ct.getMGlevels(), ct.lap_type);
        if (onpe0)
            (*MPIdata::sout)
                << "global_lbfgs: Update reference orbitals " << endl;
        ref_orbitals_->assign(**orbitals_);
        vh_init_->assign(electrostat_.getVh(), 'd');
    }
}

void GlobalLBFGS::setQuenchTol() const
{
    Control& ct   = *(Control::instance());
    ct.tol_energy = 0.1 * stepper_->etol();
    if (onpe0)
        (*MPIdata::sout) << setprecision(12) << fixed
                         << "global_lbfgs: Set SC convergence criterion to "
                         << ct.tol_energy << endl;
}

void GlobalLBFGS::dumpRestart()
{
    Control& ct              = *(Control::instance());
    Mesh* mymesh             = Mesh::instance();
    const pb::PEenv& myPEenv = mymesh->peenv();
    const pb::Grid& mygrid   = mymesh->grid();
    int gdim[3] = { mygrid.gdim(0), mygrid.gdim(1), mygrid.gdim(2) };

    HDFrestart h5file(
        string(ct.out_restart_file), myPEenv, gdim, ct.out_restart_file_type);

    if (onpe0) (*MPIdata::sout) << "global_lbfgs: dumpRestart()" << endl;
    mgmol_strategy_.write_hdf5(h5file, rho_.rho_, ions_, **orbitals_, lrs_);
    stepper_->write_hdf5(h5file);
}

void GlobalLBFGS::updatePotAndMasks()
{
    Control& ct = *(Control::instance());
    if (!stepper_->check_last_step_accepted())
        electrostat_.setupInitialVh(*vh_init_);

    // Update items that change when the ionic coordinates change
    mgmol_strategy_.moveVnuc(ions_);

    // if( ct.lr_update ){
    //    mgmol_strategy_.update_masks();
    //}
    mgmol_strategy_.move_orbitals(orbitals_);

    g_kbpsi_->setup(ions_, **orbitals_);

    rho_.setup(ct.orbital_type, (*orbitals_)->getGlobalIndexes());

    const short update_dm = (ct.OuterSolver() == OuterSolverType::ABPG) ? 1 : 0;
    mgmol_strategy_.updateProjectedMatrices(**orbitals_, ions_, update_dm);
}

bool GlobalLBFGS::lbfgsLastStepNotAccepted() const
{
    return !stepper_->check_last_step_accepted();
}

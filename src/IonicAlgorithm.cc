// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "IonicAlgorithm.h"
#include "Control.h"
#include "MGmol.h"
#include "MasksSet.h"
#include "Mesh.h"

using namespace std;

template <class OrbitalsType>
IonicAlgorithm<OrbitalsType>::IonicAlgorithm(OrbitalsType** orbitals,
    Ions& ions, Rho<OrbitalsType>& rho, ConstraintSet& constraints,
    std::shared_ptr<LocalizationRegions> lrs, MasksSet& masks,
    MGmol<OrbitalsType>& strategy)
    : orbitals_(orbitals),
      ions_(ions),
      rho_(rho),
      constraints_(constraints),
      lrs_(lrs),
      masks_(masks),
      mgmol_strategy_(strategy),
      stepper_(nullptr),
      ions_names_(ions_.getLocalNames()),
      tau0_(ions_.getTau0()),
      taup_(ions_.getTaup()),
      fion_(ions_.getFion()),
      pmass_(ions_.getPmass()),
      atmove_(ions_.getAtmove()),
      gid_(ions_.getGids())
{
    assert(atmove_.size() == pmass_.size());
    assert(3 * atmove_.size() == tau0_.size());
}

template <class OrbitalsType>
void IonicAlgorithm<OrbitalsType>::registerStepper(IonicStepper* stepper)
{
    stepper_ = stepper;
}

template <class OrbitalsType>
void IonicAlgorithm<OrbitalsType>::init(HDFrestart* h5f_file)
{
    printWithTimeStamp("IonicAlgorithm<OrbitalsType>::init()...", cout);

    Control& ct = *(Control::instance());

    bool flag_init = false;
    if (ct.restart_info > 0)
    {
        int status = stepper_->init(*h5f_file);
        if (status < 0) mgmol_abort();

        // if restart data for lbfgs found
        if (status == 0)
        {
            if (onpe0)
                (*MPIdata::sout)
                    << "use restart info for IonicAlgorithm" << endl;
            ions_.setPositionsToTau0();
            ions_.setup();

            ions_.printPositions((*MPIdata::sout));

            // Update items that change when the ionic coordinates change
            mgmol_strategy_.moveVnuc(ions_);

            flag_init = true;
        }
    }

    if (!flag_init)
    {
        // fill tau0, taup with values in ions
        ions_.setTau0();
        taup_ = tau0_;

        // enforce constraints before 1st step
        constraints_.addConstraints(ions_);
        constraints_.setup(ions_);
        constraints_.enforceConstraints(20);

        tau0_ = taup_;
        ions_.setPositionsToTau0();

        ions_.setup();
    }
}

template <class OrbitalsType>
int IonicAlgorithm<OrbitalsType>::quenchElectrons(const int itmax, double& etot)
{
    int ret = mgmol_strategy_.quench(**orbitals_, ions_, itmax, 0, etot);

    return ret;
}

template <class OrbitalsType>
void IonicAlgorithm<OrbitalsType>::setupConstraints()
{
    // constraints need to be added again and setup as atoms
    // may have moved and local atoms are not the same anymore
    constraints_.addConstraints(ions_);

    constraints_.setup(ions_);
}

template <class OrbitalsType>
int IonicAlgorithm<OrbitalsType>::run1step()
{
    // compute taup
    int conv = stepper_->run();

    ions_.updateTaupInteractingIons();

    // enforce constraints on taup
    constraints_.enforceConstraints(20);

    // Move ions
    tau0_ = taup_;

    // set ions_
    ions_.setPositionsToTau0();
    ions_.setVelocitiesToVel();

    ions_.setup();

    // constraints need to be setup again as atoms may have moved
    setupConstraints();

    ions_.printPositionsGlobal((*MPIdata::sout));

    return conv;
}

template <class OrbitalsType>
void IonicAlgorithm<OrbitalsType>::computeForces()
{
    mgmol_strategy_.force(**orbitals_, ions_);

    ions_.updateForcesInteractingIons();

    constraints_.printConstraintsForces((*MPIdata::sout));

    constraints_.projectOutForces();

    // Print forces
    ions_.printForcesGlobal((*MPIdata::sout));
}

template <class OrbitalsType>
void IonicAlgorithm<OrbitalsType>::setForces(const vector<vector<double>>& f)
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

    constraints_.projectOutForces();
}

template <class OrbitalsType>
bool IonicAlgorithm<OrbitalsType>::checkTolForces(const double tol)
{
    assert(3 * atmove_.size() == fion_.size());

    const int na = (int)atmove_.size();
    double f2    = 0.;
    for (int i = 0; i < na; i++)
    {
        if (atmove_[i])
            f2 = max(f2, fion_[3 * i + 0] * fion_[3 * i + 0]
                             + fion_[3 * i + 1] * fion_[3 * i + 1]
                             + fion_[3 * i + 2] * fion_[3 * i + 2]);
    }

    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    mmpi.allreduce(&f2, 1, MPI_MAX);

    if (onpe0)
        (*MPIdata::sout) << setprecision(3) << scientific
                         << "Max. |force| = " << sqrt(f2) << endl;
    short flag_convF = (f2 < tol * tol);

    // make sure flag_convF is the same on all pes
    mmpi.bcast(&flag_convF, 1);

    return (flag_convF);
}

template <class OrbitalsType>
void IonicAlgorithm<OrbitalsType>::dumpRestart()
{
    Control& ct              = *(Control::instance());
    Mesh* mymesh             = Mesh::instance();
    const pb::PEenv& myPEenv = mymesh->peenv();
    const pb::Grid& mygrid   = mymesh->grid();
    unsigned gdim[3] = { mygrid.gdim(0), mygrid.gdim(1), mygrid.gdim(2) };

    HDFrestart h5file(
        string(ct.out_restart_file), myPEenv, gdim, ct.out_restart_file_type);

    if (onpe0) (*MPIdata::sout) << "IonicAlgorithm: dumpRestart()" << endl;
    mgmol_strategy_.write_hdf5(h5file, rho_.rho_, ions_, **orbitals_, lrs_);

    // write information specific to stepper
    stepper_->write_hdf5(h5file);
}

template <class OrbitalsType>
void IonicAlgorithm<OrbitalsType>::updatePotAndMasks()
{
    // Update items that change when the ionic coordinates change
    mgmol_strategy_.moveVnuc(ions_);

    // Control& ct = *(Control::instance());
    // if( ct.lr_update )
    //{
    //    mgmol_strategy_.update_masks();
    //}
    mgmol_strategy_.move_orbitals(orbitals_);
}

template class IonicAlgorithm<LocGridOrbitals<ORBDTYPE>>;
template class IonicAlgorithm<ExtendedGridOrbitals<ORBDTYPE>>;

// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "MD_IonicStepper.h"
#include "Control.h"
#include "MGmol_MPI.h"
#include "MPIdata.h"
#include "hdf5.h"

#include <cassert>
#include <iomanip>
#include <iostream>
#include <math.h>
using namespace std;

const double scmass = 1822.89;

// kb_au=8.617343e-5 [eV/K] / 27.211608 [eV/Ha]
const double kb_au = 3.16678939e-06; // [Ha/K]

// Input: assume taum is initialized with values of velocities
MD_IonicStepper::MD_IonicStepper(const double dt, vector<short>& atmove,
    vector<double>& tau0, vector<double>& taup, vector<double>& taum,
    vector<double>& fion, const vector<double>& pmass,
    vector<unsigned short>& rand_states)
    : IonicStepper(dt, atmove, tau0, taup),
      taum_(taum),
      pmass_(pmass),
      fion_(fion),
      rand_states_(rand_states)
{
    assert(3 * pmass.size() == tau0.size());
    assert(3 * atmove.size() == tau0.size());
    assert(taup.size() == tau0.size());

    ttherm_ = 0; // default is off

    taum_.resize(tau0.size(), NAN);

    for (int i = 0; i < (int)pmass.size(); i++)
        assert(pmass[i] > 0.);

    // unset parameters set to invalid values
    tkel_    = -1.;
    thtime_  = -1.;
    thwidth_ = -1.;
    gamma_   = -1.;

    int size_tau = (int)tau0_.size();
    int ione     = 1;
    double one   = 1.;
    if (dt > 0.0)
    {
        if (onpe0)
            (*MPIdata::sout) << "MD_IonicStepper() with dt =  " << dt_ << endl;
        // taum_ was initialized with velocities
        // set taum_ to displacements = -dt*vel
        double alpha = -1. * dt;
        DSCAL(&size_tau, &alpha, &taum_[0], &ione);

        // initialize taup_ (to define velocities)
        double minus_one = -1.;
        taup_            = tau0_;
        DAXPY(&size_tau, &minus_one, &taum_[0], &ione, &taup_[0], &ione);

        // Now set taum_ to previous positions: tau0_ - dt*vel
        DAXPY(&size_tau, &one, &tau0_[0], &ione, &taum_[0], &ione);
    }
}

void MD_IonicStepper::setThermostat(const int ttherm, const double tkel,
    const double thtime, const double thwidth, const int nconstraints)
{
    assert(ttherm == 0 || ttherm == 1 || ttherm == 2 || ttherm == 3);
    if (ttherm > 0) assert(thtime > 0.);

    ttherm_  = ttherm;
    tkel_    = tkel;
    thtime_  = thtime;
    thwidth_ = thwidth;
    gamma_   = 1. / (2. * thtime_);

    ndofs_ -= nconstraints;

    Control& ct = *(Control::instance());
    if (onpe0 && ct.verbose > 0)
        (*MPIdata::sout) << "MD_IonicStepper::setThermostat(), ndofs= "
                         << ndofs_ << endl;
}

double MD_IonicStepper::kineticEnergy()
{
    assert(dt_ > 0.);

    double ke           = 0.;
    const double inv2dt = 1. / (2. * dt_);
    const int na        = (int)atmove_.size();
    for (int ia = 0; ia < na; ia++)
        if (atmove_[ia])
        {
            double v2 = 0.;
            for (short j = 0; j < 3; j++)
            {
                const int i    = 3 * ia + j;
                const double v = (taup_[i] - taum_[i]) * inv2dt;
                v2 += v * v;
            }
            ke += pmass_[ia] * v2;
        }

    MGmol_MPI& mmpi(*(MGmol_MPI::instance()));
    mmpi.allreduce(&ke, 1, MPI_SUM);

    return 0.5 * ke;
}

double MD_IonicStepper::temperature()
{
    if (ndofs_ <= 0) return 0.;
    return kineticEnergy() / (0.5 * ndofs_ * kb_au);
}

int MD_IonicStepper::init(HDFrestart& /*h5f_file*/) { return 0; }

int MD_IonicStepper::write_hdf5(HDFrestart& h5f_file)
{
    hid_t file_id = h5f_file.file_id();
    if (file_id < 0) return 0;

#ifdef MGMOL_USE_HDF5P
    const bool write_flag = (onpe0 || h5f_file.useHdf5p());
#else
    const bool write_flag = onpe0;
#endif

    if (write_flag)
    {
        int status = writePositions(h5f_file);
        if (status < 0) return status;

        if (dt_ > 0.)
        {
            writeVelocities(h5f_file);
            std::string datasetname("/Ionic_previous_positions");
            writeAtomicFields(h5f_file, taum_, datasetname, true);
        }
        else
        {
            std::string datasetname("/Ionic_velocities");
            writeAtomicFields(h5f_file, taum_, datasetname, true);
        }
        std::string datasetname("/Ionic_forces");
        writeAtomicFields(h5f_file, fion_, datasetname, true);
    }

    //
    // save random state --- this may not be necessary since
    // ions.writeRandomStates is called by write_hdf5() in restart.cc
    // prior to this call.
    //
    if (write_flag)
    {
        writeRandomStates(h5f_file, rand_states_);
    }

    return 0;
}

///////////////////////////////////////////////////////////////////////////////////

void MD_IonicStepper::addLangevinForces()
{
    if (ttherm_ != 2) return;
    if (dt_ <= 0.) return;

    if (onpe0)
        (*MPIdata::sout) << "Langevin Thermostat for target temperature "
                         << tkel_ << " [Kelvin]" << endl;
    const double alpha = sqrt(2. * gamma_ * kb_au * tkel_ / dt_);
    const int na       = (int)atmove_.size();
    for (int ia = 0; ia < na; ia++)
    {
        if (atmove_[ia])
        {
            const double sqrtmass = sqrt(pmass_[ia]);

            // draw random variables of average 0 and variance 1
            int idx               = 3 * ia;
            unsigned short* state = &rand_states_[idx];
            for (int j = 0; j < 3; j++)
            {

                double rdf = erand48(state) + erand48(state) + erand48(state)
                             + erand48(state) + erand48(state) + erand48(state)
                             + erand48(state) + erand48(state) + erand48(state)
                             + erand48(state) + erand48(state) + erand48(state)
                             - 6.0;

                fion_[idx + j] += alpha * sqrtmass * rdf;
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////

int MD_IonicStepper::run()
{
    Control& ct = *(Control::instance());

    //(*MPIdata::sout)<<"MD_IonicStepper::run()"<<endl;
    const int na = (int)atmove_.size();
    assert((int)fion_.size() == 3 * na);
    for (int ia = 0; ia < na; ia++)
        assert(pmass_[ia] > 0.1);

    double lambda = 1.;

    if (dt_ > 0.0)
    {
        if (ttherm_ == 1)
        {
            if (onpe0)
                (*MPIdata::sout)
                    << "Berendsen Thermostat for target temperature " << tkel_
                    << " [Kelvin]" << endl;
            double tempm = getTemperatureAtTimeMinusHalf();

            if (ndofs_ > 0)
            {
                if (onpe0)
                    (*MPIdata::sout) << "Current temperature " << tempm
                                     << " [Kelvin]" << endl;
                lambda = sqrt(1. + dt_ * (tkel_ / tempm - 1.) / thtime_);
            }
        }
        else if (ttherm_ == 2)
        {
            // assumes Langevin force has been added into fion_ already

            const double factor = 1. / (1. + 0.5 * gamma_ * dt_);
            if (onpe0)
                (*MPIdata::sout) << "Langevin Thermostat with gamma_=" << gamma_
                                 << ", factor=" << factor << endl;

            for (int ia = 0; ia < na; ia++)
            {
                if (atmove_[ia])
                {
                    assert(pmass_[ia] > 0.);
                    const double invmass = 1. / pmass_[ia];
                    // Brunger, Brooks, Karplus,
                    // Chem. Phys. Lett. 105, 495 (1982)
                    for (int j = 0; j < 3; j++)
                    {
                        taup_[3 * ia + j]
                            = factor
                              * (2. * tau0_[3 * ia + j] - taum_[3 * ia + j]
                                  + dt_ * dt_ * invmass * fion_[3 * ia + j]
                                  + 0.5 * dt_ * gamma_ * taum_[3 * ia + j]);
                    }
                }
                else
                {
                    for (int j = 0; j < 3; j++)
                    {
                        taup_[3 * ia + j] = tau0_[3 * ia + j];
                    }
                }
            }
        }
        else if (ttherm_ == 3)
        {
            if (onpe0)
                (*MPIdata::sout) << "SCALING Thermostat for target temperature "
                                 << tkel_ << " [Kelvin]" << endl;
            double tempm = getTemperatureAtHalfStep();
            // if ( onpe0 )
            //{
            //    (*MPIdata::sout)<<"SCALING Thermostat, tempm = "<<tempm
            //                    <<", tkel = "<<tkel_<<endl;
            //    (*MPIdata::sout)<<"SCALING Thermostat, thwidth_ = "<<thwidth_
            //                    <<", thtime_ = "<<thtime_<<endl;
            //}
            assert(thwidth_ > 0.);
            const double eta = tanh((tempm - tkel_) / thwidth_) / thtime_;
            lambda           = (1.0 - eta * fabs(dt_));
            if (onpe0 && ct.verbose > 2)
                (*MPIdata::sout) << "SCALING Thermostat, eta = " << eta
                                 << ", lambda = " << lambda << endl;
        }

        // Verlet with velocity rescaling (Berendsen)
        if (ttherm_ != 2) updateWithVelocityScaling(lambda);
    }
    else
    {
        for (int ia = 0; ia < na; ia++)
        {
            for (int j = 0; j < 3; j++)
            {
                taup_[3 * ia + j] = tau0_[3 * ia + j];
            }
        }
    }

    return 0;
}

void MD_IonicStepper::printVelocities(ostream& os) const
{
    if (onpe0)
    {
        const double invscmass = 1. / scmass;

        os << endl << " IONIC POSITIONS AND VELOCITIES (at t+1/2):" << endl;
        os << setw(10) << "Mass" << setw(10) << "X" << setw(10) << "Y"
           << setw(10) << "Z" << setw(10) << "Vx" << setw(10) << "Vy"
           << setw(10) << "Vz" << endl;
        os.setf(ios::right, ios::adjustfield);
        os.setf(ios::fixed, ios::floatfield);
        const double invdt = 1. / dt_;
        const int na       = (int)atmove_.size();
        for (int ia = 0; ia < na; ia++)
        {
            os << "## ";
            os << setprecision(4) << fixed << setw(10) << pmass_[ia] * invscmass
               << setw(10) << 0.5 * (taup_[3 * ia] + tau0_[3 * ia]) << setw(10)
               << 0.5 * (taup_[3 * ia + 1] + tau0_[3 * ia + 1]) << setw(10)
               << 0.5 * (taup_[3 * ia + 2] + tau0_[3 * ia + 2])
               << setprecision(3) << scientific << setw(12)
               << invdt * (taup_[3 * ia] - tau0_[3 * ia]) << setw(12)
               << invdt * (taup_[3 * ia + 1] - tau0_[3 * ia + 1]) << setw(12)
               << invdt * (taup_[3 * ia + 2] - tau0_[3 * ia + 2]) << endl;
        }
    }
}

void MD_IonicStepper::getVelocities(vector<float>& velocities) const
{
    velocities.clear();

    const double invdt = 1. / dt_;
    const int na       = (int)taup_.size();

    for (int ia = 0; ia < na; ia++)
    {
        velocities.push_back((float)(invdt * (taup_[3 * ia] - tau0_[3 * ia])));
        velocities.push_back(
            (float)(invdt * (taup_[3 * ia + 1] - tau0_[3 * ia + 1])));
        velocities.push_back(
            (float)(invdt * (taup_[3 * ia + 2] - tau0_[3 * ia + 2])));
    }
}

double MD_IonicStepper::etol(void) const { return 0.0; }

void MD_IonicStepper::updateTau()
{
    if (dt_ > 0.)
    {
        taum_ = tau0_;
    }
    tau0_ = taup_;
    if (dt_ > 0.)
    {
        // update taup_ to be able to compute velocity:
        // taup_ <- tau0_-taum_
        int size_tau = (int)tau0_.size();
        int ione     = 1;
        double alpha = 1.;
        DAXPY(&size_tau, &alpha, &tau0_[0], &ione, &taup_[0], &ione);
        alpha = -1.;
        DAXPY(&size_tau, &alpha, &taum_[0], &ione, &taup_[0], &ione);
    }
}

void MD_IonicStepper::updateWithVelocityScaling(const double lambda)
{
    const int na = (int)atmove_.size();

    for (int ia = 0; ia < na; ia++)
    {
        for (int j = 0; j < 3; j++)
        {
            taup_[3 * ia + j] = tau0_[3 * ia + j];
        }
        if (atmove_[ia])
        {
            const double invmass = 1. / pmass_[ia];
            for (int j = 0; j < 3; j++)
            {
                taup_[3 * ia + j]
                    += lambda
                       * (tau0_[3 * ia + j] - taum_[3 * ia + j]
                           + dt_ * dt_ * invmass * fion_[3 * ia + j]);
            }
        }
    }
}

double MD_IonicStepper::getTemperatureAtTimeMinusHalf()
{
    const int na = (int)atmove_.size();

    double ekinm = 0.0;
    for (int ia = 0; ia < na; ia++)
        if (atmove_[ia])
        {
            for (int j = 0; j < 3; j++)
            {
                double tmp = tau0_[3 * ia + j] - taum_[3 * ia + j];
                ekinm += tmp * tmp * pmass_[ia];
            }
        }
    ekinm = 0.5 * ekinm / (dt_ * dt_);

    // get global intermediate energy
    MGmol_MPI& mmpi(*(MGmol_MPI::instance()));
    double etmp = ekinm;
    mmpi.allreduce(&etmp, 1, MPI_SUM);

    // for start with no velocities, cannot use T(t-0.5*dt)!!
    if (etmp < 1.e-3)
    {
        double ekinf = 0.;
        for (int ia = 0; ia < na; ia++)
            if (atmove_[ia])
            {
                const double invmass = 1. / pmass_[ia];
                for (int j = 0; j < 3; j++)
                {
                    double tmp = dt_ * invmass * fion_[3 * ia + j];
                    ekinf += tmp * tmp * pmass_[ia];
                }
            }
        ekinm += 0.5 * ekinf;

        // get global kinetic energy
        mmpi.allreduce(&ekinm, 1, MPI_SUM);
    }
    else
    {
        ekinm = etmp;
    }
    double tempm = 0.;
    if (ndofs_ > 0)
    {
        tempm = (ekinm / (0.5 * kb_au * (double)ndofs_));
    }

    return tempm;
}

double MD_IonicStepper::getTemperatureAtHalfStep()
{
    const int na = (int)atmove_.size();
    vector<double> vhalf(3 * na);
    const double invdt = 1. / dt_;

    for (int ia = 0; ia < na; ia++)
        if (atmove_[ia])
        {
            const double invmass = 1. / pmass_[ia];
            for (int j = 0; j < 3; j++)
            {
                double tmp = (tau0_[3 * ia + j] - taum_[3 * ia + j]) * invdt;
                vhalf[3 * ia + j] = tmp + 0.5 * invmass * fion_[3 * ia + j];
            }
        }

    double ekin = 0.;
    for (int ia = 0; ia < na; ia++)
        if (atmove_[ia])
        {
            for (int j = 0; j < 3; j++)
            {
                ekin += pmass_[ia] * vhalf[3 * ia + j] * vhalf[3 * ia + j];
            }
        }
    ekin *= 0.5;

    // get global kinetic energy
    MGmol_MPI& mmpi(*(MGmol_MPI::instance()));
    mmpi.allreduce(&ekin, 1, MPI_SUM);

    double tempm = 0.;
    if (ndofs_ > 0)
    {
        tempm = (ekin / (0.5 * kb_au * (double)ndofs_));
    }

    return tempm;
}

// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "ConstraintSet.h"
#include "Control.h"
#include "DFTsolver.h"
#include "DMStrategy.h"
#include "Energy.h"
#include "ExtendedGridOrbitals.h"
#include "Hamiltonian.h"
#include "Ions.h"
#include "LocGridOrbitals.h"
#include "LocalizationRegions.h"
#include "MD_IonicStepper.h"
#include "MDfiles.h"
#include "MGmol.h"
#include "MGmol_blas1.h"
#include "MPIdata.h"
#include "MasksSet.h"
#include "Mesh.h"
#include "OrbitalsExtrapolation.h"
#include "OrbitalsExtrapolationFactory.h"
#include "Potentials.h"
#include "ProjectedMatricesMehrstellen.h"
#include "ProjectedMatricesSparse.h"
#include "Rho.h"
#include "SpreadsAndCenters.h"
#include "mgmol_Signal.h"
#include "tools.h"

#ifdef MGMOL_HAS_LIBROM
#include "KBPsiMatrixSparse.h"
#include "PinnedH2O.h"
#include "rom_workflows.h"
#endif

#include <sstream>
#include <string>
#include <vector>

Timer md_iterations_tm("md_iterations");
Timer md_tau_tm("md_tau");
Timer md_moveVnuc_tm("md_moveVnuc");
Timer md_updateMasks_tm("md_updateMasks");
Timer md_extrapolateOrbitals_tm("md_extrapolateOrbitals");
Timer md_updateDMandEnergy_tm("md_updateDMandEnergy");

#define DUMP_MAX_NUM_TRY 5

template <class OrbitalsType>
void MGmol<OrbitalsType>::moveVnuc(Ions& ions)
{
    md_moveVnuc_tm.start();

    Potentials& pot = hamiltonian_->potential();

    // save Hartree potential internally
    pot.backupVh();

    // Update items that change when the ionic coordinates change
    pot.axpVcompToVh(1.);
    setupPotentials(ions);
    pot.axpVcompToVh(-1.);

    proj_matrices_->setHiterativeIndex(-1, -1);
    hamiltonian_->setHlOutdated();

    md_moveVnuc_tm.stop();
}

template <class OrbitalsType>
void MGmol<OrbitalsType>::preWFextrapolation()
{
    Control& ct = *(Control::instance());
    if (ct.OuterSolver() == OuterSolverType::ABPG
        || ct.OuterSolver() == OuterSolverType::NLCG)
    {
        if (ct.dm_mix < 1. || !ct.fullyOccupied()) proj_matrices_->stripDM();
    }
}

template <class OrbitalsType>
void MGmol<OrbitalsType>::postWFextrapolation(OrbitalsType* orbitals)
{
    Control& ct = *(Control::instance());

    orbitals->computeGramAndInvS();

    if (ct.OuterSolver() == OuterSolverType::ABPG
        || ct.OuterSolver() == OuterSolverType::NLCG)
    {
        if (ct.dm_mix < 1. || !ct.fullyOccupied())
            proj_matrices_->dressupDM();
        else
            proj_matrices_->setDMto2InvS();
    }
}

template <class OrbitalsType>
void MGmol<OrbitalsType>::extrapolate_centers(bool small_move)
{
    assert(lrs_);

    Control& ct = *(Control::instance());
    if (onpe0 && ct.verbose > 0)
        os_ << "Extrapolate LR centers..." << std::endl;

    if (small_move && ct.lrs_extrapolation == 2)
    {
        lrs_->extrapolateCentersQuadratic();
    }
    else
    {
        // for large moves, extrapolation does nothing
        // but save current centers
        if (ct.lrs_extrapolation == 1 || ct.lrs_extrapolation == 2)
            lrs_->extrapolateCentersLinear(small_move);

        else if (ct.lrs_extrapolation == 10)
            lrs_->extrapolateCentersVerlet(small_move);
    }
}

template <class OrbitalsType>
OrbitalsType* MGmol<OrbitalsType>::new_orbitals_with_current_LRs(bool setup)
{
    Mesh* mymesh           = Mesh::instance();
    const pb::Grid& mygrid = mymesh->grid();
    Control& ct            = *(Control::instance());

    if (ct.verbose > 1 && onpe0)
        os_ << "Build new orbitals with new masks..." << std::endl;

    if (ct.lr_update) update_masks();

    // need to build new orbitals as masks have changed
    OrbitalsType* new_orbitals = new OrbitalsType("NewMasks", mygrid,
        mymesh->subdivx(), ct.numst, ct.bcWF, proj_matrices_.get(), lrs_,
        currentMasks_.get(), corrMasks_.get(), local_cluster_.get(), setup);

    return new_orbitals;
}

template <class OrbitalsType>
void MGmol<OrbitalsType>::update_orbitals_LRs(OrbitalsType** orbitals)
{
    OrbitalsType* new_orbitals = new_orbitals_with_current_LRs();
    new_orbitals->assign(**orbitals);
    delete (*orbitals);
    (*orbitals) = new_orbitals;

    (*orbitals)->applyMask();
}

template <class OrbitalsType>
void MGmol<OrbitalsType>::extrapolate_orbitals(OrbitalsType** orbitals)
{
    md_extrapolateOrbitals_tm.start();

    OrbitalsType* new_orbitals = new_orbitals_with_current_LRs();

    orbitals_extrapol_->extrapolate_orbitals(orbitals, new_orbitals);

    (*orbitals)->incrementIterativeIndex();

    md_extrapolateOrbitals_tm.stop();
}

template <class OrbitalsType>
void MGmol<OrbitalsType>::move_orbitals(OrbitalsType** orbitals)
{
    Control& ct = *(Control::instance());

    if (onpe0 && ct.verbose > 1) os_ << "Move orbitals..." << std::endl;

    OrbitalsType* new_orbitals = new_orbitals_with_current_LRs();

    // copy old data
    new_orbitals->assign(**orbitals);

    delete *orbitals;
    *orbitals = new_orbitals;

    if (ct.isLocMode())
    {
        (*orbitals)->applyMask();
        (*orbitals)->normalize();
    }
    (*orbitals)->incrementIterativeIndex();
}

template <class OrbitalsType>
int MGmol<OrbitalsType>::update_masks()
{
    assert(lrs_);

    Control& ct = *(Control::instance());
    if (!ct.isLocMode()) return 0;

    md_updateMasks_tm.start();

    if (onpe0 && ct.verbose > 1) os_ << "update_masks()" << std::endl;

    currentMasks_->update(lrs_);
    corrMasks_->update(lrs_);

    md_updateMasks_tm.stop();

    return 0;
}

void checkMaxForces(const std::vector<double>& fion,
    const std::vector<short>& atmove, std::ostream& os)
{
    assert(3 * atmove.size() == fion.size());

    const int na = (int)atmove.size();
    double f2    = 0.;
    for (int i = 0; i < na; i++)
    {
        if (atmove[i])
            f2 = std::max(f2, fion[3 * i + 0] * fion[3 * i + 0]
                                  + fion[3 * i + 1] * fion[3 * i + 1]
                                  + fion[3 * i + 2] * fion[3 * i + 2]);
    }
    if (onpe0)
        os << "Max. |force| = " << std::setprecision(4) << std::scientific
           << std::sqrt(f2) << std::endl;
}

template <class OrbitalsType>
int MGmol<OrbitalsType>::dumpMDrestartFile(OrbitalsType& orbitals, Ions& ions,
    Rho<OrbitalsType>& rho, const bool write_extrapolated_wf, const short count)
{
    MGmol_MPI& mmpi(*(MGmol_MPI::instance()));
    Control& ct              = *(Control::instance());
    Mesh* mymesh             = Mesh::instance();
    const pb::PEenv& myPEenv = mymesh->peenv();
    const pb::Grid& mygrid   = mymesh->grid();
    const unsigned gdim[3] = { mygrid.gdim(0), mygrid.gdim(1), mygrid.gdim(2) };

    std::string filename(std::string(ct.out_restart_file));
    // add an integer corresponding to attempt number/count
    // to allow several attempts at creating and writing file
    if (count > 0)
    {
        std::stringstream s;
        s << count;
        filename += s.str();
    }

    HDFrestart h5file(filename, myPEenv, gdim, ct.out_restart_file_type);

    OrbitalsType previous_orbitals("ForDumping", orbitals, false);
    if (!orbitals_extrapol_->getRestartData(previous_orbitals))
        previous_orbitals.assign(orbitals);

    // write all restart info in HDF5 file
    int ierr = write_hdf5(h5file, rho.rho_, ions, previous_orbitals, lrs_);
    mmpi.allreduce(&ierr, 1, MPI_MIN);

    if (ierr < 0)
    {
        if (onpe0)
            (*MPIdata::serr)
                << "dumpMDrestartFile: cannot write ...previous_orbitals..."
                << std::endl;
        return ierr;
    }

    if (write_extrapolated_wf && ct.out_restart_info > 2)
    {
        // write extra  info needed for seamless MD restart
        ierr = orbitals.write(h5file, "ExtrapolatedFunction");
        mmpi.allreduce(&ierr, 1, MPI_MIN);
        if (ierr < 0)
        {
            if (onpe0)
                (*MPIdata::serr) << "dumpMDrestartFile: cannot write "
                                    "...ExtrapolatedFunction..."
                                 << std::endl;
            return ierr;
        }

        // write DM associated with non-extrapolated wavefunctions
        // (last computed solution of KS equations)
        proj_matrices_->writeSavedDM(h5file);
    }

    ierr = h5file.close();
    mmpi.allreduce(&ierr, 1, MPI_MIN);
    if (ierr < 0)
    {
        if (onpe0)
            (*MPIdata::serr)
                << "dumpMDrestartFile: cannot close file..." << std::endl;
        return ierr;
    }

    return 0;
}

template <class OrbitalsType>
void MGmol<OrbitalsType>::md(OrbitalsType** orbitals, Ions& ions)
{
    Control& ct = *(Control::instance());

    // initialize stepper data
    std::vector<std::string>& ions_names(ions.getLocalNames());
    std::vector<double>& tau0(ions.getTau0()); // tau0[3*ia+j]
    std::vector<double>& taup(ions.getTaup()); // taup[3*ia+j]
    std::vector<double>& taum(ions.getTaum()); // taum[3*ia+j]
    std::vector<double>& fion(ions.getFion()); // fion[3*ia+j]
    std::vector<double>& pmass(ions.getPmass()); // pmass[ia]
    std::vector<short>& atmove(ions.getAtmove()); // atmove[ia]
    std::vector<unsigned short>& rand_states(
        ions.getRandStates()); // initial random states

    // initialize taum with velocities
    std::vector<double>& vel(ions.getVelocities());
    taum = vel;

    DFTsolver<OrbitalsType>::resetItCount();

    orbitals_extrapol_.reset(OrbitalsExtrapolationFactory<OrbitalsType>::create(
        ct.WFExtrapolation()));

    MD_IonicStepper* stepper = new MD_IonicStepper(
        ct.dt, atmove, tau0, taup, taum, fion, pmass, rand_states);
    stepper->setThermostat(ct.thermostat_type, ct.tkel, ct.thtime, ct.thwidth,
        constraints_->size());

    constraints_->printConstraints(os_);

    ions.printPositions(os_);

    if (ct.restart_info > 1)
    {
        int flag_extrapolated_data
            = h5f_file_->checkDataExists("ExtrapolatedFunction0000");
        if (flag_extrapolated_data == 0)
            flag_extrapolated_data
                = h5f_file_->checkDataExists("ExtrapolatedFunction0");

        if (ct.restart_info > 2)
        {
            if (flag_extrapolated_data)
            {
                if (onpe0) os_ << "Create new orbitals_minus1..." << std::endl;

                orbitals_extrapol_->setupPreviousOrbitals(&current_orbitals_,
                    proj_matrices_.get(), lrs_, local_cluster_.get(),
                    currentMasks_.get(), corrMasks_.get(), *h5f_file_);

                // need to reset a few things as we just read new orbitals
                (*orbitals)->computeGramAndInvS();
            }

            DFTsolver<OrbitalsType>::setItCountLarge();
        }

        // check if we are restarting from an MD dump
        // no extrapolated functions -> atomic positions were extrapolated
        // in stepper->init()
        if (!flag_extrapolated_data)
        {
            moveVnuc(ions);
        }

        h5f_file_.reset();
    }

    bool ROM_MVP
        = (ct.getROMOptions().rom_stage == ROMStage::ONLINE_PINNED_H2O_3DOF);
#ifdef MGMOL_HAS_LIBROM
    // ROM - initialize orbitals and density matrix
    if (ROM_MVP)
    {
        if (onpe0) os_ << "Setup ROM MVP solver..." << std::endl;
        ExtendedGridOrbitals<ORBDTYPE>** extended_orbitals
            = reinterpret_cast<ExtendedGridOrbitals<ORBDTYPE>**>(orbitals);
        (*extended_orbitals)->set(ct.getROMOptions().basis_file, ct.numst);
        (*extended_orbitals)->orthonormalizeLoewdin();
        (*extended_orbitals)->setDataWithGhosts(true);
        (*extended_orbitals)->setIterativeIndex(10);

        std::shared_ptr<ProjectedMatricesInterface> projmatrices
            = getProjectedMatrices();
        projmatrices->setDMuniform(ct.getNelSpin());
        projmatrices->printDM(os_);
    }
#endif

    // additional SC steps to compensate random start
    if (ct.restart_info < 3 && !ROM_MVP)
    {
        double eks = 0.;
        quench(**orbitals, ions, ct.max_electronic_steps, 20, eks);
    }

    ct.max_changes_pot = 0;

    bool extrapolated_flag = true;
    if (ct.dt <= 0.) extrapolated_flag = false;

    int librom_snapshot_freq = ct.getROMOptions().librom_snapshot_freq;
    if (librom_snapshot_freq == -1) librom_snapshot_freq = ct.md_print_freq;

    MDfiles md_files;

    // main MD iteration loop
    for (int mdstep = 1; mdstep <= ct.num_MD_steps; mdstep++)
    {
        if (ct.checkTimeout())
        {
            if (onpe0)
                os_ << "Signal caught: No more MD iterations" << std::endl;
            break;
        }

        md_iterations_tm.start();

        double eks              = 0.;
        int retval              = 0;
        bool small_move         = true;
        bool last_move_is_small = true;

        bool force_on_ions = true;
#ifdef MGMOL_HAS_LIBROM
        // Ions for ROM
        Mesh* mymesh            = Mesh::instance();
        const pb::Grid& mygrid  = mymesh->grid();
        const double lattice[3] = { mygrid.ll(0), mygrid.ll(1), mygrid.ll(2) };
        std::vector<double> positions;
        std::vector<short> anumbers;
        getAtomicPositions(positions);
        getAtomicNumbers(anumbers);
        Ions* ROM_ions;

        // Pinned H2O 3 DOF
        PinnedH2O H2O_molecule;
        if (ct.getROMOptions().rom_stage == ROMStage::ONLINE_PINNED_H2O_3DOF)
        {
            if (onpe0)
                os_ << "Rotate Pinned H2O molecule in timestep " << mdstep
                    << std::endl;
            H2O_molecule.rotate(positions, anumbers);
            if (onpe0) H2O_molecule.print(os_);
            ROM_ions
                = new Ions(positions, anumbers, lattice, ions_->getSpecies());
            setupPotentials(*ROM_ions);
            force_on_ions = false;
        }
#endif

        if (ROM_MVP)
        {
            md_updateDMandEnergy_tm.start();
            updateDMandEnergy(**orbitals, *ROM_ions, eks);
            md_updateDMandEnergy_tm.stop();
        }
        else
        {
            retval = quench(**orbitals, ions, ct.max_electronic_steps, 0, eks);
        }

        do
        {
            // update localization regions
            if (ct.adaptiveLRs())
            {
                assert(lrs_);
                adaptLR(spreadf_.get()
#ifdef MGMOL_USE_SCALAPACK
                            ,
                    nullptr
#endif
                );

                last_move_is_small = lrs_->moveIsSmall();

                // update cluster for load balancing
                if (ct.load_balancing_alpha > 0.0
                    && mdstep % ct.load_balancing_modulo == 0)
                {
                    local_cluster_->computeClusters(
                        ct.load_balancing_max_iterations);
                }

                // printWithTimeStamp("quench done...",cout);

                // do extra cycles if centers resulting from quench
                // are "significantly" different from initial centers
                if (!last_move_is_small)
                {
                    printWithTimeStamp(
                        "WARNING: large move->extra inner cycle...", std::cout);
                    small_move = false;
                    move_orbitals(orbitals);

                    (*orbitals)->computeGramAndInvS();
                    dm_strategy_->update(**orbitals);

                    // reduce number of steps to keep total run time about the
                    // same
                    ct.num_MD_steps--;
                    // printWithTimeStamp("extra work done...",cout);
                }
            }
        } while (!last_move_is_small);

        if (retval < 0)
        {
            extrapolated_flag = false;
            if (ct.dt > 0.)
            {
                if (onpe0)
                {
                    os_ << "WARNING md(): quench returned value " << retval
                        << std::endl;
                    os_ << "STOP md..." << std::endl;
                }
                break;
            }
        }

        if (onpe0)
            os_ << std::setprecision(12) << std::fixed << "%%  "
                << md_iteration_ << "  IONIC CONFIGURATION ENERGY = " << eks
                << std::endl;

        // Compute forces
        if (force_on_ions) force(**orbitals, ions);

#ifdef MGMOL_HAS_LIBROM
        if (ct.getROMOptions().rom_stage == ROMStage::ONLINE_PINNED_H2O_3DOF)
        {
            force(**orbitals, *ROM_ions);
            // Pinned H2O 3 DOF
            if (onpe0)
                os_ << "Transpose rotate the PinnedH2O molecule" << std::endl;
            std::vector<double> forces;
            ROM_ions->getForces(forces);
            H2O_molecule.transpose_rotate(positions, anumbers, forces);
            ions.setLocalForces(forces, positions);
            delete ROM_ions;
        }

        if (ct.getROMOptions().rom_stage == ROMStage::TEST_ORBITAL)
        {
            if (onpe0)
            {
                os_ << "Projecting orbitals onto ROM subspaces to compare "
                    << ((ct.getROMOptions().compare_md) ? "MD dynamics"
                                                        : "force")
                    << std::endl;
                os_ << "Loading ROM basis " << ct.getROMOptions().basis_file
                    << std::endl;
                os_ << "ROM basis dimension = "
                    << ct.getROMOptions().num_orbbasis << std::endl;
            }

            // Project orbitals to ROM subspace
            project_orbital(ct.getROMOptions().basis_file,
                ct.getROMOptions().num_orbbasis, **orbitals);
            if (ct.getROMOptions().compare_md)
            {
                // overwrite ions force for MD time stepping
                force(**orbitals, ions);
            }
            else
            {
                // write ROM_ions force for one-step comparison
                ROM_ions = new Ions(
                    positions, anumbers, lattice, ions_->getSpecies());
                force(**orbitals, *ROM_ions);
                std::string zero = "0";
                if (ions_->getNumIons() < 256 || ct.verbose > 2)
                {
                    if (ct.verbose > 0) ROM_ions->printForcesGlobal(os_);
                }
                else if (zero.compare(ct.md_print_filename) == 0)
                {
                    ROM_ions->printForcesLocal(os_);
                }
                delete ROM_ions;
            }
        }
#endif

        // set fion
        ions.getLocalForces(fion);

        // constraints need to be added again and setup as atoms
        // may have moved and local atoms are not the same anymore
        constraints_->addConstraints(ions);

        constraints_->setup(ions);

        constraints_->printConstraintsForces(os_);

        constraints_->projectOutForces();

        // Print atomic coordinates and forces
        if (md_iteration_ % ct.md_print_freq == 0)
        {
            std::string zero = "0";
            if (zero.compare(ct.md_print_filename) != 0)
            {
                std::vector<float> spreads;
                std::vector<Vector3D> centers;
                std::vector<int> lgids;
                if (ct.isLocMode())
                {
                    spreadf_->computeLocalSpreads(spreads);
                    spreadf_->getLocalCenters(centers);
                    spreadf_->getLocalGids(lgids);
                }

                md_files.printDataInFiles(ions_names, tau0, fion, taum, centers,
                    spreads, lgids, md_iteration_, ct.dt);
            }

            // also print in stdout for small problems or high verbosity
            if (ions_->getNumIons() < 256 || ct.verbose > 2)
            {
                if (ct.verbose > 0) ions.printForcesGlobal(os_);
            }
            else if (zero.compare(ct.md_print_filename) == 0)
            {
                ions.printForcesLocal(os_);
            }
        }

        if (ct.dt <= 0.) checkMaxForces(fion, atmove, os_);

        // add random forces if Langevin dynamics
        stepper->addLangevinForces();
        constraints_->projectOutForces(50);

        // Compute taup
        stepper->run();
        constraints_->enforceConstraints(20);
        if (ct.enforceVmass0) ions.removeMassCenterMotion();
        if (ct.verbose > 2 && ct.dt > 0.) stepper->printVelocities(os_);

        // compute velocities and temperatures at t+1/2
        double ekin = 0.;
        if (ct.dt > 0.)
        {
            // T=Ek/(kB*3/2*N)
            const double temperature = stepper->temperature();
            ekin                     = stepper->kineticEnergy();
            if (onpe0)
            {
                os_ << std::setprecision(6);
                os_ << "Kinetic   Energy= " << ekin;
                os_ << std::setprecision(5);
                os_ << " (T= " << temperature << " )" << std::endl;
            }
        }

        if (onpe0)
        {
            os_ << std::setprecision(10);
            os_ << "Total     Energy= " << ekin + eks << std::endl;
        }

        stepper->updateTau();

        // update local information about ions
        ions.updateIons();

        ions.setup();

        ct.steps = md_iteration_;

        moveVnuc(ions);

        if (!small_move)
        {
            orbitals_extrapol_->clearOldOrbitals();
            lrs_->clearOldCenters();
        }

        // save DM for possible restart write
        // note: extrapolation is going to modify it!
        if ((ct.out_restart_info > 2)
            && (((md_iteration_ % ct.checkpoint) == 0)
                || (mdstep == ct.num_MD_steps)))
            proj_matrices_->saveDM();

        preWFextrapolation();

        if (ct.dt > 0.
            || ct.WFExtrapolation() == WFExtrapolationType::Reversible)
        {
            if (ct.lrs_extrapolation > 0) extrapolate_centers(small_move);

            extrapolate_orbitals(orbitals);
        }
        else
            move_orbitals(orbitals);

        postWFextrapolation(*orbitals);

        // rho_->setup(ct.orbital_type,(*orbitals)->getGlobalIndexes());
        // rho_->update(**orbitals);

        md_time_ += ct.dt;
        if (ct.dt > 0.) md_iteration_++;

        if (ct.checkpoint && ct.out_restart_file != "0")
            if (ct.out_restart_info > 0)
                if ((md_iteration_ % ct.checkpoint) == 0
                    && mdstep < ct.num_MD_steps)
                {
                    int ierr    = -1;
                    short count = 0;
                    while (ierr < 0 && count < DUMP_MAX_NUM_TRY)
                    {
                        dump_tm_.start();
                        ierr = dumpMDrestartFile(
                            **orbitals, ions, *rho_, extrapolated_flag, count);
                        dump_tm_.stop();
                        if (onpe0 && ierr < 0 && count < (DUMP_MAX_NUM_TRY - 1))
                            std::cout
                                << "dumpMDrestartFile() failed... try again..."
                                << std::endl;
                        if (ierr < 0) sleep(1.);
                        count++;
                    }

                    printWithTimeStamp("dumped restart file...", std::cout);
                }

#ifdef MGMOL_HAS_LIBROM
        // Save orbital snapshots
        if (ct.getROMOptions().save_librom_snapshot > 0
            && md_iteration_ % librom_snapshot_freq == 0)
        {
            int ierr = save_orbital_snapshot(
                ct.md_print_filename + "_mdstep" + std::to_string(mdstep),
                **orbitals);

            if (ierr < 0)
                os_ << "WARNING md(): writing ROM snapshot data failed!!!"
                    << std::endl;
        }
#endif

        md_iterations_tm.stop();

    } // md loop

    // final dump
    if (ct.out_restart_info > 0)
    {
        int ierr    = -1;
        short count = 0;
        while (ierr < 0 && count < DUMP_MAX_NUM_TRY)
        {
            dump_tm_.start();
            ierr = dumpMDrestartFile(
                **orbitals, ions, *rho_, extrapolated_flag, count);
            dump_tm_.stop();

            if (onpe0 && ierr < 0 && count < (DUMP_MAX_NUM_TRY - 1))
                std::cout << "dumpMDrestartFile() failed... try again..."
                          << std::endl;
            if (ierr < 0) sleep(1.);
            count++;
        }

        printWithTimeStamp("dumped last restart file...", std::cout);
    }

    delete stepper;
    orbitals_extrapol_.reset();
}

template <class OrbitalsType>
void MGmol<OrbitalsType>::loadRestartFile(const std::string filename)
{
    if (onpe0) std::cout << "loadRestartFile..." << std::endl;
    MGmol_MPI& mmpi(*(MGmol_MPI::instance()));
    Control& ct              = *(Control::instance());
    Mesh* mymesh             = Mesh::instance();
    const pb::PEenv& myPEenv = mymesh->peenv();

    /* For now, we only consider double-precision hdf5 I/O. */
    assert(ct.restart_info > 3);
    assert((ct.AtomsDynamic() == AtomsDynamicType::MD)
           || (ct.AtomsDynamic() == AtomsDynamicType::Quench));

    HDFrestart h5file(filename, myPEenv, ct.out_restart_file_type);

    int ierr = read_restart_data(h5file, *rho_, *current_orbitals_);
    if (ierr < 0)
    {
        if (onpe0)
            std::cerr << "loadRestartFile: failed to read the restart file."
                      << std::endl;

        mmpi.abort();
    }
    if (!ct.fullyOccupied())
    {
        // overwrite DM with restart data in dataset Density_Matrix_WF
        if (h5file.checkDataExists("Density_Matrix_WF"))
            ierr = proj_matrices_->readWFDM(h5file);
    }

    if (h5file.checkDataExists("Preceding_Hartree"))
    {
        ions_->readRestartPreviousPositions(h5file);
        ions_->resetPositionsToPrevious();
        ions_->setup();

        Potentials& pot = hamiltonian_->potential();
        pot.initialize(*ions_);
        if (onpe0) std::cout << "Reset VhRho to backup..." << std::endl;
        pot.resetVhRho2Backup();
        electrostat_->setupRhoc(pot.rho_comp());
    }

    ierr = h5file.close();
    mmpi.allreduce(&ierr, 1, MPI_MIN);
    if (ierr < 0)
    {
        if (onpe0)
            (*MPIdata::serr)
                << "loadRestartFile: cannot close file..." << std::endl;
        return;
    }

    return;
}

template class MGmol<LocGridOrbitals<ORBDTYPE>>;
template class MGmol<ExtendedGridOrbitals<ORBDTYPE>>;

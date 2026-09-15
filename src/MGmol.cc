// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "global.h"

#include "ABPG.h"
#include "AOMMprojector.h"
#include "AndersonMix.h"
#include "ConstraintSet.h"
#include "Control.h"
#include "DFTsolver.h"
#include "DMStrategyFactory.h"
#include "DavidsonSolver.h"
#include "Electrostatic.h"
#include "Energy.h"
#include "EnergySpreadPenalty.h"
#include "FDkernels.h"
#include "FDoper.h"
#include "FIRE.h"
#include "Forces.h"
#include "GridFunc.h"
#include "HDFrestart.h"
#include "Hamiltonian.h"
#include "Ions.h"
#include "KBPsiMatrixSparse.h"
#include "LBFGS.h"
#include "LocGridOrbitals.h"
#include "LocalMatrices2ReplicatedMatrix.h"
#include "LocalizationRegions.h"
#include "MDfiles.h"
#include "MGOrbitalsPreconditioning.h"
#include "MGkernels.h"
#include "MGmol.h"
#include "MPIdata.h"
#include "MVPSolver.h"
#include "MasksSet.h"
#include "Mesh.h"
#include "PackedCommunicationBuffer.h"
#include "PoissonInterface.h"
#include "Potentials.h"
#include "Power.h"
#include "PowerGen.h"
#include "Preconditioning.h"
#include "ProjectedMatricesMehrstellen.h"
#include "ProjectedMatricesSparse.h"
#include "ReplicatedMatrix.h"
#include "ReplicatedMatrix2SquareLocalMatrices.h"
#include "ReplicatedVector.h"
#include "Rho.h"
#include "SP2.h"
#include "SpreadPenalty.h"
#include "SpreadPenaltyVolume.h"
#include "SpreadsAndCenters.h"
#include "SubspaceProjector.h"
#include "XCfunctionalFactory.h"
#include "XConGrid.h"
#include "manage_memory.h"

#ifdef MGMOL_USE_SCALAPACK
#include "DistMatrix.h"
#include "DistMatrix2SquareLocalMatrices.h"
#include "DistVector.h"
#include "GrassmanLineMinimization.h"
#include "MLWFTransform.h"
#include "SparseDistMatrix.h"
#include "SubMatrices.h"
#endif

namespace mgmol
{
std::ostream* out = nullptr;
}

std::string description;

extern Timer computeHij_tm;
extern Timer get_kbpsi_tm;
extern Timer get_Hpsi_and_Hij_tm;
extern Timer vnlpsi_tm;
extern Timer sygv_tm;
extern Timer split_allreduce_sums_double_tm;
extern Timer sgemm_tm;
extern Timer dgemm_tm;
extern Timer mpgemm_tm;
extern Timer tttgemm_tm;
extern Timer bligemm_tm;
extern Timer dsyrk_tm;
extern Timer ssyrk_tm;
extern Timer mpsyrk_tm;
extern Timer tttsyrk_tm;
extern Timer ttdot_tm;
extern Timer loopdot_tm;
extern Timer loopaxpy_tm;
extern Timer loopscal_tm;
extern Timer loopcp_tm;
extern Timer md_iterations_tm;
extern Timer md_tau_tm;
extern Timer md_moveVnuc_tm;
extern Timer md_updateMasks_tm;
extern Timer md_extrapolateOrbitals_tm;
extern Timer md_updateRhoAndPot_tm;
extern Timer md_updateDMandEnergy_tm;
extern Timer quench_tm;
extern Timer ions_setupInteractingIons_tm;
extern Timer ions_setup_tm;
extern Timer updateCenters_tm;
#ifdef MGMOL_USE_SCALAPACK
extern Timer get_NOLMO_tm;
extern Timer get_MLWF_tm;
#endif

#include "mgmol_Signal.h"
std::set<int> Signal::recv_;

template <class OrbitalsType>
MGmol<OrbitalsType>::MGmol(MPI_Comm comm, std::ostream& os,
    std::string input_filename, std::string lrs_filename,
    std::string constraints_filename)
    : os_(os), comm_(comm)
{
    constraints_.reset(new ConstraintSet());

    mgmol::out = &os;

    current_orbitals_ = nullptr;

    setupFromInput(input_filename);

    /*
     * Extra setup if using localization regions
     */
    setupLRs(lrs_filename);

    if (!constraints_filename.empty())
        setupConstraintsFromInput(constraints_filename);
}

template <class OrbitalsType>
MGmol<OrbitalsType>::~MGmol()
{
    if (current_orbitals_) delete current_orbitals_;
}

template <>
void MGmol<LocGridOrbitals<ORBDTYPE>>::initialMasks()
{
    assert(lrs_);

    Control& ct = *(Control::instance());

    if (ct.verbose > 0)
        printWithTimeStamp("MGmol<OrbitalsType>::initialMasks()...", os_);

    currentMasks_
        = std::shared_ptr<MasksSet>(new MasksSet(false, ct.getMGlevels()));
    currentMasks_->setup(lrs_);

    corrMasks_ = std::shared_ptr<MasksSet>(new MasksSet(true, 0));
    corrMasks_->setup(lrs_);
}

template <>
void MGmol<ExtendedGridOrbitals<ORBDTYPE>>::initialMasks()
{
}

template <class OrbitalsType>
template <typename MemorySpaceType>
int MGmol<OrbitalsType>::initial()
{
    Control& ct     = *(Control::instance());
    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    Mesh* mymesh    = Mesh::instance();

    assert(ct.numst >= 0);

    if (ct.verbose > 0)
        printWithTimeStamp("MGmol<OrbitalsType>::initial()...", os_);

    init_tm_.start();

    const pb::Grid& mygrid = mymesh->grid();

    hamiltonian_->setup(mygrid, ct.lap_type);

    pb::Lap<ORBDTYPE>* lapop
        = ct.Mehrstellen() ? hamiltonian_->lapOper() : nullptr;
    g_kbpsi_.reset(new KBPsiMatrixSparse(lapop));

    check_anisotropy();

    if (ct.verbose > 0)
        printWithTimeStamp(
            "MGmol<OrbitalsType>::initial(), create ProjectedMatrices...", os_);

    // If not an initial run read data from files
    if (ct.restart_info > 2 && ct.isLocMode())
    {
        std::string name = "ExtrapolatedFunction";
        if (ct.verbose > 0)
            printWithTimeStamp(
                "read LRs from ExtrapolatedFunction database...", os_);
        int n = read_restart_lrs(*h5f_file_, name);
        if (n == 0)
        {
            if (ct.verbose > 0)
                printWithTimeStamp("read LRs from Function database...", os_);
            name = "Function";
            n    = read_restart_lrs(*h5f_file_, name);
        }
        if (n < 0) return n;

        if (n > 0) lrs_->setup();
    }

    if (ct.isLocMode())
    {
        double dlrsmin = lrs_->computeMinDistBetweenLocalPairs(
            std::cout, (ct.verbose > 2));
        if (dlrsmin < 1.e-3)
        {
            std::cout << "WARNING: Min. distance between LR centers is "
                      << dlrsmin << "!!!" << std::endl;
        }

        // initialize and setup load balancing object
        // set number of iterations to 10.
        if (ct.load_balancing_alpha > 0.0)
        {
            local_cluster_.reset(new ClusterOrbitals(lrs_));
            local_cluster_->setup();
            local_cluster_->computeClusters(ct.load_balancing_max_iterations);
        }
    }

    // initialize data distribution objects
    bool with_spin = (mmpi.nspin() > 1);

    bool use_replicated_matrix = ct.rmatrices;

    if (ct.Mehrstellen())
    {
        if (use_replicated_matrix)
            proj_matrices_.reset(
                new ProjectedMatricesMehrstellen<ReplicatedMatrix>(
                    ct.numst, with_spin, ct.occ_width_));
        else
        {
#ifdef MGMOL_USE_SCALAPACK
            proj_matrices_.reset(new ProjectedMatricesMehrstellen<
                dist_matrix::DistMatrix<DISTMATDTYPE>>(
                ct.numst, with_spin, ct.occ_width_));
#else
            std::cerr << "Not implemented" << std::endl;
#endif
        }
    }
    else if (ct.short_sighted)
        proj_matrices_.reset(new ProjectedMatricesSparse(
            ct.numst, ct.occ_width_, lrs_, local_cluster_.get()));
    else if (use_replicated_matrix)
        proj_matrices_.reset(new ProjectedMatrices<ReplicatedMatrix>(
            ct.numst, with_spin, ct.occ_width_));
    else
    {
#ifdef MGMOL_USE_SCALAPACK
        proj_matrices_.reset(
            new ProjectedMatrices<dist_matrix::DistMatrix<DISTMATDTYPE>>(
                ct.numst, with_spin, ct.occ_width_));
#else
        std::cerr << "Not implemented" << std::endl;
#endif
    }

    forces_.reset(new Forces<OrbitalsType>(
        hamiltonian_.get(), rho_.get(), proj_matrices_.get()));

    if (ct.verbose > 0)
        printWithTimeStamp(
            "MGmol<OrbitalsType>::initial(), create MasksSet...", os_);

    initialMasks();

    BlockVector<ORBDTYPE, MemorySpaceType>::setOverAllocateFactor(
        ct.orbitalsOverallocateFactor());

    if (ct.verbose > 0)
        printWithTimeStamp("MGmol<OrbitalsType>::initial(), create T...", os_);

    current_orbitals_ = new OrbitalsType("Primary", mygrid, mymesh->subdivx(),
        ct.numst, ct.bcWF, proj_matrices_.get(), lrs_, currentMasks_.get(),
        corrMasks_.get(), local_cluster_.get(), true);

    increaseMemorySlotsForOrbitals<ORBDTYPE, MemorySpaceType>();

    Potentials& pot            = hamiltonian_->potential();
    pb::Lap<ORBDTYPE>* lapOper = hamiltonian_->lapOper();

    // If not an initial run read data from files
    if (ct.restart_info > 0)
    {
        md_time_      = h5f_file_->getMDTimeFromFile();
        md_iteration_ = h5f_file_->getMDstepFromFile();

        int ierr = read_restart_data(*h5f_file_, *rho_, *current_orbitals_);
        if (ierr < 0) return -1;

        if (ct.restart_info > 2)
        {
            current_orbitals_->applyMask();
        }
    }
    else
    {
        md_time_      = 0.;
        md_iteration_ = 1;
    }

    if (ct.MD_last_step_ > 0)
    {
        ct.num_MD_steps = ct.MD_last_step_ - md_iteration_ + 1;
    }

    mmpi.barrier();
    if (ct.verbose > 0) current_orbitals_->printChromaticNumber(os_);

    // Random initialization of the wavefunctions
    if (ct.restart_info <= 2)
    {
        if (ct.verbose > 0)
            printWithTimeStamp(
                "MGmol<OrbitalsType>::initial(), init wf and masks...", os_);

        // Make temp mask for initial random wave functions
        if (ct.init_loc == 1 && currentMasks_)
        {
            float cut_init = ct.initRadius();
            assert(cut_init > 0.);
            currentMasks_->initialize(lrs_, 0, cut_init);
        }

        // Initialize states
        current_orbitals_->initWF(lrs_);

        // initialize masks again
        if (ct.init_loc == 1 && currentMasks_)
        {
            currentMasks_->initialize(lrs_, 0);
        }
    }

    // Initialize the radial potential stuff
    if (ct.verbose > 0) printWithTimeStamp("initKBR()...", os_);
    initKBR();

    ions_->setup();

    double d = ions_->computeMinLocalSpacing();
    if (d < 1.e-3)
    {
        std::cerr
            << "ERROR: min. distance between ions is smaller than 1.e-3!!!\n";
    }

    // Initialize the nuclear local potential and the compensating charges
    if (ct.verbose > 0) printWithTimeStamp("setupPotentials()...", os_);
    setupPotentials(*ions_);

    // initialize Rho
    if (ct.verbose > 0) printWithTimeStamp("Initialize Rho...", os_);
    if (ct.restart_info <= 1) proj_matrices_->setDMuniform(ct.getNelSpin());

    rho_->setup(ct.getOrthoType(), current_orbitals_->getOverlappingGids());

    if (ct.restart_info <= 1)
    {
        if (ct.init_type == 0) // random functions
        {
            rho_->init(pot.rho_comp());
        }
        else
        {
            rho_->update(*current_orbitals_);
        }
    }

    if (ct.verbose > 0) printWithTimeStamp("Initialize XC functional...", os_);
    xcongrid_
        = std::shared_ptr<XConGrid>(XCfunctionalFactory<OrbitalsType>::create(
            ct.xctype, mmpi.nspin(), *rho_, pot));
    assert(xcongrid_);

    // initialize nl potentials with restart values if possible
    //    if( ct.restart_info>1 )
    {
        electrostat_->setupInitialVh(pot.vh_rho());
        electrostat_->computeVhRho(*rho_); // for energy

        // xc computed from rho
        xcongrid_->update();
    }

    current_orbitals_->setDataWithGhosts();
    current_orbitals_->trade_boundaries();

    // initialize matrices S and invB

    if (ct.numst > 0)
    {
        if (ct.verbose > 0)
            printWithTimeStamp("Compute initial Gram matrix...", os_);
        current_orbitals_->computeGramAndInvS(ct.verbose);
        if (ct.verbose > 0)
            printWithTimeStamp("Compute initial B matrix...", os_);
        current_orbitals_->computeBAndInvB(*lapOper);
        if (ct.verbose > 0)
            printWithTimeStamp("Compute initial condition number...", os_);
        current_orbitals_->checkCond(100000., ct.AtomsMove());
    }

    if (ct.restart_info == 0)
    {
        if (ct.verbose > 0) printWithTimeStamp("update_pot...", os_);
        update_pot(*ions_);
    }

    if (ct.isLocMode() || ct.isSpreadFunctionalActive())
    {
        Vector3D origin(mygrid.origin(0), mygrid.origin(1), mygrid.origin(2));
        Vector3D ll(mygrid.ll(0), mygrid.ll(1), mygrid.ll(2));
        spreadf_.reset(new SpreadsAndCenters<OrbitalsType>(origin, ll));
    }

    bool energy_with_spread_penalty = false;
    if (ct.isSpreadFunctionalActive())
    {
        if (ct.isSpreadFunctionalVolume())
        {
            spread_penalty_.reset(
                new SpreadPenaltyVolume<OrbitalsType>(spreadf_.get(),
                    ct.spreadPenaltyTarget(), ct.spreadPenaltyAlphaFactor(),
                    ct.spreadPenaltyDampingFactor()));
        }
        else if (ct.isSpreadFunctionalEnergy())
        {
            energy_with_spread_penalty = true;
            spread_penalty_.reset(
                new EnergySpreadPenalty<OrbitalsType>(spreadf_.get(),
                    ct.spreadPenaltyTarget(), ct.spreadPenaltyAlphaFactor()));
        }
        else
            spread_penalty_.reset(
                new SpreadPenalty<OrbitalsType>(spreadf_.get(),
                    ct.spreadPenaltyTarget(), ct.spreadPenaltyAlphaFactor(),
                    ct.spreadPenaltyDampingFactor()));
    }

    std::shared_ptr<SpreadPenaltyInterface<OrbitalsType>> spread_penalty
        = energy_with_spread_penalty ? spread_penalty_ : nullptr;
    energy_ = std::shared_ptr<Energy<OrbitalsType>>(new Energy<OrbitalsType>(
        mygrid, pot, *electrostat_, *rho_, *xcongrid_, spread_penalty.get()));

    if (ct.verbose > 0) printWithTimeStamp("Setup matrices...", os_);

    updateHmatrix(*current_orbitals_, *ions_);

    // HMVP algorithm requires that H is initialized
    if (use_replicated_matrix)
        dm_strategy_.reset(
            DMStrategyFactory<OrbitalsType, ReplicatedMatrix>::create(comm_,
                os_, *ions_, rho_.get(), energy_.get(), electrostat_.get(),
                hamiltonian_.get(), this, proj_matrices_.get(),
                current_orbitals_));
    else
    {
#ifdef MGMOL_USE_SCALAPACK
        dm_strategy_.reset(DMStrategyFactory<OrbitalsType,
            dist_matrix::DistMatrix<double>>::create(comm_, os_, *ions_,
            rho_.get(), energy_.get(), electrostat_.get(), hamiltonian_.get(),
            this, proj_matrices_.get(), current_orbitals_));
#endif
    }

    // theta = invB * Hij
    proj_matrices_->updateThetaAndHB();

    dm_strategy_->initialize(*current_orbitals_);

    if (ct.verbose > 1)
    {
        proj_matrices_->printMatrices(os_);
        proj_matrices_->printDM(os_);
    }

    init_tm_.stop();

    if (ct.verbose > 0) printWithTimeStamp("Initialization done...", os_);

    return 0;
} // initial()

template <class OrbitalsType>
void MGmol<OrbitalsType>::run()
{
    total_tm_.start();

    Control& ct = *(Control::instance());

#ifdef DEBUG
    Mesh* mymesh             = Mesh::instance();
    const pb::PEenv& myPEenv = mymesh->peenv();
    *MPIdata::sout << " Run begins on processor " << myPEenv.mytask()
                   << std::endl;
#endif

    double eks = 0.;

    if (ct.verbose > 0) printWithTimeStamp("Run...", os_);
    // Dispatch to the method chosen
    switch (ct.AtomsDynamic())
    {
        case AtomsDynamicType::Quench:
            quench(
                *current_orbitals_, *ions_, ct.max_electronic_steps, 20, eks);

            // Forces for the last states
            force(*current_orbitals_, *ions_);

            constraints_->addConstraints(*ions_);

            constraints_->setup(*ions_);

            constraints_->printConstraintsForces(os_);

            constraints_->projectOutForces(20);

            if ((ions_->getNumIons() <= 1024 || ct.verbose > 1))
                ions_->printForcesGlobal(os_);
            if (ct.verbose > 1)
                forces_->dVsdRhoComponent(*current_orbitals_, *ions_);

            finalEnergy();

            printMM();

            break;

        case AtomsDynamicType::MD:
            md(&current_orbitals_, *ions_);
            // finalEnergy();
            break;

        case AtomsDynamicType::LBFGS:
            lbfgsrlx(&current_orbitals_, *ions_);
            // finalEnergy();
            break;

        case AtomsDynamicType::FIRE:
            runfire(&current_orbitals_, *ions_);
            // finalEnergy();
            break;

        default:
            (*MPIdata::serr) << "run: Undefined MD method" << std::endl;
    }

    cleanup();
} // run()

template <class OrbitalsType>
void MGmol<OrbitalsType>::finalEnergy()
{
    // Get the total energy
    const double ts = 0.5 * proj_matrices_->computeEntropy(); // in [Ha]
    total_energy_   = energy_->evaluateTotal(
        ts, proj_matrices_.get(), *ions_, *current_orbitals_, 2, os_);
}

template <class OrbitalsType>
void MGmol<OrbitalsType>::printMM()
{
    Control& ct = *(Control::instance());
    if (ct.tmatrices == 1)
    {
        std::ofstream tfile("s.mm", std::ios::out);
        proj_matrices_->printGramMM(tfile);
        std::ofstream tfileh("h.mm", std::ios::out);
#ifdef MGMOL_USE_SCALAPACK
        ProjectedMatrices<dist_matrix::DistMatrix<DISTMATDTYPE>>* projmatrices
            = dynamic_cast<
                ProjectedMatrices<dist_matrix::DistMatrix<DISTMATDTYPE>>*>(
                proj_matrices_.get());
        assert(projmatrices != nullptr);
        projmatrices->printHamiltonianMM(tfileh);
#else
        std::cerr << "MGmol<OrbitalsType>::printMM() not implemented"
                  << std::endl;
#endif
    }
}

/* Writes out header information */
template <class OrbitalsType>
void MGmol<OrbitalsType>::write_header()
{
    Mesh* mymesh           = Mesh::instance();
    const pb::Grid& mygrid = mymesh->grid();

    time_t tt;

    time(&tt);
    char* timeptr = ctime(&tt);

    Control& ct     = *(Control::instance());
    Potentials& pot = hamiltonian_->potential();

    if (onpe0)
    {

        os_ << "//////////////////////////////////////////////////////////"
            << std::endl;
        os_ << std::endl;
#ifdef GITHASH
#define xstr(x) #x
#define LOGGIT(x) os_ << " MGmol: git_hash " << xstr(x) << std::endl;
        LOGGIT(GITHASH);
        os_ << std::endl;
#endif
        os_ << " Compiled: " << __DATE__ << ", " << __TIME__ << std::endl;
        os_ << " Real-space finite difference ab initio calculations\n";
        os_ << std::endl;
        os_ << "//////////////////////////////////////////////////////////"
            << std::endl;

        os_ << std::endl << std::endl << description << std::endl;
        os_ << " Run started at " << timeptr << std::endl;

        os_ << " Orbitals precision (in bytes): " << sizeof(ORBDTYPE)
            << std::endl
            << std::endl;

        Potentials& pot = hamiltonian_->potential();
        pot.writeNames(os_);

        mymesh->print(os_);

        pb::Lap<ORBDTYPE>* lapop = hamiltonian_->lapOper();
        os_ << " Laplacian Discretization: " << lapop->name() << std::endl;

#ifdef SMP_NODE
        os_ << " " << omp_get_max_threads() << " thread"
            << (omp_get_max_threads() > 1 ? "s " : " ");
        os_ << "active" << std::endl << std::endl;
#endif
#ifdef MGMOL_USE_SCALAPACK
        if (!ct.rmatrices)
        {
            os_ << " ScaLapack block size: "
                << dist_matrix::DistMatrix<DISTMATDTYPE>::getBlockSize()
                << std::endl;
        }

        if (!ct.short_sighted)
        {
            MatricesBlacsContext& mbc(MatricesBlacsContext::instance());
            mbc.print(os_);
        }
#endif

        ct.printPoissonOptions(os_);
    } // onpe0

    if (current_orbitals_ && ct.verbose > 0)
    {
        current_orbitals_->printNumst(os_);
        current_orbitals_->printChromaticNumber(os_);
    }

    int nions = ions_->getNumIons();
    if (onpe0)
    {
        os_ << " Number of ions     = " << nions << std::endl;
        os_ << " Total charge in cell        = " << pot.getChargeInCell()
            << std::endl;
        os_ << " Electronic temperature (K)  = " << ct.etemp_ << std::endl;
        os_ << " Fermi-Dirac width kb*T (Ha) = " << 0.5 * ct.occ_width_
            << std::endl;

        ct.print(os_);

        os_ << std::endl
            << std::endl
            << " Atomic species information:" << std::endl
            << std::endl;
        os_ << std::fixed << std::setprecision(5);

        const std::vector<Species>& sp(ct.getSpecies());
        for (int idx = 0; idx < (int)sp.size(); idx++)
        {
            const Species& sp_ion(sp[idx]);

            os_ << " Species #" << idx + 1 << ": ";
            sp_ion.print(os_);

            os_ << " dim_l        = " << sp_ion.dim_l()
                << "    ->Diameter    = " << sp_ion.dim_l() * mygrid.hmin()
                << "[bohr]" << std::endl;
            os_ << " dim_nl      = " << sp_ion.dim_nl()
                << "    ->Diameter    = " << sp_ion.dim_nl() * mygrid.hmin()
                << "[bohr]" << std::endl;
        }
        if (ct.short_sighted)
        {
            os_ << std::endl;
            os_ << " Short_Sighted Solver Parameters: " << std::endl
                << std::endl;
            os_ << " Accelerator = Flexible GMRES(" << ct.fgmres_kim << ")"
                << std::endl;
            if (ct.ilu_type == 0)
                os_ << " Preconditioner = ilu" << ct.ilu_lof << std::endl;
            else if (ct.ilu_type == 1)
                os_ << " Preconditioner = modified ilu" << std::endl;
            else if (ct.ilu_type == 2)
                os_ << " Preconditioner = standard ilu" << std::endl;
            else
                os_ << " Preconditioner = ilu0 preconditioner" << std::endl;
            os_ << std::scientific
                << " fgmres convergence tolerance = " << ct.fgmres_tol
                << std::endl;
            os_ << " fgmres max. iterations = " << ct.fgmres_maxits
                << std::endl;
            os_ << std::scientific
                << " drop tolerance for (standard/ modified) ilu = "
                << ct.ilu_droptol << std::endl;
            os_ << " max. fill-in for (standard/ modified) ilu = "
                << ct.ilu_maxfil << std::endl;
        }
    } // onpe0

    // Write out the ionic postions and displacements
    ions_->printPositions(os_);

    if (ct.isLocMode() && ct.verbose > 3) lrs_->printAllRegions(os_);
}

template <class OrbitalsType>
void MGmol<OrbitalsType>::check_anisotropy()
{
    Mesh* mymesh           = Mesh::instance();
    const pb::Grid& mygrid = mymesh->grid();

    if (mygrid.anisotropy() > 1.2)
    {
        write_header();
        if (onpe0)
            (*MPIdata::serr) << " hmax=" << mygrid.hmax()
                             << ", hmin=" << mygrid.hmin() << std::endl;
        (*MPIdata::serr) << "init: Anisotropy too large: "
                         << mygrid.anisotropy() << std::endl;
        MGmol_MPI& mmpi = *(MGmol_MPI::instance());
        mmpi.abort();
    }
}

template <class OrbitalsType>
void MGmol<OrbitalsType>::printEigAndOcc()
{
    Control& ct = *(Control::instance());
    if (!(ct.fullyOccupied() && ct.getOrthoType() != OrthoType::Eigenfunctions
            && ct.occupationWidthIsZero())
        && onpe0)
    {
        // try with ReplicatedMatrix first
        std::shared_ptr<ProjectedMatrices<ReplicatedMatrix>> projmatrices
            = std::dynamic_pointer_cast<ProjectedMatrices<ReplicatedMatrix>>(
                proj_matrices_);
        if (projmatrices)
        {
            projmatrices->printEigenvalues(os_);
            projmatrices->printOccupations(os_);
        }
#ifdef MGMOL_USE_SCALAPACK
        else
        {
            std::shared_ptr<
                ProjectedMatrices<dist_matrix::DistMatrix<DISTMATDTYPE>>>
                projmatrices = std::dynamic_pointer_cast<
                    ProjectedMatrices<dist_matrix::DistMatrix<DISTMATDTYPE>>>(
                    proj_matrices_);
            assert(projmatrices);

            projmatrices->printEigenvalues(os_);
            projmatrices->printOccupations(os_);
        }
#endif
    }
}

#if 0
double get_trilinval(const double xc, const double yc, const double zc,
                     const double h0, const double h1, const double h2,
                     const Vector3D& ref, const Vector3D& lattice,
                     RadialInter& lpot)
{
    double val=0.;
    const double shift=0.5;
    const double hh0=shift*h0;
    const double hh1=shift*h1;
    const double hh2=shift*h2;
    for(int i=-1;i<2;i++){
        double xcc=xc+i*hh0;
        double w0=1.-fabs(0.5*i);
        for(int j=-1;j<2;j++){
            double ycc=yc+j*hh1;
            double w1=1.-fabs(0.5*j);
            for(int k=-1;k<2;k++){
                double zcc=zc+k*hh2;
                double w2=1.-fabs(0.5*k);
                
                Vector3D vc(xcc,ycc,zcc);
                double w=w0*w1*w2*0.125;
                double r = vc.minimage(ref,lattice);            
                val += w*lpot.cubint(r);
            }
        }
    }
    return val;
}
#endif

template <class OrbitalsType>
void MGmol<OrbitalsType>::setupPotentials(Ions& ions)
{
    init_nuc_tm_.start();

    Control& ct = *(Control::instance());
    if (onpe0 && ct.verbose > 1) os_ << "Init vnuc and rhoc..." << std::endl;

    Potentials& pot = hamiltonian_->potential();

    // initialize poentials based on ionic positions and their species
    pot.initialize(ions);

    if (ct.verbose > 0) printWithTimeStamp("Setup kbpsi...", os_);
    g_kbpsi_->setup(ions);

    electrostat_->setupRhoc(pot.rho_comp());

    if (onpe0 && ct.verbose > 3) os_ << " setupPotentials done" << std::endl;

    init_nuc_tm_.stop();
}

template <class OrbitalsType>
void MGmol<OrbitalsType>::printTimers()
{
    Control& ct = *(Control::instance());
    if (onpe0)
    {
        os_ << std::setprecision(2) << std::fixed << std::endl;
        os_ << " Timing (real time: min_time / avg_time / max_time / "
               "min_#_calls / avg_#_calls / max_#_calls): "
            << std::endl;
        os_ << " =============================================================="
               "====== "
            << std::endl;
    }
    pb::GridFunc<double>::printTimers(os_);
    pb::GridFunc<float>::printTimers(os_);
    pb::GridFuncVector<double>::printTimers(os_);
    pb::GridFuncVector<float>::printTimers(os_);
    pb::printMGkernelTimers(os_);
    pb::printFDkernelTimers(os_);
    pb::FDoperInterface::printTimers(os_);
    OrbitalsType::printTimers(os_);
    SinCosOps<OrbitalsType>::printTimers(os_);
    GridMask::printTimers(os_);

    sgemm_tm.print(os_);
    dgemm_tm.print(os_);
    mpgemm_tm.print(os_);
    tttgemm_tm.print(os_);
    bligemm_tm.print(os_);

    ssyrk_tm.print(os_);
    dsyrk_tm.print(os_);
    mpsyrk_tm.print(os_);
    tttsyrk_tm.print(os_);
    ttdot_tm.print(os_);

    loopcp_tm.print(os_);
    loopaxpy_tm.print(os_);
    loopscal_tm.print(os_);
    loopdot_tm.print(os_);

    ReplicatedMatrix2SquareLocalMatrices::printTimers(os_);
    LocalMatrices2ReplicatedMatrix::printTimers(os_);

    MGmol_MPI::printTimers(os_);

    g_kbpsi_->printTimers(os_);

    get_kbpsi_tm.print(os_);
    Hamiltonian<OrbitalsType>::apply_Hloc_tm().print(os_);
    computeHij_tm_.print(os_);
    Rho<OrbitalsType>::printTimers(os_);
    XConGrid::get_xc_tm_.print(os_);
    get_Hpsi_and_Hij_tm_.print(os_);
    get_res_tm_.print(os_);
    comp_res_tm_.print(os_);
    vnlpsi_tm.print(os_);
#ifdef MGMOL_USE_SCALAPACK
    dist_matrix::SubMatrices<double>::printTimers(os_);
    DistMatrix2SquareLocalMatrices::printTimers(os_);
    dist_matrix::SparseDistMatrix<DISTMATDTYPE>::printTimers(os_);
    dist_matrix::DistMatrix<DISTMATDTYPE>::printTimers(os_);

    get_MLWF_tm.print(os_);
    get_NOLMO_tm.print(os_);
#endif
    Energy<OrbitalsType>::eval_te_tm().print(os_);
    Electrostatic::solve_tm().print(os_);
    PoissonInterface::printTimers(os_);
    AndersonMix<OrbitalsType>::update_tm().print(os_);
    proj_matrices_->printTimers(os_);
    ShortSightedInverse::printTimers(os_);
    if (std::is_same<OrbitalsType, ExtendedGridOrbitals<ORBDTYPE>>::value)
    {
#ifdef MGMOL_USE_SCALAPACK
        MVPSolver<ExtendedGridOrbitals<ORBDTYPE>,
            dist_matrix::DistMatrix<DISTMATDTYPE>>::printTimers(os_);
#endif
        MVPSolver<ExtendedGridOrbitals<ORBDTYPE>,
            ReplicatedMatrix>::printTimers(os_);
    }
    VariableSizeMatrixInterface::printTimers(os_);
    DataDistribution::printTimers(os_);
    PackedCommunicationBuffer::printTimers(os_);
    LinearSolverMatrix<lsdatatype>::printTimers(os_);
    PreconILU<pcdatatype>::printTimers(os_);
    LinearSolver::printTimers(os_);
    Table::printTimers(os_);
    LocalMatrices<MATDTYPE, MemorySpace::Host>::printTimers(os_);
    Power<LocalVector<double, MemorySpace::Host>,
        SquareLocalMatrices<double, MemorySpace::Host>>::printTimers(os_);
    SP2::printTimers(os_);
    if (lrs_) lrs_->printTimers(os_);
    local_cluster_->printTimers(os_);
    forces_->printTimers(os_);
    if (ct.OuterSolver() == OuterSolverType::ABPG)
        ABPG<OrbitalsType>::printTimers(os_);
#ifdef MGMOL_USE_SCALAPACK
    else if (ct.OuterSolver() == OuterSolverType::NLCG)
        GrassmanLineMinimization<OrbitalsType>::printTimers(os_);
#endif
    adaptLR_tm_.print(os_);
    updateCenters_tm.print(os_);
    md_iterations_tm.print(os_);
    md_tau_tm.print(os_);
    md_moveVnuc_tm.print(os_);
    init_nuc_tm_.print(os_);
    md_updateMasks_tm.print(os_);
    md_extrapolateOrbitals_tm.print(os_);
    md_updateDMandEnergy_tm.print(os_);
    quench_tm.print(os_);
    evnl_tm_.print(os_);
    ions_setupInteractingIons_tm.print(os_);
    ions_setup_tm.print(os_);
    init_tm_.print(os_);
    dump_tm_.print(os_);
    setup_tm_.print(os_);
    HDFrestart::printTimers(os_);
#ifdef HAVE_MAGMA
    BlockVector<ORBDTYPE, MemorySpace::Device>::printTimers(os_);
#endif
    PowerGen<ReplicatedMatrix, ReplicatedVector>::printTimers(os_);
    DavidsonSolver<ExtendedGridOrbitals<ORBDTYPE>,
        ReplicatedMatrix>::printTimers(os_);
    ChebyshevApproximation<ReplicatedMatrix>::printTimers(os_);
#ifdef MGMOL_USE_SCALAPACK
    PowerGen<dist_matrix::DistMatrix<double>,
        dist_matrix::DistVector<double>>::printTimers(os_);
#endif
    BlockVector<ORBDTYPE, MemorySpace::Host>::printTimers(os_);
    if (ct.rmatrices)
    {
        DavidsonSolver<ExtendedGridOrbitals<ORBDTYPE>,
            ReplicatedMatrix>::printTimers(os_);
        ChebyshevApproximation<ReplicatedMatrix>::printTimers(os_);
    }
#ifdef MGMOL_USE_SCALAPACK
    else
    {
        DavidsonSolver<ExtendedGridOrbitals<ORBDTYPE>,
            dist_matrix::DistMatrix<DISTMATDTYPE>>::printTimers(os_);
        ChebyshevApproximation<
            dist_matrix::DistMatrix<DISTMATDTYPE>>::printTimers(os_);
    }
#endif
    MGOrbitalsPreconditioning<OrbitalsType, float>::printTimers(os_);
    MGOrbitalsPreconditioning<OrbitalsType, double>::printTimers(os_);
    MDfiles::printTimers(os_);
    ChebyshevApproximationInterface::printTimers(os_);
}

template <class OrbitalsType>
void MGmol<OrbitalsType>::initKBR()
{
    Mesh* mymesh           = Mesh::instance();
    const pb::Grid& mygrid = mymesh->grid();
    Control& ct            = *(Control::instance());
    Potentials& pot        = hamiltonian_->potential();

    const double hmax = mygrid.hmax();
    std::vector<Species>& sp(ct.getSpecies());
    if (onpe0 && ct.verbose > 0)
        os_ << "initKBR() for " << sp.size() << " species, hmax=" << hmax
            << std::endl;

    int mpi_rank;
    MPI_Comm_rank(comm_, &mpi_rank);
    int mpi_size;
    MPI_Comm_size(comm_, &mpi_size);

    // Distributed loop over species
    short counter = 0;
    for (std::vector<Species>::iterator isp = sp.begin(); isp != sp.end();
         ++isp)
    {
        if (counter % mpi_size == mpi_rank)
        {
            isp->initPotentials(pot.pot_type(counter), hmax, true);
        }
        counter++;
    }

    // now broadcast initialized potentials to all MPI tasks
    // from task which did the work
    counter = 0;
    for (std::vector<Species>::iterator isp = sp.begin(); isp != sp.end();
         ++isp)
    {
        isp->syncKBP(counter % mpi_size);
        counter++;
    }
}

template <class OrbitalsType>
double MGmol<OrbitalsType>::get_evnl(const Ions& ions)
{
    evnl_tm_.start();
    Control& ct = *(Control::instance());

    double val = std::numeric_limits<double>::signaling_NaN();
    if (ct.short_sighted)
    {
        std::shared_ptr<ProjectedMatricesSparse> projmatrices
            = std::dynamic_pointer_cast<ProjectedMatricesSparse>(
                proj_matrices_);
        assert(projmatrices);

        val = g_kbpsi_->getEvnl(ions, projmatrices.get());
    }
    else
    {
        if (ct.rmatrices)
        {
            std::shared_ptr<ProjectedMatrices<ReplicatedMatrix>> projmatrices
                = std::dynamic_pointer_cast<
                    ProjectedMatrices<ReplicatedMatrix>>(proj_matrices_);
            assert(projmatrices);

            val = g_kbpsi_->getEvnl(ions, projmatrices.get());
        }
#ifdef MGMOL_USE_SCALAPACK
        else
        {
            std::shared_ptr<
                ProjectedMatrices<dist_matrix::DistMatrix<DISTMATDTYPE>>>
                projmatrices = std::dynamic_pointer_cast<
                    ProjectedMatrices<dist_matrix::DistMatrix<DISTMATDTYPE>>>(
                    proj_matrices_);
            assert(projmatrices);

            val = g_kbpsi_->getEvnl(ions, projmatrices.get());
        }
#endif
    }

    evnl_tm_.stop();

    return val;
}

template <class OrbitalsType>
double MGmol<OrbitalsType>::getTotalEnergy()
{
    return total_energy_;
}

template <class OrbitalsType>
void MGmol<OrbitalsType>::setup()
{
    total_tm_.start();
    setup_tm_.start();

    Control& ct     = *(Control::instance());
    MGmol_MPI& mmpi = *(MGmol_MPI::instance());

    if (ct.verbose > 0)
        printWithTimeStamp("MGmol<OrbitalsType>::setup()...", os_);

    if (ct.verbose > 0) printWithTimeStamp("Setup VH...", os_);
    electrostat_ = std::shared_ptr<Electrostatic>(new Electrostatic(
        ct.getPoissonFDtype(), ct.bcPoisson, ct.screening_const));
    electrostat_->setup(ct.vh_init);

    rho_ = std::shared_ptr<Rho<OrbitalsType>>(new Rho<OrbitalsType>());
    rho_->setVerbosityLevel(ct.verbose);

#ifdef HAVE_MAGMA
    int ierr = initial<MemorySpace::Device>();
#else
    int ierr = initial<MemorySpace::Host>();
#endif

    if (ierr < 0) mmpi.abort();

    // Write header to stdout
    write_header();
    if (ct.verbose > 5)
    {
        Mesh* mymesh             = Mesh::instance();
        const pb::PEenv& myPEenv = mymesh->peenv();
        myPEenv.printPEnames(os_);
    }

    if (ct.verbose > 0)
        printWithTimeStamp("MGmol<OrbitalsType>::setup done...", os_);

    setup_tm_.stop();
}

template <class OrbitalsType>
void MGmol<OrbitalsType>::dumpRestart()
{
    Control& ct = *(Control::instance());

    // Save data to restart file
    if (ct.out_restart_info > 0)
    {
        Mesh* mymesh           = Mesh::instance();
        const pb::Grid& mygrid = mymesh->grid();
        unsigned gdim[3] = { mygrid.gdim(0), mygrid.gdim(1), mygrid.gdim(2) };
        const pb::PEenv& myPEenv = mymesh->peenv();

        // create restart file
        std::string filename(std::string(ct.out_restart_file));
        HDFrestart h5restartfile(
            filename, myPEenv, gdim, ct.out_restart_file_type);

        int ierr = write_hdf5(
            h5restartfile, rho_->rho_, *ions_, *current_orbitals_, lrs_);

        if (ierr < 0)
            os_ << "WARNING: writing restart data failed!!!" << std::endl;

#ifdef MGMOL_HAS_LIBROM
        // Save orbital snapshots
        if (ct.getROMOptions().save_librom_snapshot > 0
            && ct.AtomsDynamic() == AtomsDynamicType::Quench)
        {
            ierr = save_orbital_snapshot(filename, *current_orbitals_);

            if (ierr < 0)
                os_ << "WARNING: writing ROM snapshot data failed!!!"
                    << std::endl;
        }
#endif
    }
}

template <class OrbitalsType>
void MGmol<OrbitalsType>::cleanup()
{
    closing_tm_.start();

    Control& ct = *(Control::instance());

    printTimers();

    // Save data to restart file
    if (!ct.AtomsMove())
    {
        dumpRestart();
    }

    MPI_Barrier(comm_);
    closing_tm_.stop();
    total_tm_.stop();
    if (onpe0)
    {
        os_ << " **************************************************************"
               "****** "
            << std::endl;
    }
    closing_tm_.print(os_);
    total_tm_.print(os_);
}

template <>
void MGmol<LocGridOrbitals<ORBDTYPE>>::projectOutKernel(
    LocGridOrbitals<ORBDTYPE>& phi)
{
    assert(aomm_ != nullptr);
    aomm_->projectOut(phi);
}

template <class OrbitalsType>
void MGmol<OrbitalsType>::projectOutKernel(OrbitalsType& phi)
{
    (void)phi;

    return;
}

template <class OrbitalsType>
void MGmol<OrbitalsType>::precond_mg(OrbitalsType& phi)
{
    assert(orbitals_precond_);

    Control& ct = *(Control::instance());

    Potentials& pot            = hamiltonian_->potential();
    pb::Lap<ORBDTYPE>* lapOper = hamiltonian_->lapOper();

    const short precision = ct.precond_precision_;
    if (precision == 32)
    {
        using OrbitalsPrecond = MGOrbitalsPreconditioning<OrbitalsType, float>;

        std::shared_ptr<OrbitalsPrecond> orbitals_precond
            = std::dynamic_pointer_cast<OrbitalsPrecond>(orbitals_precond_);

        orbitals_precond->setGamma(
            *lapOper, pot, ct.getMGlevels(), proj_matrices_.get());
    }
    else if (precision == 64)
    {
        using OrbitalsPrecond = MGOrbitalsPreconditioning<OrbitalsType, double>;

        std::shared_ptr<OrbitalsPrecond> orbitals_precond
            = std::dynamic_pointer_cast<OrbitalsPrecond>(orbitals_precond_);

        orbitals_precond->setGamma(
            *lapOper, pot, ct.getMGlevels(), proj_matrices_.get());
    }
    else
    {
        std::cerr << "Precision " << precision
                  << " not supported for orbitals preconditioner!!!"
                  << std::endl;
    }

    orbitals_precond_->precond(phi);
}

template <class OrbitalsType>
double MGmol<OrbitalsType>::computeResidual(OrbitalsType& phi,
    OrbitalsType& hphi, Ions& ions, OrbitalsType& res,
    const KBPsiMatrixSparse* const kbpsi, const bool print_residual,
    const bool norm_res)

{
    comp_res_tm_.start();
    // os_<<"computeResidual()"<<endl;

    Control& ct(*(Control::instance()));

    proj_matrices_->computeInvB();

    // get H*phi stored in hphi
    // and psi^T H psi in Hij
    getHpsiAndTheta(ions, phi, hphi, kbpsi);

    double norm2Res
        = computeConstraintResidual(phi, hphi, res, print_residual, norm_res);

    if (ct.isSpreadFunctionalEnergy()) addResidualSpreadPenalty(phi, res);

    comp_res_tm_.stop();

    return norm2Res;
}

//////////////////////////////////////////////////////////////////////////////
// compute res using psi and hpsi
template <class OrbitalsType>
void MGmol<OrbitalsType>::computeResidualUsingHPhi(OrbitalsType& psi,
    const OrbitalsType& hphi, OrbitalsType& res, const bool applyB)
{
    assert(psi.isCompatibleWith(hphi));
    assert(psi.isCompatibleWith(res));

    get_res_tm_.start();

    proj_matrices_->updateSubMatT();

    SquareLocalMatrices<MATDTYPE, MemorySpace::Host>& localT(
        proj_matrices_->getLocalT());

    pb::Lap<ORBDTYPE>* lapop = hamiltonian_->lapOper();
    const int ncolors        = psi.chromatic_number();
    if (ncolors > 0)
    {
        // compute B*psi and store in tmp
        ORBDTYPE* old_storage = nullptr;
        std::vector<ORBDTYPE> tmp;
#ifdef HAVE_MAGMA
        ORBDTYPE* tmp_dev;
#endif
        if (applyB)
        {
            const int ld = psi.getLda();
            tmp.resize(ld * ncolors);
            psi.setDataWithGhosts();
            psi.trade_boundaries();
            for (int i = 0; i < ncolors; i++)
            {
                lapop->rhs(psi.getFuncWithGhosts(i), &tmp[0] + ld * i);
            }

            // psi points to tmp temporarily
            old_storage = psi.getPsi(0);
#ifdef HAVE_MAGMA
            tmp_dev
                = MemorySpace::Memory<ORBDTYPE, MemorySpace::Device>::allocate(
                    ld * ncolors);
            MemorySpace::copy_to_dev(tmp, tmp_dev);
            psi.set_storage(tmp_dev);
#else
            psi.set_storage(&tmp[0]);
#endif
        }

        // get B*phi*theta and store it in res in [Ry]
        // (even if Ritz functions mode)
        psi.multiplyByMatrix(localT, res);

        if (applyB)
        {
            psi.set_storage(old_storage);
#ifdef HAVE_MAGMA
            MemorySpace::Memory<ORBDTYPE, MemorySpace::Device>::free(tmp_dev);
#endif
        }

        // res = (B*phi*theta - H*phi) in [Ry]
        res.axpy((ORBDTYPE)(-1.), hphi);
    }

    get_res_tm_.stop();
}

//////////////////////////////////////////////////////////////////////////////

template <class OrbitalsType>
double MGmol<OrbitalsType>::computeConstraintResidual(OrbitalsType& orbitals,
    const OrbitalsType& hphi, OrbitalsType& res, const bool print_residual,
    const bool compute_norm_res)
{
    Control& ct(*(Control::instance()));

    computeResidualUsingHPhi(orbitals, hphi, res, ct.Mehrstellen());

    res.applyCorrMask();

    double normRes = -1.;
    if (print_residual || compute_norm_res)
    {
        if (ct.checkMaxResidual())
        {
            normRes = 0.5 * res.maxAbsValue(); // factor 0.5 for Rydberg to
                                               // Hartree conversion
            if (onpe0 && print_residual)
                os_ << std::setprecision(2) << std::scientific
                    << "max. |Residual_ij| = " << normRes << std::endl;
        }
        else
        {
            double norm2Res = res.dotProduct(res);
            normRes = 0.5 * std::sqrt(norm2Res); // factor 0.5 for Rydberg to
                                                 // Hartree conversion
            if (onpe0 && print_residual)
                os_ << std::setprecision(2)
                    << std::scientific
                    //                   <<"|| Relative residual || =
                    //                   "<<normRes/ct.getNel()<<endl;
                    << "|| Residual || = " << normRes << std::endl;
        }
    }

    return normRes;
}

//////////////////////////////////////////////////////////////////////////////
// Get preconditioned residual in res_orbitals
template <class OrbitalsType>
double MGmol<OrbitalsType>::computePrecondResidual(OrbitalsType& phi,
    OrbitalsType& hphi, OrbitalsType& res, Ions& ions, KBPsiMatrixSparse* kbpsi,
    const bool print_residual, const bool norm_res)

{
    Control& ct = *(Control::instance());

    double norm2Res = computeResidual(
        phi, hphi, ions, res, kbpsi, print_residual, norm_res);

    if (ct.withPreconditioner())
    {
        // PRECONDITIONING
        // compute the preconditioned steepest descent direction
        // -> res
        precond_mg(res);
    }

    return norm2Res;
}

//    Function to update potentials vh and vxc:
//
//    The new potentials are computed as a linear combination
//    of the old ones (input "vh" and "vxc") and the ones
//    corresponding to the input "rho".
template <class OrbitalsType>
void MGmol<OrbitalsType>::update_pot(
    const pb::GridFunc<POTDTYPE>& vh_init, const Ions& ions)
{
    electrostat_->setupInitialVh(vh_init);
    update_pot(ions);
}

template <class OrbitalsType>
void MGmol<OrbitalsType>::update_pot(const Ions& ions)
{
#ifdef PRINT_OPERATIONS
    if (onpe0) os_ << "Update potentials" << std::endl;
#endif

    Control& ct     = *(Control::instance());
    Potentials& pot = hamiltonian_->potential();

    // Update exchange-correlation potential
    xcongrid_->update();

    // Generate new hartree potential
    electrostat_->computeVh(ions, *rho_, pot);

    const bool flag_mixing = (fabs(ct.mix_pot - 1.) > 1.e-3);

    // update total potential
    if (flag_mixing)
    {
        pot.computeDeltaV(rho_->rho_);
        pot.updateVtot(ct.mix_pot);
    }
    else
        pot.updateVtot(rho_->rho_);
}

template <class OrbitalsType>
void MGmol<OrbitalsType>::addResidualSpreadPenalty(
    OrbitalsType& phi, OrbitalsType& res)
{
    assert(spread_penalty_);

    spread_penalty_->addResidual(phi, res);
}

template <class OrbitalsType>
void MGmol<OrbitalsType>::getAtomicPositions(std::vector<double>& tau)
{
    ions_->getPositions(tau);
}

template <class OrbitalsType>
void MGmol<OrbitalsType>::getAtomicNumbers(std::vector<short>& an)
{
    ions_->getAtomicNumbers(an);
}

template <class OrbitalsType>
void MGmol<OrbitalsType>::updateDMandEnergy(
    OrbitalsType& orbitals, Ions& ions, double& eks)
{
    // initialize electronic density
    rho_->update(orbitals);

    // initialize potential
    update_pot(ions);

    // initialize projected matrices
    updateHmatrix(orbitals, ions);
    proj_matrices_->updateThetaAndHB();

    // compute DM
    std::shared_ptr<DMStrategy<OrbitalsType>> dm_strategy(
        DMStrategyFactory<OrbitalsType,
            dist_matrix::DistMatrix<double>>::create(comm_, os_, ions,
            rho_.get(), energy_.get(), electrostat_.get(), hamiltonian_.get(),
            this, proj_matrices_.get(), &orbitals));

    dm_strategy->update(orbitals);

    // evaluate energy and forces
    double ts = 0.;
    eks       = energy_->evaluateTotal(
        ts, proj_matrices_.get(), ions, orbitals, 2, os_);
}

template <class OrbitalsType>
double MGmol<OrbitalsType>::evaluateEnergyAndForces(
    const std::vector<double>& tau, const std::vector<short>& atnumbers,
    std::vector<double>& forces)
{
    return evaluateEnergyAndForces(current_orbitals_, tau, atnumbers, forces);
}

template <class OrbitalsType>
double MGmol<OrbitalsType>::evaluateEnergyAndForces(Orbitals* orbitals,
    const std::vector<double>& tau, const std::vector<short>& atnumbers,
    std::vector<double>& forces)
{
    assert(tau.size() == 3 * atnumbers.size());

    Control& ct = *(Control::instance());

    // create a new temporary Ions object to be used for
    // energy end forces calculation
    Mesh* mymesh            = Mesh::instance();
    const pb::Grid& mygrid  = mymesh->grid();
    const double lattice[3] = { mygrid.ll(0), mygrid.ll(1), mygrid.ll(2) };
    Ions ions(tau, atnumbers, lattice, ions_->getSpecies());

    setupPotentials(ions);

    double eks              = 0.;
    OrbitalsType* dorbitals = dynamic_cast<OrbitalsType*>(orbitals);
    quench(*dorbitals, ions, ct.max_electronic_steps, 20, eks);

    force(*dorbitals, ions);

    ions.getForces(forces);

    return eks;
}

template <class OrbitalsType>
double MGmol<OrbitalsType>::evaluateDMandEnergyAndForces(Orbitals* orbitals,
    const std::vector<double>& tau, const std::vector<short>& atnumbers,
    std::vector<double>& forces)
{
    Control& ct = *(Control::instance());

    OrbitalsType* dorbitals = dynamic_cast<OrbitalsType*>(orbitals);

    // create a new temporary Ions object to be used for
    // energy end forces calculation
    Mesh* mymesh            = Mesh::instance();
    const pb::Grid& mygrid  = mymesh->grid();
    const double lattice[3] = { mygrid.ll(0), mygrid.ll(1), mygrid.ll(2) };
    Ions ions(tau, atnumbers, lattice, ions_->getSpecies());

    setupPotentials(ions);

    // initialize electronic density
    rho_->update(*dorbitals);

    // initialize potential
    update_pot(ions);

    // initialize projected matrices
    updateHmatrix(*dorbitals, ions);
    proj_matrices_->updateThetaAndHB();

    // compute DM
    if (ct.rmatrices)
    {
        std::shared_ptr<DMStrategy<OrbitalsType>> dm_strategy(
            DMStrategyFactory<OrbitalsType, ReplicatedMatrix>::create(comm_,
                os_, ions, rho_.get(), energy_.get(), electrostat_.get(),
                hamiltonian_.get(), this, proj_matrices_.get(), dorbitals));

        dm_strategy->update(*dorbitals);
    }
#ifdef MGMOL_USE_SCALAPACK
    else
    {
        std::shared_ptr<DMStrategy<OrbitalsType>> dm_strategy(
            DMStrategyFactory<OrbitalsType,
                dist_matrix::DistMatrix<double>>::create(comm_, os_, ions,
                rho_.get(), energy_.get(), electrostat_.get(),
                hamiltonian_.get(), this, proj_matrices_.get(), dorbitals));

        dm_strategy->update(*dorbitals);
    }
#endif

    // evaluate energy and forces
    double ts  = 0.;
    double eks = energy_->evaluateTotal(
        ts, proj_matrices_.get(), ions, *dorbitals, 2, os_);

    force(*dorbitals, ions);

    ions.getForces(forces);

    return eks;
}

template class MGmol<LocGridOrbitals<ORBDTYPE>>;
template class MGmol<ExtendedGridOrbitals<ORBDTYPE>>;
template int MGmol<LocGridOrbitals<ORBDTYPE>>::initial<MemorySpace::Host>();
template int
MGmol<ExtendedGridOrbitals<ORBDTYPE>>::initial<MemorySpace::Host>();
#ifdef HAVE_MAGMA
template int MGmol<LocGridOrbitals<ORBDTYPE>>::initial<MemorySpace::Device>();
template int
MGmol<ExtendedGridOrbitals<ORBDTYPE>>::initial<MemorySpace::Device>();
#endif

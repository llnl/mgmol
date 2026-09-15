// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#ifndef MGMOL_H
#define MGMOL_H

#include "mgmol_config.h"

#include "Energy.h"
#include "GridFuncVector.h"
#include "Hamiltonian.h"
#include "Lap.h"

#include "Control.h"
#include "MGmol_prototypes.h"
#include <iostream>

// inline double one(const double r){ return 1.; }

inline double linear(const double r) { return 1. - r; }

class XConGrid;
class Electrostatic;
class ConstraintSet;
class MLWFTransform;
class NOLMOTransform;
class OrbitalsTransform;
template <class OrbitalsType>
class ProjectedMatrices;
class ProjectedMatricesInterface;
class KBPsiMatrix;
class KBPsiMatrixSparse;
class MasksSet;

template <class OrbitalsType>
class IonicAlgorithm;

#include "AOMMprojector.h"
#include "ClusterOrbitals.h"
#include "DMStrategy.h"
#include "Energy.h"
#include "ExtendedGridOrbitals.h"
#include "Forces.h"
#include "Ions.h"
#include "LocGridOrbitals.h"
#include "MGOrbitalsPreconditioning.h"
#include "MGmolInterface.h"
#include "OrbitalsExtrapolation.h"
#include "Rho.h"
#include "SpreadPenaltyInterface.h"
#include "SpreadsAndCenters.h"

#include <memory>

template <class OrbitalsType>
class MGmol : public MGmolInterface
{
private:
    std::ostream& os_;

    MPI_Comm comm_;

    std::shared_ptr<XConGrid> xcongrid_;

    OrbitalsType* current_orbitals_;

    std::shared_ptr<AOMMprojector> aomm_;

    std::shared_ptr<Ions> ions_;

    std::shared_ptr<Rho<OrbitalsType>> rho_;

    std::shared_ptr<Energy<OrbitalsType>> energy_;

    std::shared_ptr<Hamiltonian<OrbitalsType>> hamiltonian_;

    std::shared_ptr<Forces<OrbitalsType>> forces_;

    std::shared_ptr<MasksSet> currentMasks_;
    std::shared_ptr<MasksSet> corrMasks_;

    std::shared_ptr<ProjectedMatricesInterface> proj_matrices_;

    std::shared_ptr<IonicAlgorithm<OrbitalsType>> geom_optimizer_;

    std::shared_ptr<LocalizationRegions> lrs_;

    std::shared_ptr<ClusterOrbitals> local_cluster_;

    std::shared_ptr<KBPsiMatrixSparse> g_kbpsi_;

    std::shared_ptr<SpreadsAndCenters<OrbitalsType>> spreadf_;

    std::shared_ptr<SpreadPenaltyInterface<OrbitalsType>> spread_penalty_;

    std::shared_ptr<DMStrategy<OrbitalsType>> dm_strategy_;

    std::shared_ptr<HDFrestart> h5f_file_;

    std::shared_ptr<OrbitalsPreconditioning<OrbitalsType>> orbitals_precond_;

    double total_energy_;
    std::shared_ptr<ConstraintSet> constraints_;

    std::shared_ptr<OrbitalsExtrapolation<OrbitalsType>> orbitals_extrapol_;

    float md_time_;
    int md_iteration_;

    // private functions
    void check_anisotropy();
    double get_charge(RHODTYPE* rho);
    void printTimers();
    int read_rho_and_pot_hdf5(HDFrestart& file, Rho<OrbitalsType>& rho);
    int read_restart_lrs(HDFrestart& h5f_file, const std::string& dset_name);
    int read_restart_data(
        HDFrestart& h5f_file, Rho<OrbitalsType>& rho, OrbitalsType& orbitals);
    void write_header();
    void getKBPsiAndHij(OrbitalsType& orbitals_i, OrbitalsType& orbitals_j,
        Ions& ions, KBPsiMatrixSparse* kbpsi,
        ProjectedMatricesInterface* projmatrices);
    void computeHnlPhiAndAdd2HPhi(Ions& ions, OrbitalsType& phi,
        OrbitalsType& hphi, const KBPsiMatrixSparse* const kbpsi);
    int dumpMDrestartFile(OrbitalsType& orbitals, Ions& ions,
        Rho<OrbitalsType>& rho, const bool write_extrapolated_wf,
        const short count);

#ifdef MGMOL_USE_SCALAPACK
    void swapColumnsVect(dist_matrix::DistMatrix<DISTMATDTYPE>& evect,
        const dist_matrix::DistMatrix<DISTMATDTYPE>& hb2N,
        const std::vector<double>& eval,
        dist_matrix::DistMatrix<DISTMATDTYPE>& work);
    void wftransform(OrbitalsType*, OrbitalsType*, Ions&);
#endif
    int readLRsFromInput(std::ifstream* tfile);
    void preWFextrapolation();
    void postWFextrapolation(OrbitalsType* orbitals);
    void computeResidualUsingHPhi(OrbitalsType& psi, const OrbitalsType& hphi,
        OrbitalsType& res, const bool applyB = true);

    template <typename MemorySpaceType>
    int initial();
    void initialMasks();
    int setupLRsFromInput(const std::string filename);

    int setupLRs(const std::string input_file) override;
    int setupFromInput(const std::string input_file) override;
    int setupConstraintsFromInput(const std::string input_file) override;

    // timers
    static Timer total_tm_;
    static Timer setup_tm_;
    static Timer closing_tm_;
    static Timer init_tm_;
    static Timer dump_tm_;
    static Timer adaptLR_tm_;
    static Timer evnl_tm_;
    static Timer get_res_tm_;
    static Timer computeHij_tm_;
    static Timer get_Hpsi_and_Hij_tm_;
    static Timer comp_res_tm_;
    static Timer init_nuc_tm_;

public:
    std::shared_ptr<Electrostatic> electrostat_;

    MGmol(MPI_Comm comm, std::ostream& os, std::string input_filename,
        std::string lrs_filename, std::string constraints_filename);

    ~MGmol() override;

    void setup() override;

    /* access functions */
    OrbitalsType* getOrbitals() { return current_orbitals_; }
    std::shared_ptr<Hamiltonian<OrbitalsType>> getHamiltonian()
    {
        return hamiltonian_;
    }
    std::shared_ptr<Rho<OrbitalsType>> getRho() { return rho_; }
    std::shared_ptr<Ions> getIons() { return ions_; }

    void run() override;

    /*
     * Evaluate the energy and forces for an atomic configuration
     * specified by tau (input)
     */
    double evaluateEnergyAndForces(const std::vector<double>& tau,
        const std::vector<short>& atnumbers,
        std::vector<double>& forces) override;

    /*
     * Evaluate the energy and forces for an atomic configuration
     * specified by tau (input), using the input "orbitals" as initial
     * guess for the wavefunctions
     */
    double evaluateEnergyAndForces(Orbitals* orbitals,
        const std::vector<double>& tau, const std::vector<short>& atnumbers,
        std::vector<double>& forces) override;

    /*
     * Evaluate the energy and forces for an atomic configuration
     * specified by tau (input), using the input "orbitals" as wavefunctions
     * (fixed)
     */
    double evaluateDMandEnergyAndForces(Orbitals* orbitals,
        const std::vector<double>& tau, const std::vector<short>& atnumbers,
        std::vector<double>& forces) override;

    /*
     * get internal atomic positions
     */
    void getAtomicPositions(std::vector<double>& tau) override;

    void getAtomicNumbers(std::vector<short>& an) override;

    void setupPotentials(Ions& ions);
    void initKBR();

    void printEigAndOcc();

    int readCoordinates(const std::string& filename, const bool cell_relative);
    double computeConstraintResidual(OrbitalsType& orbitals,
        const OrbitalsType& hphi, OrbitalsType& res, const bool print_residual,
        const bool norm_res);

    void md(OrbitalsType** orbitals, Ions& ions);
    void lbfgsrlx(OrbitalsType** orbitals, Ions& ions);

    template <class MatrixType>
    void computeHij(OrbitalsType& orbitals_i, OrbitalsType& orbitals_j,
        const Ions& ions, const KBPsiMatrixSparse* const kbpsi_i,
        const KBPsiMatrixSparse* const kbpsi_j, MatrixType& mat,
        const bool consolidate);

    template <class MatrixType>
    void computeHij(OrbitalsType& orbitals_i, OrbitalsType& orbitals_j,
        const Ions& ions, const KBPsiMatrixSparse* const kbpsi_i,
        MatrixType& mat, const bool consolidate);

    void computeHij(OrbitalsType& orbitals_i, OrbitalsType& orbitals_j,
        const Ions& ions, const KBPsiMatrixSparse* const kbpsi,
        ProjectedMatricesInterface*);

    void update_pot(const pb::GridFunc<POTDTYPE>& vh_init, const Ions& ions);
    void update_pot(const Ions& ions);
    int quench(OrbitalsType& orbitals, Ions& ions, const int max_steps,
        const int iprint, double& last_eks);
    int outerSolve(OrbitalsType& orbitals, OrbitalsType& work_orbitals,
        Ions& ions, const int max_steps, const int iprint, double& last_eks);
    void runfire(OrbitalsType** orbitals, Ions& ions);
    void moveVnuc(Ions& ions);
    void resetProjectedMatricesAndDM(OrbitalsType& orbitals, Ions& ions);
    int getMLWF(MLWFTransform& mlwft, OrbitalsType& orbitals,
        OrbitalsType& work_orbitals, const double dd, const bool apply_flag);
    bool rotateStatesPairsCommonCenter(
        OrbitalsType& orbitals, OrbitalsType& work_orbitals);
    bool rotateStatesPairsOverlap(
        OrbitalsType& orbitals, OrbitalsType& work_orbitals, const double);
    void disentangleOrbitals(
        OrbitalsType& orbitals, OrbitalsType& work_orbitals, Ions&, int&);
    void updateHmatrix(OrbitalsType& orbitals, Ions& ions);
    void getHpsiAndTheta(Ions& ions, OrbitalsType& phi, OrbitalsType& hphi,
        const KBPsiMatrixSparse* const kbpsi);
    void getHpsiAndTheta(Ions& ions, OrbitalsType& phi, OrbitalsType& hphi);
    double computePrecondResidual(OrbitalsType& phi, OrbitalsType& hphi,
        OrbitalsType& res, Ions& ions, KBPsiMatrixSparse* kbpsi,
        const bool print_residual, const bool norm_res);
    void addResidualSpreadPenalty(OrbitalsType& phi, OrbitalsType& res);
    int get_NOLMO(NOLMOTransform& noot, OrbitalsType& orbitals,
        OrbitalsType& work_orbitals, const double dd, const bool apply_flag);
    void adaptLR(const SpreadsAndCenters<OrbitalsType>* spreadf
#ifdef MGMOL_USE_SCALAPACK
        ,
        const OrbitalsTransform* ot
#endif
    );
    int update_masks();
    void move_orbitals(OrbitalsType** orbitals);
    int getMLWF2states(const int st1, const int st2, OrbitalsType& orbitals,
        OrbitalsType& work_orbitals);
    void extrapolate_centers(bool small_move);
    void compute_centers(bool small_move, OrbitalsType** orbitals);
    void extrapolate_orbitals(OrbitalsType** orbitals);
    OrbitalsType* new_orbitals_with_current_LRs(bool setup = true);
    void update_orbitals_LRs(OrbitalsType** orbitals);
    void clearOldOrbitals();
    void getKBPsiAndHij(OrbitalsType& orbitals, Ions& ions);
    int write_hdf5(const std::string& filename,
        std::vector<std::vector<RHODTYPE>>& rho, Ions& ions,
        OrbitalsType& orbitals);
    int write_hdf5(HDFrestart& h5f_file,
        std::vector<std::vector<RHODTYPE>>& rho, Ions& ions,
        OrbitalsType& orbitals, std::shared_ptr<LocalizationRegions> lrs);
    double get_evnl(const Ions& ions);
    void sebprintPositions();
    void sebprintForces();
    int nions() { return ions_->getNumIons(); }
    double getTotalEnergy();
    void cleanup() override;
    void geomOptimSetup();
    void geomOptimQuench();
    void geomOptimComputeForces();
    int geomOptimRun1Step();
    void geomOptimDumpRestart();
    void geomOptimSetForces(const std::vector<std::vector<double>>& f);
    short geomOptimCheckTolForces(const double tol_force);

    void finalEnergy();
    void printMM();

    void projectOutKernel(OrbitalsType& phi);

    void precond_mg(OrbitalsType& orbitals);
    double computeResidual(OrbitalsType& orbitals, OrbitalsType& work_orbitals,
        Ions& ions, OrbitalsType& res, const KBPsiMatrixSparse* const kbpsi,
        const bool print_residual, const bool norm_res);
    double computeResidual(OrbitalsType& orbitals, OrbitalsType& work_orbitals,
        Ions& ions, OrbitalsType& res, const bool print_residual,
        const bool norm_res)
    {
        return computeResidual(orbitals, work_orbitals, ions, res,
            g_kbpsi_.get(), print_residual, norm_res);
    }

    void applyAOMMprojection(OrbitalsType&);
    void force(OrbitalsType& orbitals, Ions& ions)
    {
        forces_->force(orbitals, ions);
    }
    void setPositions(const std::vector<double>& positions,
        const std::vector<short>& atnumbers)
    {
        ions_->setPositions(positions, atnumbers);
    }

    /*
     * simply dump current state
     */
    void dumpRestart() override;

    void loadRestartFile(const std::string filename);

    std::shared_ptr<ProjectedMatricesInterface> getProjectedMatrices() override
    {
        return proj_matrices_;
    }

#ifdef MGMOL_HAS_LIBROM
    int save_orbital_snapshot(std::string file_path, OrbitalsType& orbitals);
    void project_orbital(
        std::string file_path, int rdim, OrbitalsType& orbitals);
#endif
    void updateDMandEnergy(OrbitalsType& orbitals, Ions& ions, double& eks);
};
// Instantiate static variables here to avoid clang warnings
template <class OrbitalsType>
Timer MGmol<OrbitalsType>::adaptLR_tm_("MGmol::adaptLR");
template <class OrbitalsType>
Timer MGmol<OrbitalsType>::dump_tm_("MGmol::dump");
template <class OrbitalsType>
Timer MGmol<OrbitalsType>::total_tm_("MGmol::total");
template <class OrbitalsType>
Timer MGmol<OrbitalsType>::setup_tm_("MGmol::setup");
template <class OrbitalsType>
Timer MGmol<OrbitalsType>::closing_tm_("MGmol::closing");
template <class OrbitalsType>
Timer MGmol<OrbitalsType>::init_tm_("MGmol::init");
template <class OrbitalsType>
Timer MGmol<OrbitalsType>::evnl_tm_("MGmol::evnl");
template <class OrbitalsType>
Timer MGmol<OrbitalsType>::get_res_tm_("MGmol::comp_res_from_Hphi");
template <class OrbitalsType>
Timer MGmol<OrbitalsType>::computeHij_tm_("MGmol::computeHij");
template <class OrbitalsType>
Timer MGmol<OrbitalsType>::get_Hpsi_and_Hij_tm_("MGmol::get_Hpsi_and_Hij");
template <class OrbitalsType>
Timer MGmol<OrbitalsType>::comp_res_tm_("MGmol::comp_res");
template <class OrbitalsType>
Timer MGmol<OrbitalsType>::init_nuc_tm_("MGmol::init_nuc");
#endif

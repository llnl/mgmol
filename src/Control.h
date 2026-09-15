// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#ifndef CONTROL_H
#define CONTROL_H

#include "Species.h"
#include "Timeout.h"

/* enumeration and option variables for libROM */
#include "mgmol_config.h"
#include "rom_Control.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <mpi.h>

class Potentials;

namespace boost
{
namespace program_options
{
    class variables_map;
}
}

enum class OuterSolverType
{
    ABPG,
    PolakRibiere,
    NLCG,
    Davidson,
    UNDEFINED
};

enum class WFExtrapolationType
{
    Order2,
    Order3,
    Reversible,
    UNDEFINED
};

enum class AtomsDynamicType
{
    Quench,
    MD,
    LBFGS,
    FIRE,
    UNDEFINED
};

enum class DMNonLinearSolverType
{
    Mixing,
    MVP,
    HMVP,
    UNDEFINED
};

enum class DMEigensolverType
{
    Eigensolver,
    Chebyshev,
    SP2,
    UNDEFINED
};

enum class OrthoType
{
    Eigenfunctions,
    Nonorthogonal,
    Orthonormal,
    UNDEFINED
};

enum class PoissonFDtype
{
    h4M,
    h2,
    h4,
    h6,
    h8,
    h4MP
};

// Main control structure
class Control
{
private:
    static Control* pinstance_;

    // this class uses its own communicator to bcast options to
    // all tasks involved in run
    static MPI_Comm comm_global_;

    // spin of system
    static float total_spin_; // +/-1/2, +/-1, ...
    static bool with_spin_;

    int mype_;

    // pointer to array of Species
    std::vector<Species> sp_;

    // pseudopotentials filenames
    std::vector<std::string> pot_filenames_;
    std::vector<short> pseudopot_flags_;

    // Uses localized orbitals or not
    bool loc_mode_;

    // Number of electrons
    int nel_; // total
    int nempty_;

    // Number electrons for local spin
    // Could be a fraction if total number of e- is odd
    double nelspin_;

    // total charge of system ions+electrons
    short system_charge_;

    // occupation of each orbitals when prescribed
    std::vector<float> orbitals_occupations_;

    float min_distance_centers_;

    // threshold below which action is taken to reduce linear dependence between
    // functions at each MD step
    float threshold_eigenvalue_gram_;

    // threshold below which action is taken to reduce linear dependence between
    // functions during quench
    float threshold_eigenvalue_gram_quench_;

    // Max. distance between pairs for MLWF transform
    float pair_mlwf_distance_threshold_;

    // relative tolerance of KS energy convergence
    float conv_rtol_;

    short conv_criterion_;

    // multiplicative factor for spread penalty
    float spread_penalty_damping_;
    float spread_penalty_target_;
    float spread_penalty_alpha_;
    short spread_penalty_type_;

    short dm_use_old_;

    // Number of electronic steps per ionic step
    short max_electronic_steps_loose_;
    short max_electronic_steps_tight_;
    //
    // dielectric flag for Poisson solver
    // 0 = no diel. parameter with MG solver for Poisson
    // 1 = diel. parameter with MG solver for Poisson
    // 10 = no diel. parameter with PCG for Poisson
    // 11 = diel. parameter with PCG for Poisson
    short diel_flag_;

    // corloring algorithm
    // 0 =global RLF
    // 1 =global greedy
    // 10=local RLF
    // 11=local greedy
    short coloring_algo_;

    // preconditioning type
    // 10 = MG, block implementation
    short precond_type_;

    short it_algo_type_;

    short orbital_type_;

    short wf_extrapolation_;

    short atoms_dyn_;

    short dm_algo_;

    // flag to decide if condition number of Gram matrix
    // should be computed during quench (value 2) or
    // only at the end of quench (value 1)
    short computeCondGram_;

    // max. distance for atomic information to be communicated
    float maxDistanceAtomicInfo_;

    float aomm_radius_;
    float aomm_threshold_factor_;

    // rescaling factor for velocities at restart
    double rescale_v_;

    float overallocate_factor_;

    static std::string run_directory_;

    Timeout timeout_;

    // DM solver:
    // simple(0), MVP(1)
    short DM_solver_;

    Control();

    ~Control() {};
    Control(const Control& ct) { (void)ct; };

    void printRestartLink();

    /* libROM related options */
    ROMPrivateOptions rom_pri_option;

public:
    static Control* instance()
    {
        assert(comm_global_ != MPI_COMM_NULL);
        if (pinstance_ == nullptr)
        {
            pinstance_ = new Control();
        }
        return pinstance_;
    }
    static void deleteInstance()
    {
        if (pinstance_ != nullptr)
        {
            delete pinstance_;
            pinstance_ = nullptr;
        }
    }
    static void setup(const MPI_Comm comm, const bool with_spin,
        const float total_spin, std::string run_directory = ".");

    void setDefaultValues();
    bool withSpin() { return with_spin_; }

    bool globalColoring() const { return (coloring_algo_ / 10 == 0); }

    bool RLFColoring() const { return (coloring_algo_ % 10 == 0); }
    bool use_old_dm() const { return (dm_use_old_ == 1); }

    std::string getFullFilename(const std::string& filename)
    {
        return run_directory_ + "/" + filename;
    }

    int getNel() const { return nel_; }

    double getNelSpin() const
    {
        assert(nelspin_ >= 0.);
        return nelspin_;
    }

    float getSpin() const { return total_spin_; }

    void setNempty(const int nempty) { nempty_ = nempty; }

    bool prescribedOccupations() { return (!orbitals_occupations_.empty()); }

    void getOccupations(std::vector<double>& occ)
    {
        assert(occ.size() == orbitals_occupations_.size());

        std::copy(orbitals_occupations_.begin(), orbitals_occupations_.end(),
            occ.begin());
    }

    short getMGlevels() { return mg_levels_; }

    bool withPreconditioner() const { return (mg_levels_ >= 0); }

    void convergeTightly()
    {
        max_electronic_steps = max_electronic_steps_tight_;
    }

    void convergeLoosely()
    {
        max_electronic_steps = max_electronic_steps_loose_;
    }

    void print(std::ostream&);
    void printPoissonOptions(std::ostream& os);

    void sync(void);
    void adjust();
    int checkState();
    void readRestartInfo(std::ifstream* tfile);
    int readThermostatInfo(std::ifstream* tfile);
    void printThermostatInfo(std::ostream& os) const;
    void setNumst(const short myspin, const int nval);
    void setNumIons(const int num_ions);
    int setPreconditionerParameters(const short type, const float factor,
        const bool project_out, const short nlevels, const float);
    int setShortSightedSolverParameters(const float fact, const float stol,
        const float dtol, const short kim, const short itmax, const short lfil,
        const short maxfill, const short ilutype);
    void setSpreadRadius();
    bool checkTimeout() { return timeout_.check(); }

    bool occupationWidthIsZero() { return occ_width_ < 1.e-12; }

    void setLocMode(
        const float, const float, const float, const float, const float);

    bool isLocMode() const { return loc_mode_; }

    bool adaptiveLRs()
    {
        assert(wannier_transform_type >= 0);
        assert(lr_update >= 0);

        return (wannier_transform_type && loc_mode_ && lr_update);
    }

    bool adaptiveLRsizes()
    {
        assert(wannier_transform_type >= 0);
        assert(lr_update >= 0);

        return (wannier_transform_type && loc_mode_ && lr_update
                && lr_updates_type > 0);
    }

    float getMinDistanceCenters() const { return min_distance_centers_; }

    void setTolEigenvalueGram(const float tol);
    float getThresholdEigenvalueGram() const
    {
        assert(threshold_eigenvalue_gram_ >= 0.);

        return threshold_eigenvalue_gram_;
    }
    float getThresholdEigenvalueGramQuench() const
    {
        assert(threshold_eigenvalue_gram_quench_ >= 0.);

        return threshold_eigenvalue_gram_quench_;
    }

    float getThresholdDistancePairMLWF() const
    {
        assert(pair_mlwf_distance_threshold_ >= 0.);

        return pair_mlwf_distance_threshold_;
    }

    void global_exit();

    bool Mehrstellen() const { return (lap_type == 0 || lap_type == 10); }

    PoissonFDtype getPoissonFDtype() const { return poisson_lap_type_; }

    void setColoringAlgo(const short coloring_algo)
    {
        coloring_algo_ = coloring_algo;
    }

    void setDielAlgo(const short dielflag, const short nu1, const short nu2,
        const short nlev)
    {
        diel_flag_ = dielflag;

        diel            = diel_flag_ % 10;
        poisson_pc_nu1  = nu1;
        poisson_pc_nu2  = nu2;
        poisson_pc_nlev = nlev;
    }

    // 10 or larger means PCG, otherwise MG V-cycles
    bool MGPoissonSolver() { return (diel_flag_ / 10 == 0); }

    bool LangevinThermostat() { return (thermostat_type == 1); }

    //
    // data
    //
    // domain
    float lx_;
    float ly_;
    float lz_;
    float ox_;
    float oy_;
    float oz_;

    // mesh
    unsigned ngpts_[3];

    short wf_dyn; // quench method
    short wf_m; // number of wf to keep in memory

    int numst;

    short lrs_compute;
    short lrs_extrapolation;

    float betaAnderson;

    // Number of MG levels for preconditioning
    short mg_levels_;
    short mg_npresmoothing_;
    short mg_npostsmoothing_;

    // dielectric model for solvation
    short diel;
    // Parameters for MG solver/ preconditioner for Poisson problem
    short poisson_pc_nu1;
    short poisson_pc_nu2;
    short poisson_pc_nlev;

    /*!
     * Poisson preconditioner precision (32 or 64)
     */
    short poisson_pc_data_;

    PoissonFDtype poisson_lap_type_;

    short lap_type;

    short orthof; // orthogonalization frequency

    short precond_precision_;

    // screening constant for potential mixing
    float screening_const;

    bool restart_run;

    short num_species;
    int num_ions;

    // timestep in front of MG correction
    float precond_factor;
    bool precond_factor_computed;

    short project_out_psd;

    // Exchange-Correlation flag
    short xctype;

    // Actual number of steps done
    short steps;

    // convergence criterion for w.f.
    float conv_tol;

    // convergence criterion for w.f.
    float conv_tol_stop;

    // convergence criterion for forces
    float tol_forces;

    // Number of steps after which to perform checkpointing
    short checkpoint;

    // Frequency print residuals (in electronic steps)
    short iprint_residual;

    // Flag to compute Wannier centers at the end of the computation or not
    short wannier_transform_type;

    // Flag to compute Dipole moment or not
    short compute_dipole_;

    // Potential mixing parameters
    float mix_pot;
    float dm_mix;

    // Density matrix computation algorithm
    // 0 =diagonalization
    short dm_approx_order;
    short dm_approx_ndigits;
    short dm_approx_power_maxits;

    // SP2 options
    float dm_tol;

    // Initial number of v-cycles for hartree solution
    short vh_init;

    // Number of v-cycles for hartree solution
    short vh_its;

    // convergence tolerance for solving Poisson problem using PCG.
    float poisson_conv_tol;

    // Max number of changes of potential
    short max_changes_pot;

    // Localization radius
    float cut_radius;

    // electronic temperature
    float etemp_;
    float occ_width_;

    // transfer matrix flag
    short tmatrices;

    // replicated matrices
    short rmatrices;

    // Initialization with localized orbitals (1) or not (0)
    short init_loc;

    // Initialization type (0=random, 1=Gaussians)
    short init_type;
    float init_rc;

    // MD flag
    float dt;
    short enforceVmass0;
    short md_print_freq;
    std::string md_print_filename;

    // Number of electronic steps per ionic step
    short max_electronic_steps;
    short dm_inner_steps;

    // Number of steps
    short num_MD_steps;
    short MD_last_step_;

    // number of scf steps between localization centers updates
    short lr_updates_type;
    short lr_update;
    float tol_orb_centers_move;
    short lr_volume_calc;

    short dot_product_type;

    // with line minimization for electronic structure optimization
    short line_min;

    // temperature control
    short thermostat_type;
    float tkel;
    float thtime;
    float thwidth;

    // boundary conditions
    short bcWF[3];

    // boundary conditions
    short bcPoisson[3];
    short multipole_order;

    // dielectric parameters
    float e0_;
    float rho0_;
    float drho0_;

    // flag to reset Vh at beginning of each MD step
    short hartree_reset_;

    /*!
     * Components of electrostatic field
     */
    float ex_;
    float ey_;
    float ez_;

    // short-sighted computation of selected elements of inverse
    short short_sighted;
    short fgmres_kim;
    short fgmres_maxits;
    short ilu_type;
    short ilu_lof;
    short ilu_maxfil;

    float spread_factor;
    float spread_radius;
    float fgmres_tol;
    float ilu_droptol;

    bool parallel_transport;

    short use_kernel_functions;

    // restart info
    short restart_info;
    short out_restart_info;
    std::string restart_file;
    std::string out_restart_file;
    short out_restart_file_naming_strategy;
    short restart_file_type;
    short out_restart_file_type;

    short verbose;

    float load_balancing_alpha;
    float load_balancing_damping_tol;
    short load_balancing_max_iterations;
    short load_balancing_modulo;
    short write_clusters;
    std::string load_balancing_output_file;

    float reducedCutRadius() const { return 0.5 * cut_radius; }
    std::vector<Species>& getSpecies() { return sp_; }
    void setSpecies(Potentials& pot);
    void readPotFilenames(std::ifstream* tfile);
    void registerPotentials(Potentials& pot);

    bool isSpreadFunctionalActive() { return (spread_penalty_alpha_ > 0.); }
    float spreadPenaltyDampingFactor() const { return spread_penalty_damping_; }
    float spreadPenaltyAlphaFactor() const { return spread_penalty_alpha_; }

    float spreadPenaltyTarget() const { return spread_penalty_target_; }
    bool isSpreadFunctionalVolume() { return (spread_penalty_type_ == 1); }
    bool isSpreadFunctionalEnergy()
    {
        return (spread_penalty_type_ == 2 || spread_penalty_type_ == 3);
    }

    float initRadius() const
    {
        float cut_init = cut_radius;
        if (init_type == 0) // random
        {
            if (cut_radius > 2.) cut_init = 2.;
        }
        return cut_init;
    }

    bool computeCondGramQuench() const { return (computeCondGram_ > 1); }
    bool computeCondGramMD() const { return (computeCondGram_ > 0); }
    void setTolEnergy();

    float maxDistanceAtomicInfo() const { return maxDistanceAtomicInfo_; }
    int checkNLrange();
    int checkOptions();
    void setOptions(const boost::program_options::variables_map& vm);
    bool fullyOccupied()
    {
        return ((static_cast<double>(numst) - nelspin_) < 1.e-8);
    }

    float AOMMradius() const { return aomm_radius_; }
    float AOMMthresholdFactor() const { return aomm_threshold_factor_; }
    double VelocityScalingFactor() const { return rescale_v_; }
    float orbitalsOverallocateFactor() const { return overallocate_factor_; }

    bool checkResidual() const { return (conv_criterion_ > 0); }
    bool checkMaxResidual() const { return (conv_criterion_ == 2); }
    bool resetVH() const { return (hartree_reset_ > 0); }

    OuterSolverType OuterSolver()
    {
        switch (it_algo_type_)
        {
            case 0:
                return OuterSolverType::ABPG;
            case 1:
                return OuterSolverType::NLCG;
            case 2:
                return OuterSolverType::Davidson;
            case 3:
                return OuterSolverType::PolakRibiere;
            default:
                return OuterSolverType::UNDEFINED;
        }
    }

    WFExtrapolationType WFExtrapolation()
    {
        switch (wf_extrapolation_)
        {
            case 0:
                return WFExtrapolationType::Reversible;
            case 1:
                return WFExtrapolationType::Order2;
            case 2:
                return WFExtrapolationType::Order3;
            default:
                return WFExtrapolationType::UNDEFINED;
        }
    }

    AtomsDynamicType AtomsDynamic()
    {
        switch (atoms_dyn_)
        {
            case 0:
                return AtomsDynamicType::Quench;
            case 2:
                return AtomsDynamicType::MD;
            case 6:
                return AtomsDynamicType::LBFGS;
            case 7:
                return AtomsDynamicType::FIRE;
            default:
                return AtomsDynamicType::UNDEFINED;
        }
    }

    DMNonLinearSolverType DM_solver() const
    {
        switch (DM_solver_)
        {
            case 0:
                return DMNonLinearSolverType::Mixing;
            case 1:
                return DMNonLinearSolverType::MVP;
            case 2:
                return DMNonLinearSolverType::HMVP;
            default:
                return DMNonLinearSolverType::UNDEFINED;
        }
    }

    DMEigensolverType DMEigensolver() const
    {
        switch (dm_algo_)
        {
            case 0:
                return DMEigensolverType::Eigensolver;
            case 1:
                return DMEigensolverType::Chebyshev;
            case 2:
                return DMEigensolverType::SP2;
            default:
                return DMEigensolverType::UNDEFINED;
        }
    }

    OrthoType getOrthoType()
    {
        switch (orbital_type_)
        {
            case 0:
                return OrthoType::Eigenfunctions;
            case 1:
                return OrthoType::Nonorthogonal;
            case 2:
                return OrthoType::Orthonormal;
            default:
                return OrthoType::UNDEFINED;
        }
    }

    bool AtomsMove() { return (atoms_dyn_ != 0); }

    /* ROM-related options */
    void setROMOptions(const boost::program_options::variables_map& vm);
    void syncROMOptions();
    const ROMPrivateOptions getROMOptions() { return rom_pri_option; }
};

#endif

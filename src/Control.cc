// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "global.h"

#include "Control.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <sys/stat.h>
#include <unistd.h>

#include <boost/program_options.hpp>
#include <utility>

#include <mpi.h>

#include "MGmol_MPI.h"
#include "Potentials.h"
#include "tools.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

Control* Control::pinstance_   = nullptr;
MPI_Comm Control::comm_global_ = MPI_COMM_NULL;
float Control::total_spin_     = 0.;
std::string Control::run_directory_(".");
bool Control::with_spin_ = false;

template <typename T>
std::vector<T> parse_list(std::string s)
{
    for (char& c : s)
    {
        if (c == ',') c = ' ';
    }

    std::istringstream iss(s);
    std::vector<T> result;
    T value;

    while (iss >> value)
    {
        result.push_back(value);
    }

    iss >> std::ws;
    if (!iss.eof())
    {
        throw std::runtime_error("Invalid list: '" + s + "'");
    }

    return result;
}

template <typename T>
std::vector<T> get_list_option(
    const po::variables_map& vm, const std::string& name)
{
    if (!vm.count(name))
    {
        return {};
    }
    return parse_list<T>(vm[name].as<std::string>());
}

Control::Control()
{
    assert(comm_global_ != MPI_COMM_NULL);

    MPI_Comm_rank(comm_global_, &mype_);

    // default values
    lrs_extrapolation      = 1; // default
    lrs_compute            = 0;
    system_charge_         = 0.;
    poisson_pc_nu1         = 1;
    poisson_pc_nu2         = 1;
    poisson_pc_nlev        = 10;
    poisson_conv_tol       = 1.e-8;
    poisson_pc_data_       = 32;
    coloring_algo_         = 0;
    maxDistanceAtomicInfo_ = 8.;
    spread_factor          = 2.;
    conv_criterion_        = 0;
    steps                  = 0;
    dm_approx_order        = 500;
    dm_approx_ndigits      = 1;
    dm_approx_power_maxits = 100;
    wf_extrapolation_      = 1;
    verbose                = 0;

    // undefined values
    dm_algo_                         = -1;
    short_sighted                    = -1;
    it_algo_type_                    = -1;
    DM_solver_                       = -1;
    orbital_type_                    = -1;
    aomm_radius_                     = -1.;
    aomm_threshold_factor_           = -1.;
    rescale_v_                       = -1.;
    thtime                           = -1.;
    thwidth                          = -1.;
    nel_                             = -1;
    nempty_                          = -1;
    nelspin_                         = -1.;
    diel_flag_                       = -1;
    wf_dyn                           = -1;
    wf_m                             = -1;
    numst                            = -1;
    betaAnderson                     = 0.;
    diel                             = -1;
    lap_type                         = -1;
    precond_type_                    = -1;
    orthof                           = -1;
    init_loc                         = -1;
    init_type                        = -1;
    max_electronic_steps_loose_      = -1;
    max_electronic_steps_tight_      = -1;
    max_electronic_steps             = -1;
    lr_updates_type                  = -1;
    lr_update                        = -1;
    lr_volume_calc                   = -1;
    init_rc                          = -1.;
    out_restart_file_naming_strategy = 0;
    tol_orb_centers_move             = 10.e8;
    restart_file_type                = -1;
    restart_info                     = -1;
    xctype                           = -1;
    use_kernel_functions             = -1;
    dm_inner_steps                   = -1;
    enforceVmass0                    = -1;
    overallocate_factor_             = -1.;

    ngpts_[0]                         = -1;
    ngpts_[1]                         = -1;
    ngpts_[2]                         = -1;
    num_species                       = -1;
    num_ions                          = -1;
    cut_radius                        = -1.;
    bcPoisson[0]                      = -1;
    bcPoisson[1]                      = -1;
    bcPoisson[2]                      = -1;
    bcWF[0]                           = -1;
    bcWF[1]                           = -1;
    bcWF[2]                           = -1;
    out_restart_file_type             = -1;
    spread_radius                     = -1.;
    iprint_residual                   = -1;
    dot_product_type                  = -1;
    spread_penalty_damping_           = -1;
    spread_penalty_alpha_             = -1;
    spread_penalty_target_            = -1.;
    spread_penalty_type_              = -1;
    restart_run                       = false;
    conv_tol                          = -1.;
    thermostat_type                   = -1;
    hartree_reset_                    = -1;
    threshold_eigenvalue_gram_        = -1.;
    threshold_eigenvalue_gram_quench_ = -1.;
    pair_mlwf_distance_threshold_     = -1.;

    // data members set once for all (not accessible through interface)
    screening_const = 0.;
}

void Control::setup(const MPI_Comm comm_global, const bool with_spin,
    const float total_spin, std::string run_directory)
{
    assert(pinstance_ == nullptr);

    comm_global_ = comm_global;
    with_spin_   = with_spin;
    total_spin_  = total_spin;

    run_directory_ = std::move(run_directory);

    if (run_directory_.compare(".") != 0)
    {
        mode_t mode = (S_IRWXU | S_IRWXG | S_IRWXO);
        mkdir(run_directory_.c_str(), mode);
        (*MPIdata::sout) << "Create dir " << run_directory_ << std::endl;
    }
}

void Control::print(std::ostream& os)
{
    if (restart_info == 0) os << " Initial run" << std::endl;

    os << " Files used:" << std::endl;
    if (restart_info > 0)
        os << " Restart input file:  " << restart_file << std::endl;
    os << " Restart output file: " << out_restart_file << std::endl;

    if (diel)
    {
        os << " With dielectric medium:";
        os << std::setprecision(4) << std::scientific << " e0=" << e0_
           << ", rho0=" << rho0_ << ", drho0=" << drho0_ << std::endl;
    }

    os << " Boundary conditions for Poisson: " << bcPoisson[0] << ", "
       << bcPoisson[1] << ", " << bcPoisson[2] << std::endl;
    os << " Boundary conditions for Wavefunctions: " << bcWF[0] << ", "
       << bcWF[1] << ", " << bcWF[2] << std::endl;

    switch (getOrthoType())
    {
        case OrthoType::Eigenfunctions:
            os << " Works in Eigenfunctions basis" << std::endl;
            break;
        case OrthoType::Nonorthogonal:
            os << " Works in Nonorthogonal orbitals basis" << std::endl;
            break;
        case OrthoType::Orthonormal:
            os << " Works in Orthonormal orbitals basis" << std::endl;
            break;
        default:
            os << " Orbitals type undefined!!!" << std::endl;
            return;
    }

    switch (xctype)
    {
        case 0: // LDA Perdew Zunger 81
            os << "    XC using LDA with Perdew-Zunger 81" << std::endl;
            break;
        case 2:
            os << "    XC using PBE" << std::endl;
            break;
        default:
            os << "output: Unknown exchange-correlation functional"
               << std::endl;
            break;
    }
    os << " Number of species  = " << sp_.size() << std::endl;

    os << " Hartree: number initial sweeps            : " << vh_init
       << std::endl;
    os << " Hartree: number of sweeps for each Poisson: " << vh_its
       << std::endl;

    std::string str("");
    switch (conv_criterion_)
    {
        case 0:
            str = "delta energy";
            break;
        case 1:
            str = "residual";
            break;
        case 2:
            str = "maxResidual";
            break;
    }
    os << " KS convergence criterion: " << str << std::endl;
    os << " KS convergence value: " << std::scientific << std::setprecision(2)
       << conv_tol << std::endl;
    os << std::fixed;
    os << " Density matrix mixing = " << dm_mix << std::endl;
    os << std::setprecision(4) << std::scientific
       << " Density matrix tol = " << dm_tol << std::endl;
    if (DMEigensolver() == DMEigensolverType::Eigensolver)
    {
        os << " Density matrix computation algorithm = "
           << " Diagonalization " << std::endl;
    }
    else if (DMEigensolver() == DMEigensolverType::Chebyshev)
    {
        os << " Density matrix computation algorithm = "
           << " Chebyshev approximation " << std::endl;
        if (dm_approx_ndigits)
        {
            os << " Density matrix approximation precision digits = "
               << dm_approx_ndigits << std::endl;
        }
        else
        {
            os << " Density matrix approximation order = " << dm_approx_order
               << std::endl;
        }
        os << " Density matrix approximation: maxits for computing "
              "approximation interval = "
           << dm_approx_power_maxits << std::endl;
    }
    os << " Load balancing alpha for computing bias = " << load_balancing_alpha
       << std::endl;
    os << " Load balancing parameter for damping bias updates = "
       << load_balancing_damping_tol << std::endl;
    os << " Load balancing max. number of iterations = "
       << load_balancing_max_iterations << std::endl;
    os << " Control parameter for recomputing load balancing = "
       << load_balancing_modulo << std::endl;
    os << " Load balancing output filename = " << load_balancing_output_file
       << std::endl;
    if (loc_mode_)
        os << " Localization radius       = " << cut_radius << std::endl;
    os << std::endl;

    os << " preconditioner factor: " << precond_factor << std::endl;
    os << " preconditioner precision: " << precond_precision_ << std::endl;
    if (precond_type_ == 10)
    {
        os << " Multigrid preconditioning for wave functions:" << std::endl;
        os << " # of Multigrid levels     : " << mg_levels_ << std::endl;
        os << " # of pre-smoothing steps  : " << mg_npresmoothing_ << std::endl;
        os << " # of post-smoothing steps : " << mg_npostsmoothing_
           << std::endl;
    }
    else
    {
        os << " Undefined preconditioning" << std::endl;
    }
    os << " Richardson time-step    :" << precond_factor << std::endl;
    if (wf_dyn == 1)
        os << " Anderson extrapolation scheme for wave functions with beta="
           << betaAnderson << std::endl;
    os << "Mix potentials with beta=" << mix_pot << std::endl;
    if (atoms_dyn_)
    {
        switch (AtomsDynamic())
        {
            case AtomsDynamicType::Quench:
                os << std::endl
                   << std::endl
                   << " Quench the electrons" << std::endl;
                break;
            case AtomsDynamicType::MD:
                os << std::endl << std::endl << " Verlet MD" << std::endl;
                break;
            case AtomsDynamicType::LBFGS:
                os << std::endl
                   << std::endl
                   << " LBFGS geometry optimization" << std::endl;
                break;
            case AtomsDynamicType::FIRE:
                os << std::endl
                   << std::endl
                   << " FIRE geometry optimization" << std::endl;
                break;
            default:
                os << "UNKNOWN MOLECULAR DYNAMICS METHOD" << std::endl;
        }
    }

    if (!(AtomsDynamic() == AtomsDynamicType::Quench))
    {
        os << " Timestep for molecular dynamics = " << dt << std::endl;
        if (dt <= 0.) os << " Warning: time step <= 0. !!!" << std::endl;
        os << " Max. # of SC it. per MD step = " << max_electronic_steps
           << std::endl;
    }

    printThermostatInfo(os);

    if (isSpreadFunctionalActive())
    {
        os << " Includes spread penalty functional" << std::endl;
        if (spread_penalty_type_ != 2)
            os << std::setprecision(2) << std::scientific
               << " Penalty damping factor: " << spread_penalty_damping_
               << std::endl;
        os << " Target spread: " << spread_penalty_target_ << std::endl;
        os << " Spread Penalty factor: " << spread_penalty_alpha_ << std::endl;
    }
}

void Control::sync(void)
{
    if (onpe0 && verbose > 0)
        (*MPIdata::sout) << "Control::sync()" << std::endl;
    // pack
    const short size_short_buffer = 96;
    short* short_buffer           = new short[size_short_buffer];
    if (mype_ == 0)
    {
        short_buffer[0]  = wf_dyn;
        short_buffer[1]  = wf_m;
        short_buffer[2]  = multipole_order;
        short_buffer[3]  = wf_extrapolation_;
        short_buffer[4]  = diel;
        short_buffer[5]  = lap_type;
        short_buffer[6]  = precond_type_;
        short_buffer[7]  = orthof;
        short_buffer[8]  = it_algo_type_;
        short_buffer[9]  = num_species;
        short_buffer[10] = mg_levels_;
        short_buffer[11] = project_out_psd;
        short_buffer[12] = xctype;
        short_buffer[13] = steps;
        short_buffer[14] = checkpoint;
        short_buffer[15] = verbose;
        short_buffer[16] = iprint_residual;
        short_buffer[17] = wannier_transform_type;
        short_buffer[18] = vh_init;
        short_buffer[19] = vh_its;
        short_buffer[20] = max_changes_pot;
        short_buffer[21] = tmatrices;
        short_buffer[22] = init_loc;
        short_buffer[23] = init_type;
        short_buffer[24] = atoms_dyn_;
        short_buffer[25] = max_electronic_steps;
        short_buffer[26] = num_MD_steps;
        short_buffer[27] = lr_updates_type;
        short_buffer[28] = lr_update;
        short_buffer[29] = lr_volume_calc;
        short_buffer[30] = orbital_type_;
        short_buffer[31] = line_min;
        short_buffer[32] = thermostat_type;
        short_buffer[33] = bcWF[0];
        short_buffer[34] = bcWF[1];
        short_buffer[35] = bcWF[2];
        short_buffer[36] = bcPoisson[0];
        short_buffer[37] = bcPoisson[1];
        short_buffer[38] = bcPoisson[2];
        short_buffer[39] = short_sighted;
        short_buffer[40] = (short)loc_mode_;
        short_buffer[41] = restart_info;
        short_buffer[42] = restart_file_type;
        short_buffer[43] = out_restart_info;
        short_buffer[44] = out_restart_file_type;
        short_buffer[45] = (short)precond_factor_computed;
        short_buffer[46] = dot_product_type;
        short_buffer[47] = out_restart_file_naming_strategy;
        short_buffer[48] = enforceVmass0;
        short_buffer[49] = dm_inner_steps;
        short_buffer[50] = rmatrices;
        short_buffer[51] = fgmres_kim;
        short_buffer[52] = fgmres_maxits;
        short_buffer[53] = ilu_type;
        short_buffer[54] = ilu_lof;
        short_buffer[55] = ilu_maxfil;
        short_buffer[56] = coloring_algo_;
        short_buffer[57] = diel_flag_;
        short_buffer[58] = poisson_pc_nu1;
        short_buffer[59] = poisson_pc_nu2;
        short_buffer[60] = poisson_pc_nlev;
        short_buffer[61] = system_charge_;
        short_buffer[62] = md_print_freq;
        short_buffer[63] = use_kernel_functions;
        short_buffer[64] = ngpts_[0];
        short_buffer[65] = ngpts_[1];
        short_buffer[66] = ngpts_[2];
        short_buffer[67] = computeCondGram_;
        short_buffer[68] = lrs_extrapolation;
        short_buffer[69] = (short)parallel_transport;
        short_buffer[70] = (short)with_spin_;
        short_buffer[71] = conv_criterion_;
        short_buffer[72] = load_balancing_max_iterations;
        short_buffer[73] = load_balancing_modulo;
        short_buffer[74] = write_clusters;
        short_buffer[75] = DM_solver_;
        short_buffer[80] = dm_algo_;
        short_buffer[81] = dm_approx_order;
        short_buffer[82] = dm_approx_ndigits;
        short_buffer[83] = dm_approx_power_maxits;
        short_buffer[84] = spread_penalty_type_;
        short_buffer[85] = dm_use_old_;
        short_buffer[86] = max_electronic_steps_tight_;
        short_buffer[88] = hartree_reset_;
        short_buffer[89] = MD_last_step_;
        short_buffer[90] = (short)static_cast<int>(poisson_lap_type_);
        short_buffer[91] = poisson_pc_data_;
        short_buffer[92] = precond_precision_;
        short_buffer[93] = mg_npresmoothing_;
        short_buffer[94] = mg_npostsmoothing_;
        short_buffer[95] = compute_dipole_;
    }
    else
    {
        memset(&short_buffer[0], 0, size_short_buffer * sizeof(short));
    }
    const short size_int_buffer = 4;
    int* int_buffer             = new int[size_int_buffer];
    if (mype_ == 0)
    {
        int_buffer[0] = numst;
        int_buffer[1] = nel_;
        int_buffer[2] = nempty_;
        int_buffer[3] = num_ions;
    }
    else
    {
        memset(&int_buffer[0], 0, size_int_buffer * sizeof(int));
    }

    const short size_float_buffer = 48;
    float* float_buffer           = new float[size_float_buffer];
    if (mype_ == 0)
    {
        float_buffer[0]  = betaAnderson;
        float_buffer[1]  = spread_penalty_target_;
        float_buffer[2]  = precond_factor;
        float_buffer[3]  = conv_tol;
        float_buffer[4]  = tol_forces;
        float_buffer[5]  = mix_pot;
        float_buffer[6]  = dm_mix;
        float_buffer[7]  = cut_radius;
        float_buffer[9]  = occ_width_;
        float_buffer[10] = init_rc;
        float_buffer[11] = dt;
        float_buffer[12] = tol_orb_centers_move;
        float_buffer[13] = tkel;
        float_buffer[14] = thtime;
        float_buffer[15] = spread_penalty_damping_;
        float_buffer[16] = rho0_;
        float_buffer[17] = drho0_;
        float_buffer[18] = min_distance_centers_;
        float_buffer[19] = conv_tol_stop;
        float_buffer[20] = threshold_eigenvalue_gram_;
        float_buffer[21] = fgmres_tol;
        float_buffer[22] = ilu_droptol;
        float_buffer[23] = spread_factor;
        float_buffer[24] = ox_;
        float_buffer[25] = oy_;
        float_buffer[26] = oz_;
        float_buffer[27] = lx_;
        float_buffer[28] = ly_;
        float_buffer[29] = lz_;
        float_buffer[30] = total_spin_;
        float_buffer[31] = maxDistanceAtomicInfo_;
        float_buffer[32] = thwidth;
        float_buffer[33] = aomm_radius_;
        float_buffer[34] = aomm_threshold_factor_;
        float_buffer[35] = (rescale_v_ - 1.);
        float_buffer[36] = load_balancing_alpha;
        float_buffer[37] = load_balancing_damping_tol;
        float_buffer[38] = spread_penalty_alpha_;
        float_buffer[39] = overallocate_factor_;
        float_buffer[40] = threshold_eigenvalue_gram_quench_;
        float_buffer[41] = pair_mlwf_distance_threshold_;
        float_buffer[42] = e0_;
        float_buffer[43] = dm_tol;
        float_buffer[44] = poisson_conv_tol;
        float_buffer[45] = ex_;
        float_buffer[46] = ey_;
        float_buffer[47] = ez_;
    }
    else
    {
        memset(&float_buffer[0], 0, size_float_buffer * sizeof(float));
    }

    auto bcast_check = [](int mpirc)
    {
        if (mpirc != MPI_SUCCESS)
        {
            (*MPIdata::sout) << "MPI Bcast of Control failed!!!" << std::endl;
            MPI_Abort(comm_global_, EXIT_FAILURE);
        }
    };

    int mpirc;
    mpirc = MPI_Bcast(
        &short_buffer[0], size_short_buffer, MPI_SHORT, 0, comm_global_);
    bcast_check(mpirc);
    mpirc
        = MPI_Bcast(&int_buffer[0], size_int_buffer, MPI_INT, 0, comm_global_);
    bcast_check(mpirc);
    mpirc = MPI_Bcast(
        &float_buffer[0], size_float_buffer, MPI_FLOAT, 0, comm_global_);
    bcast_check(mpirc);

    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    mmpi.bcast(restart_file, comm_global_);
    mmpi.bcast(out_restart_file, comm_global_);
    mmpi.bcast(md_print_filename, comm_global_);

    short npot = pot_filenames_.size();
    mpirc      = MPI_Bcast(&npot, 1, MPI_SHORT, 0, comm_global_);
    bcast_check(mpirc);
    if (mype_ > 0)
    {
        pot_filenames_.resize(npot);
        pseudopot_flags_.resize(npot);
    }
    for (short i = 0; i < npot; i++)
    {
        short size_str = (short)pot_filenames_[i].size();
        mpirc          = MPI_Bcast(&size_str, 1, MPI_SHORT, 0, comm_global_);
        bcast_check(mpirc);

        char* buffer = new char[size_str + 1];
        if (mype_ == 0)
        {
            pot_filenames_[i].copy(buffer, std::string::npos);
            buffer[pot_filenames_[i].length()] = 0;
        }
        mpirc = MPI_Bcast(buffer, size_str + 1, MPI_CHAR, 0, comm_global_);
        bcast_check(mpirc);
        pot_filenames_[i].assign(&buffer[0], size_str);

        delete[] buffer;

        mpirc = MPI_Bcast(&pseudopot_flags_[i], 1, MPI_SHORT, 0, comm_global_);
        bcast_check(mpirc);
    }

    // unpack
    wf_dyn                           = short_buffer[0];
    wf_m                             = short_buffer[1];
    multipole_order                  = short_buffer[2];
    wf_extrapolation_                = short_buffer[3];
    diel                             = short_buffer[4];
    lap_type                         = short_buffer[5];
    precond_type_                    = short_buffer[6];
    orthof                           = short_buffer[7];
    it_algo_type_                    = short_buffer[8];
    num_species                      = short_buffer[9];
    mg_levels_                       = short_buffer[10];
    project_out_psd                  = short_buffer[11];
    xctype                           = short_buffer[12];
    steps                            = short_buffer[13];
    checkpoint                       = short_buffer[14];
    verbose                          = short_buffer[15];
    iprint_residual                  = short_buffer[16];
    wannier_transform_type           = short_buffer[17];
    vh_init                          = short_buffer[18];
    vh_its                           = short_buffer[19];
    max_changes_pot                  = short_buffer[20];
    tmatrices                        = short_buffer[21];
    init_loc                         = short_buffer[22];
    init_type                        = short_buffer[23];
    atoms_dyn_                       = short_buffer[24];
    max_electronic_steps             = short_buffer[25];
    num_MD_steps                     = short_buffer[26];
    lr_updates_type                  = short_buffer[27];
    lr_update                        = short_buffer[28];
    lr_volume_calc                   = short_buffer[29];
    orbital_type_                    = short_buffer[30];
    line_min                         = short_buffer[31];
    thermostat_type                  = short_buffer[32];
    bcWF[0]                          = short_buffer[33];
    bcWF[1]                          = short_buffer[34];
    bcWF[2]                          = short_buffer[35];
    bcPoisson[0]                     = short_buffer[36];
    bcPoisson[1]                     = short_buffer[37];
    bcPoisson[2]                     = short_buffer[38];
    short_sighted                    = short_buffer[39];
    loc_mode_                        = (bool)short_buffer[40];
    restart_info                     = short_buffer[41];
    restart_file_type                = short_buffer[42];
    out_restart_info                 = short_buffer[43];
    out_restart_file_type            = short_buffer[44];
    precond_factor_computed          = (bool)short_buffer[45];
    dot_product_type                 = short_buffer[46];
    out_restart_file_naming_strategy = short_buffer[47];
    enforceVmass0                    = short_buffer[48];
    dm_inner_steps                   = short_buffer[49];
    rmatrices                        = short_buffer[50];
    fgmres_kim                       = short_buffer[51];
    fgmres_maxits                    = short_buffer[52];
    ilu_type                         = short_buffer[53];
    ilu_lof                          = short_buffer[54];
    ilu_maxfil                       = short_buffer[55];
    coloring_algo_                   = short_buffer[56];
    diel_flag_                       = short_buffer[57];
    poisson_pc_nu1                   = short_buffer[58];
    poisson_pc_nu2                   = short_buffer[59];
    poisson_pc_nlev                  = short_buffer[60];
    system_charge_                   = short_buffer[61];
    md_print_freq                    = short_buffer[62];
    use_kernel_functions             = short_buffer[63];
    ngpts_[0]                        = short_buffer[64];
    ngpts_[1]                        = short_buffer[65];
    ngpts_[2]                        = short_buffer[66];
    computeCondGram_                 = short_buffer[67];
    lrs_extrapolation                = short_buffer[68];
    parallel_transport               = (bool)short_buffer[69];
    with_spin_                       = (bool)short_buffer[70];
    conv_criterion_                  = short_buffer[71];
    load_balancing_max_iterations    = short_buffer[72];
    load_balancing_modulo            = short_buffer[73];
    write_clusters                   = short_buffer[74];
    DM_solver_                       = short_buffer[75];
    dm_algo_                         = short_buffer[80];
    dm_approx_order                  = short_buffer[81];
    dm_approx_ndigits                = short_buffer[82];
    dm_approx_power_maxits           = short_buffer[83];
    spread_penalty_type_             = short_buffer[84];
    dm_use_old_                      = short_buffer[85];
    max_electronic_steps_tight_      = short_buffer[86];
    hartree_reset_                   = short_buffer[88];
    MD_last_step_                    = short_buffer[89];
    poisson_lap_type_  = static_cast<PoissonFDtype>(short_buffer[90]);
    poisson_pc_data_   = short_buffer[91];
    precond_precision_ = short_buffer[92];
    mg_npresmoothing_  = short_buffer[93];
    mg_npostsmoothing_ = short_buffer[94];
    compute_dipole_    = short_buffer[95];

    numst    = int_buffer[0];
    nel_     = int_buffer[1];
    nempty_  = int_buffer[2];
    num_ions = int_buffer[3];

    betaAnderson                      = float_buffer[0];
    spread_penalty_target_            = float_buffer[1];
    precond_factor                    = float_buffer[2];
    conv_tol                          = float_buffer[3];
    tol_forces                        = float_buffer[4];
    mix_pot                           = float_buffer[5];
    dm_mix                            = float_buffer[6];
    cut_radius                        = float_buffer[7];
    occ_width_                        = float_buffer[9];
    init_rc                           = float_buffer[10];
    dt                                = float_buffer[11];
    tol_orb_centers_move              = float_buffer[12];
    tkel                              = float_buffer[13];
    thtime                            = float_buffer[14];
    spread_penalty_damping_           = float_buffer[15];
    rho0_                             = float_buffer[16];
    drho0_                            = float_buffer[17];
    min_distance_centers_             = float_buffer[18];
    conv_tol_stop                     = float_buffer[19];
    threshold_eigenvalue_gram_        = float_buffer[20];
    fgmres_tol                        = float_buffer[21];
    ilu_droptol                       = float_buffer[22];
    spread_factor                     = float_buffer[23];
    ox_                               = float_buffer[24];
    oy_                               = float_buffer[25];
    oz_                               = float_buffer[26];
    lx_                               = float_buffer[27];
    ly_                               = float_buffer[28];
    lz_                               = float_buffer[29];
    total_spin_                       = float_buffer[30];
    maxDistanceAtomicInfo_            = float_buffer[31];
    thwidth                           = float_buffer[32];
    aomm_radius_                      = float_buffer[33];
    aomm_threshold_factor_            = float_buffer[34];
    rescale_v_                        = 1. + (double)(float_buffer[35]);
    load_balancing_alpha              = float_buffer[36];
    load_balancing_damping_tol        = float_buffer[37];
    spread_penalty_alpha_             = float_buffer[38];
    overallocate_factor_              = float_buffer[39];
    threshold_eigenvalue_gram_quench_ = float_buffer[40];
    pair_mlwf_distance_threshold_     = float_buffer[41];
    e0_                               = float_buffer[42];
    dm_tol                            = float_buffer[43];
    poisson_conv_tol                  = float_buffer[44];
    ex_                               = float_buffer[45];
    ey_                               = float_buffer[46];
    ez_                               = float_buffer[47];
    max_electronic_steps_loose_       = max_electronic_steps;

    delete[] short_buffer;
    delete[] int_buffer;
    delete[] float_buffer;

    unsigned short nocc = orbitals_occupations_.size();
    mpirc = MPI_Bcast(&nocc, 1, MPI_UNSIGNED_SHORT, 0, comm_global_);
    orbitals_occupations_.resize(nocc);
    if (!orbitals_occupations_.empty())
        mpirc = MPI_Bcast(orbitals_occupations_.data(),
            orbitals_occupations_.size(), MPI_FLOAT, 0, comm_global_);
}

// function to set default values when boost interface not used
void Control::setDefaultValues()
{
    assert(max_electronic_steps > -1);

    overallocate_factor_        = 1.2;
    max_electronic_steps_loose_ = max_electronic_steps;
    max_electronic_steps_tight_ = max_electronic_steps;
}

void Control::adjust()
{
    // change dm_mix default to 1. if not using Davidson, MVP or HMVP
    if ((it_algo_type_ != 2 && DM_solver_ != 1 && DM_solver_ != 2)
        && dm_mix < 0.)
        dm_mix = 1.;

    if (nel_ - 2 * numst == 0)
    {
        dm_mix = 1.;
    }
    if (getOrthoType() == OrthoType::Eigenfunctions)
    {
        orthof = 0;
        dm_mix = 1.;
        wf_dyn = 0;
    }
    if (getOrthoType() == OrthoType::Orthonormal)
    {
        orthof = 0;
    }
    if (!loc_mode_) lr_update = 0;
    if (loc_mode_ && lr_update)
        wannier_transform_type = std::max(wannier_transform_type, (short)1);
    restart_run = (restart_info > 0) ? true : false;
}

int Control::checkState()
{
    for (short i = 0; i < 3; i++)
        if ((bcPoisson[i] != 0) && (bcPoisson[i] != 1) && (bcPoisson[i] != 2))
        {
            (*MPIdata::sout)
                << "Control::checkState() -> invalid boundary conditions"
                << std::endl;
            return -1;
        }

    for (short i = 0; i < 3; i++)
        if ((bcWF[i] != 0) && (bcWF[i] != 1))
        {
            (*MPIdata::sout)
                << "Control::checkState() -> invalid WF boundary conditions"
                << std::endl;
            return -1;
        }

    if (diel != 0 && diel != 1)
    {
        (*MPIdata::sout) << "Flag diel should be 0 or 1" << std::endl;
        return -1;
    }
    if (vh_init < 0 || vh_init > 100)
    {
        (*MPIdata::sout) << "Invalid parameter vh_init" << std::endl;
        return -1;
    }
    if (vh_its < 0 || vh_its > 100)
    {
        (*MPIdata::sout) << "Invalid parameter vh_its" << std::endl;
        return -1;
    }
    if (rho0_ > .1)
    {
        (*MPIdata::sout) << "rho0=" << rho0_ << " is too large" << std::endl;
        return -1;
    }
    if (short_sighted != 0 && short_sighted != 1)
    {
        (*MPIdata::sout) << "Control::checkState() -> Short-sighted option "
                            "should be 0 (off) or 1 (on)"
                         << std::endl;
        return -1;
    }
    if (AtomsDynamic() == AtomsDynamicType::UNDEFINED)
    {
        (*MPIdata::sout)
            << "Control::checkState() -> invalid parameter for md method"
            << std::endl;
        return -1;
    }
    if (thermostat_type != 0 && thermostat_type != 1 && thermostat_type != 2
        && thermostat_type != 3)
    {
        (*MPIdata::sout) << "Control::checkState() -> Invalid thermostat option"
                         << std::endl;
        return -1;
    }
    if (wf_dyn != 0 && wf_dyn != 1)
    {
        (*MPIdata::sout) << "Control::checkState() -> Invalid quench method"
                         << std::endl;
        return -1;
    }
    if (AtomsDynamic() == AtomsDynamicType::MD)
        if (!(WFExtrapolation() == WFExtrapolationType::Reversible
                || WFExtrapolation() == WFExtrapolationType::Order2
                || WFExtrapolation() == WFExtrapolationType::Order3))
        {
            (*MPIdata::sout) << "Control::checkState() -> Invalid option for "
                                "WF extrapolation in MD!!!"
                             << std::endl;
            return -1;
        }
    if (init_loc != 0 && init_loc != 1)
    {
        (*MPIdata::sout)
            << "Control::checkState() -> Invalid orbital initialization flag"
            << std::endl;
        return -1;
    }
    if (init_type != 0 && init_type != 1 && init_type != 2)
    {
        (*MPIdata::sout)
            << "Control::checkState() -> Invalid orbital initialization shape"
            << std::endl;
        return -1;
    }
    if (numst < 0)
    {
        (*MPIdata::sout) << "Control::checkState() -> Invalid number of states"
                         << std::endl;
        return -1;
    }
    assert(xctype == 0 || xctype == 2);
    assert(mix_pot < 2. && mix_pot > 0.);
    assert(precond_type_ == 10);
    assert(project_out_psd == 0 || project_out_psd == 1);
    assert(wannier_transform_type == 0 || wannier_transform_type == 1
           || wannier_transform_type == 2);
    assert(tmatrices == 1 || tmatrices == 0);
    assert(rmatrices == 1 || rmatrices == 0);
    assert(mg_levels_ >= -1);
    assert(rho0_ > 0.);
    assert(drho0_ > 0.);
    assert(e0_ > 0.);
    assert(getOrthoType() != OrthoType::UNDEFINED);
    assert(num_MD_steps >= 0);
    assert(dt >= 0.);
    if (short_sighted > 0)
    {
        assert(fgmres_kim >= 0);
        assert(fgmres_maxits >= 0);
        assert(fgmres_tol > 0.);
        assert(ilu_droptol > 0.);
        assert(ilu_lof >= 0);
        assert(ilu_maxfil >= 0);
    }
    if (!(lap_type == 0 || lap_type == 1 || lap_type == 2 || lap_type == 3
            || lap_type == 4 || lap_type == 10))
    {
        (*MPIdata::sout) << "Control::checkState() -> Invalid Laplacian type"
                         << std::endl;
        return -1;
    }
    if (lr_update > 0 && lr_volume_calc > 0 && wannier_transform_type != 2)
    {
        (*MPIdata::sout) << "Control::lr_update=" << lr_update << std::endl;
        (*MPIdata::sout) << "Control::lr_volume_calc =" << lr_volume_calc
                         << std::endl;
        (*MPIdata::sout) << "Control::checkState(): NOLMO centers required for "
                            "LR adaptation!"
                         << std::endl;
        return -1;
    }

    int ret = checkOptions();

    return ret;
}

// set number of orbitals to compute
// given spin and number of valence electrons
// each spin 1/2 has a charge 1 e-
void Control::setNumst(const short myspin, const int neval)
{
    assert(nempty_ >= 0);
    assert(neval > 0);

    nel_ = neval - system_charge_;

    if (with_spin_) // 1 electrons/orbital
    {
        if (mype_ == 0)
            std::cout << "spin=" << total_spin_ << ", nel=" << nel_
                      << std::endl;
        assert((nel_ - (int)(2. * total_spin_)) % 2 == 0);

        numst = (nel_ - (int)(2. * total_spin_)) / 2;
        if (myspin == 0) numst += (int)(2. * total_spin_);

        numst += nempty_;
        nelspin_ = static_cast<double>(numst - nempty_);
    }
    else // no spin, 2 electrons/orbital
    {
        // only set numst if not set yet
        if (numst == -1)
        {
            numst = nel_ / 2;
            if (occupationWidthIsZero())
            {
                assert(2 * numst == nel_);
            }
            else
            {
                if (2 * numst < nel_) numst++;
            }

            numst += nempty_;
            nelspin_ = 0.5 * static_cast<double>(nel_);
        }
    }

    if (mype_ == 0)
        std::cout << "spin=" << total_spin_ << ", nel=" << neval
                  << ", nelspin_ =" << nelspin_ << ", numst=" << numst
                  << ", nempty=" << nempty_ << std::endl;
}

void Control::setTolEnergy()
{
    // if rtol has been set, use it to define conv_tol
    if (conv_rtol_ > 0.)
    {
        conv_tol = conv_rtol_ * nel_;
        if (mype_ == 0)
            (*MPIdata::sout)
                << "Tolerance on energy based on relative tolerance"
                << std::endl;
    }
    if (conv_tol_stop >= 999.) conv_tol_stop = 100. * conv_tol;
    MPI_Bcast(&conv_tol, 1, MPI_FLOAT, 0, comm_global_);
    MPI_Bcast(&conv_tol_stop, 1, MPI_FLOAT, 0, comm_global_);
    if (mype_ == 0)
        (*MPIdata::sout) << "Tolerance on energy set to " << conv_tol
                         << std::endl;

    assert(conv_tol > 0.);
}

void Control::readRestartInfo(std::ifstream* tfile)
{
    std::string zero = "0";
    if (tfile != nullptr)
    {
        // Read in the restart file names
        std::string filename;
        (*tfile) >> filename;

        restart_file.assign(run_directory_);
        restart_file.append("/");
        restart_file.append(filename);

        if (zero.compare(filename) == 0)
            restart_info = 0;
        else
        {
            (*tfile) >> restart_info;
            (*tfile) >> restart_file_type;
        }
        (*MPIdata::sout) << "Input restart file: " << restart_file << std::endl;
    }
    else
    {
        restart_info = 0;
    }

    printRestartLink();

    MPI_Bcast(&restart_info, 1, MPI_SHORT, 0, comm_global_);
    MPI_Bcast(&restart_file_type, 1, MPI_SHORT, 0, comm_global_);
    char buffer[64];
    if (mype_ == 0)
    {
        memcpy(buffer, restart_file.c_str(),
            (restart_file.size() + 1) * sizeof(char));
    }
    int mpirc = MPI_Bcast(buffer, 64, MPI_CHAR, 0, comm_global_);
    if (mpirc != MPI_SUCCESS)
    {
        (*MPIdata::sout)
            << "Control::readRestartInfo(): MPI Bcast of restart_file failed!!!"
            << std::endl;
    }
}

int Control::setPreconditionerParameters(const short type, const float factor,
    const bool project_out, const short nlevels, const float fgrid_hmax)
{
    if (type != 10)
    {
        (*MPIdata::sout) << type << ": Invalid preconditioner type"
                         << std::endl;
        return -1;
    }

    precond_type_ = type;
    if (precond_type_ == 10)
        (*MPIdata::sout) << "MG preconditioner" << std::endl;
    (*MPIdata::sout) << "Preconditioner: block implementation" << std::endl;

    precond_factor = factor;
    if (factor < 0.)
        precond_factor_computed = true;
    else
        precond_factor_computed = false;
    project_out_psd = (short)project_out;
    mg_levels_      = nlevels;

    // Define number of coarse levels for orbitals preconditioning
    short maxlevels = -1;
    for (short level = 0; level < 3; level++)
    {
        float hmaxgrid = fgrid_hmax * (1 << level);
        assert(hmaxgrid > 0);
        if (hmaxgrid < 1.4) maxlevels += 1;
    }
    if (mg_levels_ > maxlevels)
    {
        (*MPIdata::sout) << "Set MG levels number to maxlevels=" << maxlevels
                         << std::endl;
        mg_levels_ = maxlevels;
    }
    (*MPIdata::sout) << "MG preconditioner: number of levels =" << mg_levels_
                     << std::endl;
    return 0;
}

int Control::setShortSightedSolverParameters(const float fact, const float stol,
    const float dtol, const short kim, const short itmax, const short lfil,
    const short maxfill, const short ilutype)
{
    if (lfil < 0)
    {
        (*MPIdata::sout)
            << "Invalid fill level parameter for ILU preconditioner: lof = "
            << lfil << std::endl;
        return -1;
    }
    if (itmax < 0)
    {
        (*MPIdata::sout)
            << "Invalid maxits parameter for fgmres solver: maxits = " << itmax
            << std::endl;
        return -1;
    }
    if (kim < 0)
    {
        (*MPIdata::sout)
            << "Invalid Krylov dimension parameter for fgmres solver: kim = "
            << kim << std::endl;
        return -1;
    }

    spread_factor = fact;
    fgmres_tol    = stol;
    ilu_droptol   = dtol;
    fgmres_kim    = kim;
    fgmres_maxits = itmax;
    ilu_lof       = lfil;
    ilu_maxfil    = maxfill;
    ilu_type      = ilutype;

    return 0;
}

int Control::readThermostatInfo(std::ifstream* tfile)
{
    (*tfile) >> thermostat_type;
    if (thermostat_type != 0 && thermostat_type != 1 && thermostat_type != 2
        && thermostat_type != 3)
    {
        (*MPIdata::sout) << "Invalid thermostat option" << std::endl;
        return -1;
    }
    if (thermostat_type > 0)
    {
        (*tfile) >> tkel >> thtime;
        if (thtime <= 0.)
        {
            (*MPIdata::sout)
                << "Invalid thermostat option: time should be >0." << std::endl;
            return -1;
        }
    }
    return 0;
}

void Control::printThermostatInfo(std::ostream& os) const
{
    if (thermostat_type > 0 && mype_ == 0)
    {
        os << "Thermostat type: ";
        if (thermostat_type == 1)
            os << "Berendsen" << std::endl;
        else if (thermostat_type == 2)
            os << "Langevin" << std::endl;
        else if (thermostat_type == 3)
            os << "SCALING" << std::endl;
        else
            os << std::endl;
        os << "Thermostat target T: " << tkel << std::endl;
        os << "Thermostat trelaxation time: " << thtime << std::endl;
        os << "Thermostat trelaxation width: " << thwidth << std::endl;
    }
}

void Control::setLocMode(const float radius, const float lx, const float ly,
    const float lz, const float mind_centers)
{
    cut_radius            = radius;
    min_distance_centers_ = mind_centers;
    loc_mode_ = (cut_radius < lx || cut_radius < ly || cut_radius < lz);
    if (getOrthoType() != OrthoType::Nonorthogonal)
    {
        cut_radius = 1000.;
        loc_mode_  = false;
    }

    if (loc_mode_)
        init_loc = 1;
    else if (restart_info > 2)
        init_loc = 0;

    if (!loc_mode_)
    {
        lr_update             = 0;
        min_distance_centers_ = 0.;
    }
    if (loc_mode_) project_out_psd = 0;
#ifdef DEBUG
    if (mype_ == 0)
        (*MPIdata::sout) << " Localization radius=" << cut_radius << std::endl;
#endif
}

// set radius for short-sighted matrices based on
// orbitals localization radius, that is how many elements
// of Gram matrix are used to compute inverse
void Control::setSpreadRadius()
{
    assert(spread_factor > 0.);
    assert(cut_radius > 0.);

    spread_radius = spread_factor * cut_radius;
}

void Control::setNumIons(const int nions) { num_ions = nions; }

void Control::setTolEigenvalueGram(const float tol)
{
    threshold_eigenvalue_gram_ = tol;
    if (onpe0)
        (*MPIdata::sout) << "Tol. Eigenvalue Gram="
                         << threshold_eigenvalue_gram_ << std::endl;
}

void Control::global_exit() { MPI_Abort(comm_global_, EXIT_FAILURE); }

void Control::setSpecies(Potentials& pot)
{
    assert(sp_.empty());

    int i = 0;

    for (std::vector<std::string>::iterator it = pot_filenames_.begin();
         it != pot_filenames_.end(); ++it)
    {

        if (pot.pot_type(i) == 'n' || pot.pot_type(i) == 's'
            || pot.pot_type(i) == 'f')
        {
            // if(onpe0)(*MPIdata::sout)<<" pot_type="<<pot.pot_type(i)<<endl;
            Species s(comm_global_);
            sp_.push_back(s);
        }

        i++;
    }

    // Read in pseudopotentials
    pot.readAll(sp_);
}

void Control::readPotFilenames(std::ifstream* tfile)
{
    assert(pot_filenames_.empty());

    for (int i = 0; i < num_species; i++)
    {
        std::string pot_filename;
        short flag = -1;
        if (mype_ == 0)
        {
            (*tfile) >> pot_filename >> flag;
        }

        pot_filenames_.push_back(pot_filename);
        pseudopot_flags_.push_back(flag);

        if (mype_ == 0)
        {
            read_comments(*tfile);
            if (onpe0)
                (*MPIdata::sout) << "Potential file name: " << pot_filenames_[i]
                                 << std::endl;
        }
    }
}

void Control::registerPotentials(Potentials& pot)
{
    assert(pot_filenames_.size() == pseudopot_flags_.size());

    short i = 0;
    for (std::vector<std::string>::iterator it = pot_filenames_.begin();
         it != pot_filenames_.end(); ++it)
    {
        // just make sure every process does the same thing...
        MPI_Barrier(comm_global_);

        pot.registerName(*it, pseudopot_flags_[i]);
        i++;
    }
}

int Control::checkNLrange()
{
    for (std::vector<Species>::const_iterator it = sp_.begin(); it != sp_.end();
         ++it)
    {
        for (short i = 0; i < 3; i++)
            if (it->dim_nl() > static_cast<int>(ngpts_[i]))
            {
                std::cerr
                    << "WARNING: Size of cell not large enough for Species "
                    << it->name() << " in direction " << i << std::endl;
                std::cerr << " dim nl=" << it->dim_nl()
                          << " larger than n=" << ngpts_[i] << std::endl;
                return -1;
            }
    }

    return 0;
}

// set internal flags from read boost options
void Control::setOptions(const boost::program_options::variables_map& vm)
{
    printWithTimeStamp("Control::setOptions()...", std::cout);

    if (onpe0)
    {
        assert(vm.count("Domain.lx"));
        assert(vm.count("Poisson.bcx"));
        assert(vm.count("xcFunctional"));
        assert(vm.count("Orbitals.bcx"));

        std::string str;

        verbose = vm["verbosity"].as<short>();

        str = vm["xcFunctional"].as<std::string>();
        if (str.compare("LDA") == 0) xctype = 0;
        if (str.compare("PBE") == 0) xctype = 2;

        str = vm["FDtype"].as<std::string>();
        if (str.compare("Mehrstellen") == 0) lap_type = 0;
        if (str.compare("2nd") == 0) lap_type = 1;
        if (str.compare("4th") == 0) lap_type = 2;
        if (str.compare("6th") == 0) lap_type = 3;
        if (str.compare("8th") == 0) lap_type = 4;
        if (str.compare("Mehrstellen2") == 0) lap_type = 10;
        assert(lap_type >= 0);

        system_charge_ = vm["charge"].as<short>();

        lx_ = vm["Domain.lx"].as<float>();
        ly_ = vm["Domain.ly"].as<float>();
        lz_ = vm["Domain.lz"].as<float>();
        ox_ = vm["Domain.ox"].as<float>();
        oy_ = vm["Domain.oy"].as<float>();
        oz_ = vm["Domain.oz"].as<float>();

        ngpts_[0] = vm["Mesh.nx"].as<short>();
        ngpts_[1] = vm["Mesh.ny"].as<short>();
        ngpts_[2] = vm["Mesh.nz"].as<short>();

        if (vm.count("Potentials.pseudopotential"))
        {
            short filter_flag = vm["Potentials.filterPseudo"].as<char>();

            pot_filenames_ = vm["Potentials.pseudopotential"]
                                 .as<std::vector<std::string>>();
            for (unsigned short i = 0; i < pot_filenames_.size(); i++)
                pseudopot_flags_.push_back(filter_flag);
        }

        if (vm.count("Potentials.external"))
        {
            char bin_flag = vm["Potentials.binExternal"].as<bool>() ? 'a' : 'b';
            std::vector<std::string> pot_filenames
                = vm["Potentials.external"].as<std::vector<std::string>>();

            for (unsigned short i = 0; i < pot_filenames.size(); i++)
            {
                pot_filenames_.push_back(pot_filenames[i]);
                pseudopot_flags_.push_back(bin_flag);
            }
        }

        restart_file = vm["Restart.input_filename"].as<std::string>();
        if (restart_file.compare("") == 0)
        {
            restart_info = 0;
        }
        else
        {
            printRestartLink();
            restart_info = vm["Restart.input_level"].as<short>();
        }
        str = vm["Restart.input_type"].as<std::string>();
        if (str.compare("distributed") == 0)
            restart_file_type = 0;
        else
            restart_file_type = 1;

        std::string filename(vm["Restart.output_filename"].as<std::string>());
        std::string autoname("auto");
        if (autoname.compare(filename) == 0) // automatic naming of dump
        {
            filename                         = "snapshot";
            out_restart_file_naming_strategy = 1;
        }
        out_restart_file.assign(run_directory_);
        out_restart_file.append("/");
        out_restart_file.append(filename);

        out_restart_info = vm["Restart.output_level"].as<short>();
        str              = vm["Restart.output_type"].as<std::string>();
        if (str.compare("distributed") == 0) out_restart_file_type = 0;
        if (str.compare("single_file") == 0) out_restart_file_type = 1;
        if (out_restart_file_type < 0)
        {
            (*MPIdata::serr)
                << "ERROR in Control::setOptions: Invalid restart dump type"
                << std::endl;
            MPI_Abort(comm_global_, EXIT_FAILURE);
        }

        (*MPIdata::sout) << "Output restart file: " << out_restart_file
                         << " with info level " << out_restart_info
                         << std::endl;

        checkpoint = vm["Restart.interval"].as<short>();

        rescale_v_ = vm["Restart.rescale_v"].as<double>();

        // Poisson solver
        str = vm["Poisson.FDtype"].as<std::string>();
        if (str.compare("2nd") == 0) poisson_lap_type_ = PoissonFDtype::h2;
        if (str.compare("4th") == 0) poisson_lap_type_ = PoissonFDtype::h4;
        if (str.compare("6th") == 0) poisson_lap_type_ = PoissonFDtype::h6;
        if (str.compare("8th") == 0) poisson_lap_type_ = PoissonFDtype::h8;
        if (str.compare("Mehrstellen") == 0)
            poisson_lap_type_ = PoissonFDtype::h4M;
        if (str.compare("Mehrstellen2") == 0)
            poisson_lap_type_ = PoissonFDtype::h4MP;

        str = vm["Poisson.bcx"].as<std::string>();
        if (str.compare("0") == 0) bcPoisson[0] = 0;
        if (str.compare("periodic") == 0) bcPoisson[0] = 1;
        if (str.compare("charge") == 0) bcPoisson[0] = 2;

        str = vm["Poisson.bcy"].as<std::string>();
        if (str.compare("0") == 0) bcPoisson[1] = 0;
        if (str.compare("periodic") == 0) bcPoisson[1] = 1;
        if (str.compare("charge") == 0) bcPoisson[1] = 2;

        str = vm["Poisson.bcz"].as<std::string>();
        if (str.compare("0") == 0) bcPoisson[2] = 0;
        if (str.compare("periodic") == 0) bcPoisson[2] = 1;
        if (str.compare("charge") == 0) bcPoisson[2] = 2;

        str = vm["Orbitals.bcx"].as<std::string>();
        if (str.compare("0") == 0) bcWF[0] = 0;
        if (str.compare("periodic") == 0) bcWF[0] = 1;

        str = vm["Orbitals.bcy"].as<std::string>();
        if (str.compare("0") == 0) bcWF[1] = 0;
        if (str.compare("periodic") == 0) bcWF[1] = 1;

        str = vm["Orbitals.bcz"].as<std::string>();
        if (str.compare("0") == 0) bcWF[2] = 0;
        if (str.compare("periodic") == 0) bcWF[2] = 1;

        if (vm.count("Orbitals.occupations"))
        {
            orbitals_occupations_
                = get_list_option<float>(vm, "Orbitals.occupations");
        }

        str = vm["Poisson.solver"].as<std::string>();
        if (str.compare("CG") == 0 || str.compare("PCG") == 0) diel_flag_ = 10;
        if (str.compare("MG") == 0) diel_flag_ = 0;

        str = vm["Poisson.diel"].as<std::string>();
        if (str.compare("on") == 0 || str.compare("ON") == 0) diel = 1;
        if (str.compare("off") == 0 || str.compare("OFF") == 0) diel = 0;

        bool poisson_reset = vm["Poisson.reset"].as<bool>();
        hartree_reset_     = poisson_reset ? 1 : 0;

        poisson_pc_nu1   = vm["Poisson.nu1"].as<short>();
        poisson_pc_nu2   = vm["Poisson.nu2"].as<short>();
        vh_init          = vm["Poisson.max_steps_initial"].as<short>();
        vh_its           = vm["Poisson.max_steps"].as<short>();
        poisson_pc_nlev  = vm["Poisson.max_levels"].as<short>();
        rho0_            = vm["Poisson.rho0"].as<float>();
        drho0_           = vm["Poisson.beta"].as<float>();
        e0_              = vm["Poisson.e0"].as<float>();
        poisson_pc_data_ = vm["Poisson.precond_precision"].as<short>();
        poisson_conv_tol = vm["Poisson.conv_tol"].as<float>();

        ex_ = vm["Efield.ex"].as<float>();
        ey_ = vm["Efield.ey"].as<float>();
        ez_ = vm["Efield.ez"].as<float>();

        str = vm["ProjectedMatrices.solver"].as<std::string>();
        if (str.compare("short_sighted") == 0) short_sighted = 1;
        if (str.compare("exact") == 0) short_sighted = 0;

        tmatrices = vm["ProjectedMatrices.printMM"].as<bool>() ? 1 : 0;
        rmatrices = vm["ProjectedMatrices.replicated"].as<bool>() ? 1 : 0;

        if (short_sighted)
        {
            spread_factor = vm["ShortSightedInverse.spread_factor"].as<float>();
            fgmres_tol    = vm["ShortSightedInverse.tol"].as<float>();
            ilu_droptol   = vm["ShortSightedInverse.ilu_drop_tol"].as<float>();
            fgmres_kim    = vm["ShortSightedInverse.krylov_dim"].as<short>();
            fgmres_maxits
                = vm["ShortSightedInverse.max_iterations"].as<short>();
            ilu_lof = vm["ShortSightedInverse.ilu_filling_level"].as<short>();
            ilu_maxfil = vm["ShortSightedInverse.ilut_max_fill"].as<int>();
            str        = vm["ShortSightedInverse.ilu_type"].as<std::string>();
            if (str.compare("ILU") == 0)
                ilu_type = 0;
            else
                ilu_type = 1;
        }
        else
        {
            spread_factor = 1000.;
        }

        str = vm["Quench.solver"].as<std::string>();
        if (str.compare("ABPG") == 0)
        {
            it_algo_type_ = 0;
            wf_dyn        = 1;
            wf_m          = vm["ABPG.m"].as<short>();
            betaAnderson  = vm["ABPG.beta"].as<float>();
        }
        if (str.compare("PSD") == 0)
        {
            it_algo_type_ = 0;
            wf_dyn        = 0;
        }
        if (str.compare("NLCG") == 0)
        {
            it_algo_type_ = 1;
            wf_dyn        = 1;

            parallel_transport
                = vm["NLCG.parallel_transport"].as<bool>() ? 1 : 0;
        }
        if (str.compare("Davidson") == 0)
        {
            it_algo_type_ = 2;
        }
        if (str.compare("PR") == 0) // Polak-Ribiere
        {
            it_algo_type_ = 3;
        }
        std::cout << "Outer solver type: " << str << std::endl;
        assert(it_algo_type_ >= 0);

        // Preconditioner parameters
        mg_levels_         = vm["Preconditioner.num_levels"].as<short>() - 1;
        mg_npresmoothing_  = vm["Preconditioner.npresmoothing"].as<short>();
        mg_npostsmoothing_ = vm["Preconditioner.npostsmoothing"].as<short>();
        precond_precision_ = vm["Preconditioner.precision"].as<short>();
        assert(precond_precision_ == 32 || precond_precision_ == 64);

        precond_factor = vm["Quench.step_length"].as<float>();
        if (precond_factor < 0.)
        {
            switch (lap_type)
            {
                case 0:
                {
                    precond_factor = 2.;
                    break;
                }

                case 2:
                {
                    precond_factor = 1.5;
                    break;
                }

                default:
                    precond_factor = 2.;
            }
        }

        max_changes_pot = vm["Quench.num_lin_iterations"].as<short>();
        conv_tol        = vm["Quench.atol"].as<float>();
        conv_rtol_      = vm["Quench.rtol"].as<float>();

        str = vm["Quench.conv_criterion"].as<std::string>();
        if (str.compare("deltaE") == 0) conv_criterion_ = 0;
        if (str.compare("residual") == 0) conv_criterion_ = 1;
        if (str.compare("maxResidual") == 0) conv_criterion_ = 2;

        computeCondGram_ = vm["Quench.compute_cond_Gram"].as<bool>() ? 2 : 0;
        short ival       = computeCondGram_ > 0 ? 2 : 1;
        computeCondGram_
            = vm["MD.compute_cond_Gram"].as<bool>() ? ival : computeCondGram_;

        orthof          = vm["Quench.ortho_freq"].as<short>();
        iprint_residual = vm["Quench.interval_print_residual"].as<short>();
        conv_tol_stop   = vm["Quench.required_tol"].as<float>();
        threshold_eigenvalue_gram_quench_
            = vm["Quench.min_Gram_eigenvalue"].as<float>();
        pair_mlwf_distance_threshold_
            = vm["Quench.pair_mlwf_distance_threshold"].as<float>();

        spread_penalty_alpha_ = vm["SpreadPenalty.alpha"].as<float>();
        if (spread_penalty_alpha_ > 0.)
        {
            str = vm["SpreadPenalty.type"].as<std::string>();
            if (str.compare("volume") == 0)
            {
                // new interface
                spread_penalty_type_ = 1;
                spread_penalty_damping_
                    = vm["SpreadPenalty.damping"].as<float>();
                spread_penalty_target_ = vm["SpreadPenalty.target"].as<float>();
            }
            else if (str.compare("individual") == 0)
            {
                // new interface
                spread_penalty_type_ = 0;
                spread_penalty_damping_
                    = vm["SpreadPenalty.damping"].as<float>();
                spread_penalty_target_ = vm["SpreadPenalty.target"].as<float>();
            }
            else if (str.compare("energy") == 0)
            {
                // individual penalties with new interface
                spread_penalty_type_   = 2;
                spread_penalty_target_ = vm["SpreadPenalty.target"].as<float>();
            }
            else if (str.compare("XLBOMD") == 0)
            {
                spread_penalty_type_   = 3;
                spread_penalty_target_ = vm["SpreadPenalty.target"].as<float>();
            }
            else
            {
                std::cerr << "ERROR: Spread Penalty needs a type" << std::endl;
                MPI_Abort(comm_global_, EXIT_FAILURE);
            }

            if (spread_penalty_target_ <= 0.)
            {
                (*MPIdata::sout) << "Invalid value for Spread Penalty target: "
                                 << spread_penalty_target_ << std::endl;
                MPI_Abort(comm_global_, EXIT_FAILURE);
            }
        }

        aomm_radius_         = vm["AOMM.kernel_radius"].as<float>();
        use_kernel_functions = (aomm_radius_ > 0.) ? 1 : 0;

        aomm_threshold_factor_ = vm["AOMM.threshold_factor"].as<float>();

        str = vm["Coloring.algo"].as<std::string>();
        if (str.compare("RLF") == 0) coloring_algo_ = 0;
        if (str.compare("Greedy") == 0) coloring_algo_ = 1;

        str = vm["Coloring.scope"].as<std::string>();
        if (str.compare("local") == 0) coloring_algo_ += 10;

        str                         = vm["Run.type"].as<std::string>();
        max_electronic_steps        = vm["Quench.max_steps"].as<short>();
        max_electronic_steps_loose_ = max_electronic_steps;
        max_electronic_steps_tight_ = vm["Quench.max_steps_tight"].as<short>();
        if (str.compare("QUENCH") == 0)
        {
            atoms_dyn_ = 0;
        }
        if (str.compare("MD") == 0)
        {
            atoms_dyn_        = 2;
            dt                = vm["MD.dt"].as<float>();
            num_MD_steps      = vm["MD.num_steps"].as<short>();
            MD_last_step_     = vm["MD.last_step"].as<short>();
            md_print_freq     = vm["MD.print_interval"].as<short>();
            md_print_filename = vm["MD.print_directory"].as<std::string>();
            str               = vm["MD.thermostat"].as<std::string>();
            if (str.compare("ON") == 0 || str.compare("on") == 0)
            {
                str = vm["Thermostat.type"].as<std::string>();
                if (str.compare("Berendsen") == 0) thermostat_type = 1;
                if (str.compare("Langevin") == 0) thermostat_type = 2;
                if (str.compare("SCALING") == 0) thermostat_type = 3;
                if (thermostat_type < 0)
                {
                    (*MPIdata::sout) << "Invalid value for Thermostat.type: "
                                     << thermostat_type << std::endl;
                    MPI_Abort(comm_global_, EXIT_FAILURE);
                }

                tkel = vm["Thermostat.temperature"].as<float>();
                if (tkel < 0.)
                {
                    (*MPIdata::sout)
                        << "Invalid value for Thermostat.temperature: " << tkel
                        << std::endl;
                    MPI_Abort(comm_global_, EXIT_FAILURE);
                }
                thtime = vm["Thermostat.relax_time"].as<float>();
                if (thtime < 0.)
                {
                    (*MPIdata::sout)
                        << "Invalid value for Thermostat.relax_time: " << thtime
                        << std::endl;
                    MPI_Abort(comm_global_, EXIT_FAILURE);
                }

                if (str.compare("SCALING") == 0)
                {
                    thwidth = vm["Thermostat.width"].as<float>();
                    if (thwidth < 0.)
                    {
                        (*MPIdata::sout)
                            << "Invalid value for Thermostat.width: " << thwidth
                            << std::endl;
                        MPI_Abort(comm_global_, EXIT_FAILURE);
                    }
                }
            }
            else
            {
                thermostat_type = 0;
            }
            wf_extrapolation_ = vm["MD.extrapolation_type"].as<short>();
            enforceVmass0
                = vm["MD.remove_mass_center_motion"].as<bool>() ? 1 : 0;

            threshold_eigenvalue_gram_
                = vm["MD.min_Gram_eigenvalue"].as<float>();

            // override value of wf_extrapolation for XL-BOMD
            str = vm["MD.type"].as<std::string>();
        } // MD
        else
        {
            threshold_eigenvalue_gram_ = threshold_eigenvalue_gram_quench_;
        }

        if (str.compare("GeomOpt") == 0)
        {
            str = vm["GeomOpt.type"].as<std::string>();
            if (str.compare("LBFGS") == 0) atoms_dyn_ = 6;
            if (str.compare("FIRE") == 0) atoms_dyn_ = 7;
            dt = vm["GeomOpt.dt"].as<float>();

            num_MD_steps = vm["GeomOpt.max_steps"].as<short>();
            tol_forces   = vm["GeomOpt.tol"].as<float>();
            dt           = vm["GeomOpt.dt"].as<float>();
        }

        nempty_ = vm["Orbitals.nempty"].as<short>();
        str     = vm["Orbitals.type"].as<std::string>();
        if (str.compare("NO") == 0) orbital_type_ = 1;
        if (str.compare("Orthonormal") == 0) orbital_type_ = 2;
        std::cout << "Orbitals type: " << str << std::endl;
        etemp_ = vm["Orbitals.temperature"].as<float>();
        // K_B*T in Rydberg
        occ_width_ = 2. * 3.166811429e-6 * etemp_;

        str = vm["Orbitals.dotProduct"].as<std::string>();
        if (str.compare("diagonal") == 0)
            dot_product_type = 0; // use diag(inverse(S))
        if (str.compare("exact") == 0) dot_product_type = 1; // use inverse(S)

        str = vm["Orbitals.initial_type"].as<std::string>();
        if (str.compare("random") == 0) init_type = 0;
        if (str.compare("Gaussian") == 0) init_type = 1;
        if (str.compare("Fourier") == 0) init_type = 2;

        // width of initial Gaussian, otherwise just used to determine
        // if local or not
        init_rc = vm["Orbitals.initial_width"].as<float>();
        if (init_type == 1 && init_rc > 100.)
        {
            std::cout
                << std::endl
                << "!!!WARNING: Gaussian orbitals with large spreads may lead "
                   "to bad condition number for S!!!"
                << std::endl;
            std::cout << "Try using a smaller value for Orbitals.initial_width"
                      << std::endl
                      << std::endl;
        }

        overallocate_factor_ = vm["Orbitals.overallocate_factor"].as<float>();

        if (init_rc < 1000.)
            init_loc = 1;
        else
            init_loc = 0;

        cut_radius           = vm["LocalizationRegions.radius"].as<float>();
        tol_orb_centers_move = vm["LocalizationRegions.move_tol"].as<float>();
        lr_update = (short)vm["LocalizationRegions.adaptive"].as<bool>();
        min_distance_centers_
            = vm["LocalizationRegions.min_distance"].as<float>();
        lrs_compute = vm["LocalizationRegions.computation"].as<short>();
        str = vm["LocalizationRegions.extrapolation_scheme"].as<std::string>();
        if (str.compare("none") == 0)
            lrs_extrapolation = 0;
        else if (str.compare("linear") == 0)
            lrs_extrapolation = 1;
        else if (str.compare("quadratic") == 0)
            lrs_extrapolation = 2;
        else if (str.compare("Verlet") == 0)
            lrs_extrapolation = 10;

        dm_mix         = vm["DensityMatrix.mixing"].as<float>();
        dm_tol         = vm["DensityMatrix.tol"].as<float>();
        dm_inner_steps = vm["DensityMatrix.nb_inner_it"].as<short>();
        dm_use_old_    = vm["DensityMatrix.use_old"].as<bool>() ? 1 : 0;
        str            = vm["DensityMatrix.algo"].as<std::string>();
        if (str.compare("Diagonalization") == 0)
            dm_algo_ = 0;
        else if (str.compare("Chebyshev") == 0)
        {
            dm_algo_ = 1;
            // options for Chebyshev
            dm_approx_order
                = vm["DensityMatrix.approximation_order"].as<short>();
            dm_approx_ndigits
                = vm["DensityMatrix.approximation_ndigits"].as<short>();
            dm_approx_power_maxits
                = vm["DensityMatrix.approximation_power_maxits"].as<short>();
        }
        else
            dm_algo_ = 2;

        str = vm["DensityMatrix.solver"].as<std::string>();
        if (str.compare("Mixing") == 0) DM_solver_ = 0;
        if (str.compare("MVP") == 0) DM_solver_ = 1;
        if (str.compare("HMVP") == 0) DM_solver_ = 2;

        load_balancing_alpha = vm["LoadBalancing.alpha"].as<float>();
        load_balancing_damping_tol
            = vm["LoadBalancing.damping_tol"].as<float>();
        load_balancing_max_iterations
            = vm["LoadBalancing.max_iterations"].as<short>();
        load_balancing_modulo = vm["LoadBalancing.modulo"].as<short>();
        load_balancing_output_file
            = vm["LoadBalancing.output_file"].as<std::string>();
        if (load_balancing_output_file.compare("") == 0)
            write_clusters = 0;
        else
            write_clusters = 1;

        // derived flags
        loc_mode_ = cut_radius < 100. ? true : false;
        if (lrs_compute > 0 || !loc_mode_) lrs_extrapolation = 0;

        precond_type_ = 10;

        if (vm.count("Quench.MLWC"))
        {
            wannier_transform_type = vm["Quench.MLWC"].as<bool>() ? 1 : 0;
        }
        else // default
        {
            if (loc_mode_)
                wannier_transform_type = 1;
            else
                wannier_transform_type = 0;
        }
        wannier_transform_type
            = vm["Quench.MLWF"].as<bool>() ? 2 : wannier_transform_type;
        compute_dipole_ = vm["Quench.Dipole"].as<bool>() ? 1 : 0;

        maxDistanceAtomicInfo_ = vm["Parallel.atomic_info_radius"].as<float>();

        // options not available in configure file
        lr_updates_type         = 0;
        precond_factor_computed = false;
        mix_pot                 = 1.;
        project_out_psd         = 0;
        multipole_order         = 1;

    } // onpe0

    // synchronize all processors
    sync();

#ifdef MGMOL_HAS_LIBROM
    setROMOptions(vm);
#endif
}

int Control::checkOptions()
{
    if (it_algo_type_ > 3)
    {
        std::cerr << "ERROR: specified inner solver not implemented"
                  << std::endl;
        return -1;
    }

    if (getOrthoType() == OrthoType::UNDEFINED)
    {
        std::cerr << "ERROR: unknown orbitals type\n";
        return -1;
    }

    if (short_sighted && lap_type == 0)
    {
        std::cerr
            << "ERROR: Mehrstellen not compatible with Short-sighted algorithm!"
            << std::endl;
        return -1;
    }

    if (DM_solver_ > 0 && lap_type == 0)
    {
        std::cerr << "DM_solver_ = " << DM_solver_ << std::endl;
        std::cerr << "ERROR: Mehrstellen not compatible with MVP inner solvers!"
                  << std::endl;
        return -1;
    }

    if (it_algo_type_ == 3 && lap_type == 0)
    {
        std::cerr
            << "ERROR: Mehrstellen not compatible with Polak-Ribiere algorithm!"
            << std::endl;
        return -1;
    }

    if (out_restart_file_type == 1 && !globalColoring() && out_restart_info > 2)
    {
        std::cerr << "ERROR: writing single restart file with wave functions "
                     "requires global coloring!!!"
                  << std::endl;
        return -1;
    }

    if (restart_file_type == 1 && !globalColoring() && restart_info > 2)
    {
        std::cerr << "ERROR: reading single restart file with wave functions "
                     "requires global coloring!!!"
                  << std::endl;
        return -1;
    }
    if (it_algo_type_ == 2 && lap_type == 0)
    {
        (*MPIdata::sout) << "ERROR: Davidson and Mehrstellen incompatible!"
                         << std::endl;
        return -1;
    }
    if (short_sighted > 0 && !loc_mode_)
    {
        (*MPIdata::sout)
            << "ERROR: Short-sighted algorithm requires localization mode!!"
            << std::endl;
        return -1;
    }
    if (lrs_extrapolation > 0 && lrs_compute > 0)
    {
        (*MPIdata::sout) << "ERROR: must choose either extrapolation or "
                            "computation of centers."
                         << std::endl;
    }
    if (dm_approx_order < 0 && DMEigensolver() == DMEigensolverType::Chebyshev)
    {
        (*MPIdata::sout) << "ERROR: Order for Chebyshev approximation for the "
                            "density matrix must be > 0"
                         << std::endl;
    }
    if (dm_approx_ndigits < 1
        && DMEigensolver() == DMEigensolverType::Chebyshev)
    {
        (*MPIdata::sout)
            << "ERROR: Number of digits of precision for Chebyshev "
               "approximation for the density matrix must be > 0"
            << std::endl;
    }
    if (dm_approx_power_maxits < 1
        && DMEigensolver() == DMEigensolverType::Chebyshev)
    {
        (*MPIdata::sout)
            << "ERROR: Max. number of iterations for computing interval for "
               "Chebyshev approximation for the density matrix must be > 0"
            << std::endl;
    }
    if (compute_dipole_ && !rmatrices)
    {
        (*MPIdata::sout)
            << "ERROR: Dipole computation requires replicated matrices"
            << std::endl;
        return -1;
    }
    if (std::abs(ex_) > 0. && bcWF[0] == 1)
    {
        (*MPIdata::sout) << "ERROR: Electrostatic field in x-rection requires "
                            "non-periodic BC"
                         << std::endl;
        return -1;
    }
    if (std::abs(ey_) > 0. && bcWF[1] == 1)
    {
        (*MPIdata::sout) << "ERROR: Electrostatic field in y-rection requires "
                            "non-periodic BC"
                         << std::endl;
        return -1;
    }
    if (std::abs(ez_) > 0. && bcWF[2] == 1)
    {
        (*MPIdata::sout) << "ERROR: Electrostatic field in z-rection requires "
                            "non-periodic BC"
                         << std::endl;
        return -1;
    }
    return 0;
}

void Control::printRestartLink()
{
    if (mype_ == 0)
    {
        char buf[512];
        int count = readlink(restart_file.c_str(), buf, sizeof(buf));
        if (count >= 0)
        {
            buf[count] = '\0';
            printf("Restart file: %s -> %s\n", restart_file.c_str(), buf);
        }
    }
}

void Control::printPoissonOptions(std::ostream& os)
{
    os << " Finite difference scheme for Poisson: ";
    switch (poisson_lap_type_)
    {
        case PoissonFDtype::h2:
            os << "2nd order";
            break;
        case PoissonFDtype::h4:
            os << "4th order";
            break;
        case PoissonFDtype::h6:
            os << "6th order";
            break;
        case PoissonFDtype::h8:
            os << "8th order";
            break;
        case PoissonFDtype::h4M:
            os << "Mehrstellen 4th order";
            break;
        case PoissonFDtype::h4MP:
            os << "MehrstellenP 4th order";
            break;
        default:
            os << "Undefined!!!";
    }
    os << std::endl;
}

void Control::setROMOptions(const boost::program_options::variables_map& vm)
{
    printWithTimeStamp("Control::setROMOptions()...", std::cout);

    if (onpe0)
    {
        std::string str = vm["ROM.stage"].as<std::string>();
        if (str.compare("offline") == 0)
            rom_pri_option.rom_stage = ROMStage::OFFLINE;
        else if (str.compare("online") == 0)
            rom_pri_option.rom_stage = ROMStage::ONLINE;
        else if (str.compare("build") == 0)
            rom_pri_option.rom_stage = ROMStage::BUILD;
        else if (str.compare("online_pinned_H2O_3dof") == 0)
            rom_pri_option.rom_stage = ROMStage::ONLINE_PINNED_H2O_3DOF;
        else if (str.compare("test_orbital") == 0)
            rom_pri_option.rom_stage = ROMStage::TEST_ORBITAL;
        else if (str.compare("online_poisson") == 0)
            rom_pri_option.rom_stage = ROMStage::ONLINE_POISSON;
        else if (str.compare("test_poisson") == 0)
            rom_pri_option.rom_stage = ROMStage::TEST_POISSON;
        else if (str.compare("test_rho") == 0)
            rom_pri_option.rom_stage = ROMStage::TEST_RHO;
        else if (str.compare("test_ion") == 0)
            rom_pri_option.rom_stage = ROMStage::TEST_ION;
        else if (str.compare("none") == 0)
            rom_pri_option.rom_stage = ROMStage::UNSUPPORTED;

        rom_pri_option.restart_file_fmt
            = vm["ROM.offline.restart_filefmt"].as<std::string>();
        rom_pri_option.restart_file_minidx
            = vm["ROM.offline.restart_min_idx"].as<int>();
        rom_pri_option.restart_file_maxidx
            = vm["ROM.offline.restart_max_idx"].as<int>();
        rom_pri_option.basis_file
            = vm["ROM.offline.basis_file"].as<std::string>();

        str = vm["ROM.offline.variable"].as<std::string>();
        if (str.compare("orbitals") == 0)
            rom_pri_option.variable = ROMVariable::ORBITALS;
        else if (str.compare("potential") == 0)
            rom_pri_option.variable = ROMVariable::POTENTIAL;
        else
            rom_pri_option.variable = ROMVariable::NONE;

        rom_pri_option.save_librom_snapshot
            = vm["ROM.offline.save_librom_snapshot"].as<bool>();
        rom_pri_option.librom_snapshot_freq
            = vm["ROM.offline.librom_snapshot_freq"].as<int>();

        rom_pri_option.compare_md = vm["ROM.basis.compare_md"].as<bool>();
        rom_pri_option.num_orbbasis
            = vm["ROM.basis.number_of_orbital_basis"].as<int>();
        rom_pri_option.num_potbasis
            = vm["ROM.basis.number_of_potential_basis"].as<int>();
        rom_pri_option.pot_rom_file
            = vm["ROM.potential_rom_file"].as<std::string>();
    } // onpe0

    // synchronize all processors
    syncROMOptions();
}

void Control::syncROMOptions()
{
    if (onpe0 && verbose > 0)
        (*MPIdata::sout) << "Control::syncROMOptions()" << std::endl;

    MGmol_MPI& mmpi = *(MGmol_MPI::instance());

    mmpi.bcast(rom_pri_option.restart_file_fmt, comm_global_);
    mmpi.bcast(rom_pri_option.basis_file, comm_global_);
    mmpi.bcast(rom_pri_option.pot_rom_file, comm_global_);

    auto bcast_check = [](int mpirc)
    {
        if (mpirc != MPI_SUCCESS)
        {
            (*MPIdata::sout) << "MPI Bcast of Control failed!!!" << std::endl;
            MPI_Abort(comm_global_, 2);
        }
    };

    short rom_stage = (short)static_cast<int>(rom_pri_option.rom_stage);
    int mpirc;
    mpirc = MPI_Bcast(&rom_stage, 1, MPI_SHORT, 0, comm_global_);
    bcast_check(mpirc);

    mpirc = MPI_Bcast(
        &rom_pri_option.restart_file_minidx, 1, MPI_INT, 0, comm_global_);
    bcast_check(mpirc);

    mpirc = MPI_Bcast(
        &rom_pri_option.restart_file_maxidx, 1, MPI_INT, 0, comm_global_);
    bcast_check(mpirc);

    mpirc = MPI_Bcast(
        &rom_pri_option.save_librom_snapshot, 1, MPI_C_BOOL, 0, comm_global_);
    bcast_check(mpirc);

    mpirc = MPI_Bcast(
        &rom_pri_option.librom_snapshot_freq, 1, MPI_INT, 0, comm_global_);
    bcast_check(mpirc);

    short rom_var = (short)static_cast<int>(rom_pri_option.variable);
    mpirc         = MPI_Bcast(&rom_var, 1, MPI_SHORT, 0, comm_global_);
    bcast_check(mpirc);

    rom_pri_option.rom_stage = static_cast<ROMStage>(rom_stage);
    rom_pri_option.variable  = static_cast<ROMVariable>(rom_var);

    mpirc
        = MPI_Bcast(&rom_pri_option.compare_md, 1, MPI_C_BOOL, 0, comm_global_);
    bcast_check(mpirc);

    mpirc
        = MPI_Bcast(&rom_pri_option.num_orbbasis, 1, MPI_INT, 0, comm_global_);
    bcast_check(mpirc);

    mpirc
        = MPI_Bcast(&rom_pri_option.num_potbasis, 1, MPI_INT, 0, comm_global_);
    bcast_check(mpirc);
}

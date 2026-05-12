// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "MGmol_prototypes.h"
#include <cassert>
#include <fenv.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sys/cdefs.h>
#include <vector>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

int read_config(int argc, char** argv, po::variables_map& vm,
    std::string& coords_filename, std::string& lrs_filename,
    std::string& constraints_filename, float& total_spin, bool& with_spin)
{
    // use configure file if it can be found
    // std::string config_filename("mgmol.cfg");
    try
    {
        std::string config_file;

        // Declare a group of options that will be
        // allowed only on command line
        po::options_description generic("Generic options");
        generic.add_options()("version,v", "print version std::string")(
            "help,h", "produce help message")("check", "check input")(
            "config,c",
            po::value<std::string>(&config_file)->default_value("mgmol.cfg"),
            "name of configuration file")("atomicCoordinates,i",
            po::value<std::string>(&coords_filename),
            "atomic coordinates filename")("LRsFilename,l",
            po::value<std::string>(&lrs_filename), "LRs filename");

        // Declare a group of options (with default when appropriate) that
        // will be allowed in config file
        po::options_description config("Configuration");
        config.add_options()("spin", po::value<float>(), "system total spin")(
            "verbosity", po::value<short>()->default_value(1),
            "verbosity level")("constraintsFilename",
            po::value<std::string>(&constraints_filename),
            "Name of file with list of constraints")("xcFunctional",
            po::value<std::string>()->required(), "XC functional: LDA or PBE")(
            "FDtype", po::value<std::string>()->default_value("Mehrstellen"),
            "Finite Difference scheme")("charge",
            po::value<short>()->default_value(0),
            "system total charge")("Domain.lx", po::value<float>()->required(),
            "domain dimension in x direction")("Domain.ly",
            po::value<float>()->required(), "domain dimension in y direction")(
            "Domain.lz", po::value<float>()->required(),
            "domain dimension in z direction")("Domain.ox",
            po::value<float>()->required(), "domain origin in x direction")(
            "Domain.oy", po::value<float>()->required(),
            "domain origin in y direction")("Domain.oz",
            po::value<float>()->required(), "domain origin in z direction")(
            "Mesh.nx", po::value<short>()->required(),
            "mesh dimension in x direction")("Mesh.ny",
            po::value<short>()->required(), "mesh dimension in y direction")(
            "Mesh.nz", po::value<short>()->required(),
            "mesh dimension in z direction")("Potentials.pseudopotential",
            po::value<std::vector<std::string>>()->multitoken(),
            "pseudopotentials list")("Potentials.external",
            po::value<std::vector<std::string>>()->multitoken(),
            "external potentials list")("Potentials.binExternal",
            po::value<bool>()->default_value(true),
            "binary external potential")("Efield.ex",
            po::value<float>()->default_value(0.),
            "Efield in x-direction")("Efield.ey",
            po::value<float>()->default_value(0.), "Efield in y-direction")(
            "Efield.ez", po::value<float>()->default_value(0.),
            "Efield in z-direction")("Restart.input_filename",
            po::value<std::string>()->default_value(""),
            "Read restart filename/directory")("Restart.input_level",
            po::value<short>()->default_value(0),
            "Read restart level")("Restart.input_type",
            po::value<std::string>()->default_value("distributed"),
            "Read restart type: distributed or single_file")(
            "Restart.output_filename",
            po::value<std::string>()->default_value("auto"),
            "Dump restart filename/directory")("Restart.output_level",
            po::value<short>()->default_value(3),
            "Write restart level")("Restart.output_type",
            po::value<std::string>()->default_value("distributed"),
            "Write restart type: distributed or single_file")(
            "Restart.interval", po::value<short>()->default_value(1000),
            "Restart frequency")("Restart.rescale_v",
            po::value<double>()->default_value(1.),
            "rescaling factor velocity of all atoms")("Poisson.bcx",
            po::value<std::string>()->default_value("periodic"),
            "boundary condition x")("Poisson.bcy",
            po::value<std::string>()->default_value("periodic"),
            "boundary condition y")("Poisson.bcz",
            po::value<std::string>()->default_value("periodic"),
            "boundary condition z")("Poisson.diel",
            po::value<std::string>()->default_value("off"),
            "continuum solvent: on/off")("Poisson.precond_precision",
            po::value<short>()->default_value(32),
            "Precision for Poisson Preconditioner")("Run.type",
            po::value<std::string>()->default_value("QUENCH"), "Run type")(
            "Quench.solver", po::value<std::string>()->default_value("ABPG"),
            "Iterative solver for quench")("Quench.max_steps",
            po::value<short>()->default_value(200),
            "Max. steps in loose quench")("Quench.max_steps_tight",
            po::value<short>()->default_value(1000),
            "Max. steps in tight quench")("Quench.atol",
            po::value<float>()->default_value(1.e-12),
            "Abs. tol. in quench convergence")("Quench.rtol",
            po::value<float>()->default_value(-1.),
            "Rel. tol. in quench convergence")("Quench.conv_criterion",
            po::value<std::string>()->default_value("deltaE"),
            "Convergence criterion")("Quench.MLWC", po::value<bool>(),
            "Compute MLWC in quench")("Quench.MLWF",
            po::value<bool>()->default_value(false),
            "Compute MLWF (apply rotation) in quench")("Quench.Dipole",
            po::value<bool>()->default_value(false), "Compute dipole moment")(
            "Quench.num_lin_iterations", po::value<short>()->default_value(0),
            "Number of iterations without potential update in quench")(
            "Preconditioner.num_levels", po::value<short>()->default_value(2),
            "Number of levels for MG preconditioner")(
            "Preconditioner.npresmoothing",
            po::value<short>()->default_value(2),
            "Number of presmoothing steps i preconditioner")(
            "Preconditioner.npostsmoothing",
            po::value<short>()->default_value(2),
            "Number of postsmoothing steps i preconditioner")(
            "Preconditioner.precision", po::value<short>()->default_value(32),
            "Precision for MG preconditioner")("Quench.spread_penalty_damping",
            po::value<float>()->default_value(0.),
            "Spread penalty damping factor")("Quench.spread_penalty_target",
            po::value<float>()->default_value(2.),
            "Spread penalty target")("SpreadPenalty.type",
            po::value<std::string>()->default_value("individual"),
            "Spread penalty type (individual,volume)")("SpreadPenalty.damping",
            po::value<float>()->default_value(1.),
            "Spread penalty damping factor")("SpreadPenalty.target",
            po::value<float>()->default_value(-1.),
            "Spread penalty target")("SpreadPenalty.alpha",
            po::value<float>()->default_value(0.), "Spread penalty factor")(
            "MD.num_steps", po::value<short>()->default_value(1),
            "number of MD steps")("MD.last_step",
            po::value<short>()->default_value(-1), "last MD step")("MD.dt",
            po::value<float>(), "time step for MD (a.u.)")("MD.print_interval",
            po::value<short>()->default_value(1),
            "print intervale for MD data")("MD.print_directory",
            po::value<std::string>()->default_value("MD"),
            "print directory for MD data")("MD.thermostat",
            po::value<std::string>()->default_value("OFF"),
            "MD thermostat: ON or OFF")("MD.remove_mass_center_motion",
            po::value<bool>()->default_value(true),
            "Remove mass center motion")("MD.type",
            po::value<std::string>()->default_value("BOMD"), "MD type: BOMD")(
            "GeomOpt.type", po::value<std::string>()->default_value("LBFGS"),
            "Geometry optimization algorithm")("GeomOpt.tol",
            po::value<float>()->default_value(4.e-4),
            "Tolerance on forces for Geometry optimization")(
            "GeomOpt.max_steps", po::value<short>()->default_value(1),
            "max. number of Geometry optimization steps")("GeomOpt.dt",
            po::value<float>(),
            "Delta t for trial pseudo-time steps")("Thermostat.type",
            po::value<std::string>()->default_value("Langevin"),
            "Thermostat type")("Thermostat.temperature",
            po::value<float>()->default_value(-1.), "Thermostat temperature")(
            "Thermostat.relax_time", po::value<float>()->default_value(-1.),
            "Thermostat relaxation time")("Thermostat.width",
            po::value<float>()->default_value(-1.),
            "Thermostat width (for SCALING)")("Orbitals.nempty",
            po::value<short>()->default_value(0),
            "Number of empty orbitals")("Orbitals.initial_type",
            po::value<std::string>()->default_value("random"),
            "initial orbitals type")("Orbitals.initial_width",
            po::value<float>()->default_value(10000.),
            "initial orbitals radius")("Orbitals.temperature",
            po::value<float>()->default_value(0.),
            "electronic temperature [K]")("Orbitals.bcx",
            po::value<std::string>()->default_value("periodic"),
            "orbitals boundary condition x")("Orbitals.bcy",
            po::value<std::string>()->default_value("periodic"),
            "orbitals boundary condition y")("Orbitals.bcz",
            po::value<std::string>()->default_value("periodic"),
            "orbitals boundary condition z")("Orbitals.occupations",
            po::value<std::string>(),
            "orbitals occupations")("ProjectedMatrices.solver",
            po::value<std::string>()->default_value("exact"),
            "solver for projected matrices")("ProjectedMatrices.printMM",
            po::value<bool>()->default_value(false),
            "print projected matrices in MM format")(
            "ProjectedMatrices.replicated",
#ifdef MGMOL_USE_SCALAPACK
            po::value<bool>()->default_value(false),
#else
            po::value<bool>()->default_value(true),
#endif
            "use replicated projected matrices")("LocalizationRegions.radius",
            po::value<float>()->default_value(1000.),
            "Localization regions radius")("LocalizationRegions.adaptive",
            po::value<bool>()->default_value(true),
            "Localization regions adaptivity")("LocalizationRegions.move_tol",
            po::value<float>()->default_value(1000.),
            "Localization regions move tolerance")(
            "Parallel.atomic_info_radius",
            po::value<float>()->default_value(8.),
            "Max. distance for atomic data communication")(
            "LoadBalancing.alpha", po::value<float>()->default_value(0.0),
            "Parameter for computing bias for load balancing algo")(
            "LoadBalancing.damping_tol", po::value<float>()->default_value(0.9),
            "Damping parameter for computing bias for load balancing algo")(
            "LoadBalancing.max_iterations",
            po::value<short>()->default_value(50),
            "Maximum number of iterations for load balancing algo")(
            "LoadBalancing.modulo", po::value<short>()->default_value(1),
            "Modulos or parameter to control how often clusters are "
            "recomputed during md")("LoadBalancing.output_file",
            po::value<std::string>()->default_value(""),
            "Output file for dumping cluster information in vtk format");

#ifdef MGMOL_HAS_LIBROM
        setupROMConfigOption(config);
#endif

        // Hidden options, will be allowed in config file, but will not be
        // shown to the user.
        po::options_description hidden("Hidden options");
        hidden.add_options()("Quench.interval_print_residual",
            po::value<short>()->default_value(0),
            "Print interval for residual in quench (0 for none)")(
            "Quench.required_tol", po::value<float>()->default_value(1000.),
            "required tolerance (code stops if not reached)")(
            "Quench.compute_cond_Gram", po::value<bool>()->default_value(false),
            "Compute condition number of S during quench")(
            "Quench.min_Gram_eigenvalue", po::value<float>()->default_value(0.),
            "min. eigenvalue for Gram matrix")("Quench.step_length",
            po::value<float>()->default_value(-1.),
            "Step length for corrections in quench")("Quench.ortho_freq",
            po::value<short>()->default_value(1000),
            "Orthonormalization frequency")(
            "Quench.pair_mlwf_distance_threshold",
            po::value<float>()->default_value(4.),
            "Max. distance between pairs for MLWF transform")(
            "AOMM.kernel_radius", po::value<float>()->default_value(-1.),
            "Radius of kernel functions in AOMM algorithm")(
            "AOMM.threshold_factor", po::value<float>()->default_value(-1.),
            "Multiplicative factor of kernel radius to set threshold in "
            "AOMM projectors dropping")("Orbitals.type",
            po::value<std::string>()->default_value("NO"),
            "orbital type")("Orbitals.dotProduct",
            po::value<std::string>()->default_value("diagonal"),
            "orbital dot product type")("Orbitals.overallocate_factor",
            po::value<float>()->default_value(1.2),
            "safety factor to use for static allocation of orbitals")(
            "Potentials.filterPseudo", po::value<char>()->default_value('f'),
            "filter")("Poisson.solver",
            po::value<std::string>()->default_value("PCG"),
            "solver")("Poisson.e0", po::value<float>()->default_value(78.36),
            "continuum solvent: epsilon0")("Poisson.rho0",
            po::value<float>()->default_value(0.0004),
            "continuum solvent: rho0")("Poisson.beta",
            po::value<float>()->default_value(1.3),
            "continuum solvent: beta")("Poisson.FDtype",
            po::value<std::string>()->default_value("Mehrstellen"),
            "FDtype")("Poisson.nu1", po::value<short>()->default_value(1),
            "MG pre-smoothing sweeps")("Poisson.nu2",
            po::value<short>()->default_value(1), "MG post-smoothing sweeps")(
            "Poisson.max_steps", po::value<short>()->default_value(20),
            "max. nb. steps Poisson solver")("Poisson.max_steps_initial",
            po::value<short>()->default_value(20),
            "max. nb. steps Poisson solver in first solve")(
            "Poisson.max_levels", po::value<short>()->default_value(10),
            "max. nb. levels for MG solver or Precon")("Poisson.conv_tol",
            po::value<float>()->default_value(1.e-8),
            "Convergence tolerance for Poisson solver")("Poisson.reset",
            po::value<bool>()->default_value(false),
            "reset Hartree potential at each MD step")("ABPG.m",
            po::value<short>()->default_value(1),
            "History length for Anderson extrapolation")("ABPG.beta",
            po::value<float>()->default_value(1.),
            "beta for Anderson extrapolation")("NLCG.parallel_transport",
            po::value<bool>()->default_value(true),
            "Turn ON/OFF parallel transport algorithm")("MD.extrapolation_type",
            po::value<short>()->default_value(1), "MD extrapolation type")(
            "MD.compute_cond_Gram", po::value<bool>()->default_value(false),
            "Compute condition number of S at end of quench")(
            "MD.min_Gram_eigenvalue", po::value<float>()->default_value(0.),
            "min. eigenvalue for Gram matrix")(
            "ShortSightedInverse.spread_factor",
            po::value<float>()->default_value(2.),
            "Shortsighted spread factor")("ShortSightedInverse.tol",
            po::value<float>()->default_value(1.e-10),
            "Shortsighted tolerance")("ShortSightedInverse.krylov_dim",
            po::value<short>()->default_value(10),
            "Shortsighted Krylov space max. dimension")(
            "ShortSightedInverse.max_iterations",
            po::value<short>()->default_value(10),
            "Shortsighted max. number of GMRES iterations")(
            "ShortSightedInverse.ilu_type",
            po::value<std::string>()->default_value("ILU"),
            "Shortsighted ILU type: ILU or ILUT")(
            "ShortSightedInverse.ilu_drop_tol",
            po::value<float>()->default_value(1.e-5),
            "Shortsighted ILU dropping tolerance")(
            "ShortSightedInverse.ilu_filling_level",
            po::value<short>()->default_value(0),
            "Shortsighted ILU filling level")(
            "ShortSightedInverse.ilut_max_fill",
            po::value<int>()->default_value(10000),
            "Shortsighted max. filling for ILUT")("Coloring.algo",
            po::value<std::string>()->default_value("RLF"),
            "Coloring algorithm: RLF or Greedy")("Coloring.scope",
            po::value<std::string>()->default_value("local"),
            "Coloring scope: local or global")(
            "LocalizationRegions.min_distance",
            po::value<float>()->default_value(0.),
            "min. distance between Localization centers")(
            "LocalizationRegions.extrapolation_scheme",
            po::value<std::string>()->default_value("linear"),
            "Extrapolation order for localization centers")(
            "LocalizationRegions.computation",
            po::value<short>()->default_value(0),
            "Flag for computing new centers from extrapolated orbitals.")(
            "DensityMatrix.mixing", po::value<float>()->default_value(-1.),
            "Mixing coefficient for Density Matrix")("DensityMatrix.tol",
            po::value<float>()->default_value(1.e-12),
            "Tolerance for Density Matrix convergence")("DensityMatrix.solver",
            po::value<std::string>()->default_value("Mixing"),
            "Algorithm for updating Density Matrix: Mixing, MVP, HMVP")(
            "DensityMatrix.nb_inner_it", po::value<short>()->default_value(3),
            "Max. number of inner iterations in DM optimization")(
            "DensityMatrix.algo",
            po::value<std::string>()->default_value("Diagonalization"),
            "Algorithm for computing Density Matrix. "
            "Diagonalization or SP2 or Chebyshev.")("DensityMatrix.use_old",
            po::value<bool>()->default_value(true),
            "Start DM optimization with matrix of previous WF step")(
            "DensityMatrix.approximation_order",
            po::value<short>()->default_value(500),
            "Polynomial order for Chebyshev approximation "
            "of density matrix. ")("DensityMatrix.approximation_ndigits",
            po::value<short>()->default_value(1),
            "Number of digits of precision for Chebyshev "
            "approximation of density matrix. ")(
            "DensityMatrix.approximation_power_maxits",
            po::value<short>()->default_value(100),
            "Maximum number of iterations for power method "
            "to compute interval for Chebyshev "
            "approximation of density matrix. ");

        po::options_description cmdline_options;
        cmdline_options.add(generic);

        po::options_description config_file_options;
        config_file_options.add(config).add(hidden);

        po::options_description visible("Allowed options");
        visible.add(generic).add(config);

        po::positional_options_description pd;
        pd.add("atomicCoordinates", -1);

        store(po::command_line_parser(argc, argv)
                  .options(cmdline_options)
                  .positional(pd)
                  .run(),
            vm);
        notify(vm);

        std::ifstream ifs(config_file.c_str());
        if (!ifs)
        {
            std::cout << "can not open config file: " << config_file << "\n";
            return 1;
        }
        else
        {
            store(parse_config_file(ifs, config_file_options), vm);
            notify(vm);
        }

        if (vm.count("help"))
        {
            std::cout << visible << "\n";
            return 0;
        }

        if (vm.count("version"))
        {
#ifdef GITHASH
#define xstr(x) #x
#define LOG(x) std::cout << " MGmol: git_hash " << xstr(x) << std::endl;
            LOG(GITHASH);
            std::cout << std::endl;
#endif
            return 0;
        }

        if (vm.count("spin"))
        {
            total_spin = vm["spin"].as<float>();
            with_spin  = true;
            std::cout << "Spin was set to " << total_spin << std::endl;
        }
        if (vm.count("atomicCoordinates"))
        {
            coords_filename
                = vm["atomicCoordinates"].as<std::vector<std::string>>()[0];
            std::cout << "Coordinates files is: " << coords_filename << "\n";
        }
        else
        {
            if (vm["Restart.input_level"].as<short>() == 0)
            {
                throw std::runtime_error(
                    "ERROR: No coordinates file provided!!!");
            }
        }

        std::cout << "Spin: " << total_spin << "\n";

    } // try
    catch (std::exception& e)
    {
        std::cerr << e.what() << "\n";
        return 1;
    }

    return 0;
}

#ifdef MGMOL_HAS_LIBROM
void setupROMConfigOption(po::options_description& rom_cfg)
{
    rom_cfg.add_options()("ROM.stage",
        po::value<std::string>()->default_value("none"),
        "ROM workflow stage: offline; build; online; none.")(
        "ROM.offline.restart_filefmt",
        po::value<std::string>()->default_value(""),
        "File name format to read for snapshots.")(
        "ROM.offline.restart_min_idx", po::value<int>()->default_value(-1),
        "Minimum index for snapshot file format.")(
        "ROM.offline.restart_max_idx", po::value<int>()->default_value(-1),
        "Maximum index for snapshot file format.")("ROM.offline.basis_file",
        po::value<std::string>()->default_value(""),
        "File name for libROM snapshot/POD matrices.")(
        "ROM.offline.save_librom_snapshot",
        po::value<bool>()->default_value(false),
        "Save libROM snapshot file at FOM simulation.")(
        "ROM.offline.librom_snapshot_freq", po::value<int>()->default_value(-1),
        "Frequency of saving libROM snapshot file at FOM simulation.")(
        "ROM.offline.variable", po::value<std::string>()->default_value(""),
        "FOM variable to perform POD: either orbitals or potential.")(
        "ROM.basis.compare_md", po::value<bool>()->default_value(false),
        "Compare MD or single-step force.")("ROM.basis.number_of_orbital_basis",
        po::value<int>()->default_value(-1),
        "Number of orbital POD basis.")("ROM.basis.number_of_potential_basis",
        po::value<int>()->default_value(-1),
        "Number of potential POD basis to build Hartree potential ROM "
        "operator.")("ROM.potential_rom_file",
        po::value<std::string>()->default_value(""),
        "File name to save/load potential ROM operators.");
}
#endif

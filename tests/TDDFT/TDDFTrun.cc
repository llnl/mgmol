// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE


#include "Control.h"
#include "ExtendedGridOrbitals.h"
#include "LocGridOrbitals.h"
#include "MGmol.h"
#include "MGmol_MPI.h"
#include "MPIdata.h"
#include "mgmol_run.h"
#include "DensityMatrix.h"
#include "ProjectedMatrices.h"

#include <cassert>
#include <iostream>
#include <time.h>
#include <vector>
#include <iomanip>
#include <fstream>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#define LOGGING 1
#define PRINT_INTERVAL 1

int propagate_density_matrix(ReplicatedMatrix& X_real, ReplicatedMatrix& X_imag,
    ReplicatedMatrix& H, double dt, double tol, int maxits = 10);
void compute_commutator(const ReplicatedMatrix& H, const ReplicatedMatrix& X,
    ReplicatedMatrix& commHX);

int main(int argc, char** argv)
{
    int mpirc = MPI_Init(&argc, &argv);
    if (mpirc != MPI_SUCCESS)
    {
        std::cerr << "MPI Initialization failed!!!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 0);
    }

    MPI_Comm comm = MPI_COMM_WORLD;
    mgmol_init(comm);

    std::string input_filename("");
    std::string lrs_filename;
    std::string constraints_filename("");
    float total_spin = 0.;
    bool with_spin    = false;

    po::variables_map vm;
    if (MPIdata::onpe0)
    {
        read_config(argc, argv, vm, input_filename, lrs_filename,
            constraints_filename, total_spin, with_spin);
    }

    MGmol_MPI::setup(comm, std::cout, with_spin);
    MGmol_MPI& mmpi      = *(MGmol_MPI::instance());
    MPI_Comm global_comm = mmpi.commGlobal();

    Control::setup(global_comm, with_spin, total_spin);
    Control& ct = *(Control::instance());
    ct.setOptions(vm);

    int ret = ct.checkOptions();
    if (ret < 0) return ret;

    mmpi.bcastGlobal(input_filename);
    mmpi.bcastGlobal(lrs_filename);

    {
        if (MPIdata::onpe0)
        {
            std::cout << "-------------------------" << std::endl;
            std::cout << "Construct MGmol object..." << std::endl;
            std::cout << "-------------------------" << std::endl;
        }

        MGmolInterface* mgmol;
        if (ct.isLocMode())
            mgmol = new MGmol<LocGridOrbitals<ORBDTYPE>>(global_comm, *MPIdata::sout,
                input_filename, lrs_filename, constraints_filename);
        else
            mgmol = new MGmol<ExtendedGridOrbitals<ORBDTYPE>>(global_comm, *MPIdata::sout,
                input_filename, lrs_filename, constraints_filename);

        mgmol->setup();

        std::vector<double> positions;
        mgmol->getAtomicPositions(positions);
        std::vector<short> anumbers;
        mgmol->getAtomicNumbers(anumbers);

        Mesh* mymesh           = Mesh::instance();
        const pb::Grid& mygrid = mymesh->grid();

        /* cast directly to ProjectedMatrices class to use derived class functionality */
        std::shared_ptr<ProjectedMatrices<ReplicatedMatrix>> dprojmatrices =
            std::dynamic_pointer_cast<ProjectedMatrices<ReplicatedMatrix>>(
                mgmol->getProjectedMatrices());

        ExtendedGridOrbitals<ORBDTYPE> orbitals("new_orbitals", mygrid,
            mymesh->subdivx(), ct.numst, ct.bcWF, dprojmatrices.get(), nullptr,
            nullptr, nullptr, nullptr);

        const pb::PEenv& myPEenv = mymesh->peenv();
        HDFrestart h5file(ct.restart_file, myPEenv, ct.out_restart_file_type);
        orbitals.read_hdf5(h5file);   // load quenched orbitals from driver 1

        /* update iterative index to differentiate between these orbitals and
         * mgmol object's internal orbitals (aka current_orbitals_)
        */
        orbitals.setIterativeIndex(orbitals.getIterativeIndex() + 1000);
        // Re-evaluate against the reloaded orbitals to populate the DM
        // (verified: matches the original quench to ~7 sig figs; see
        // earlier comparison test)
        std::vector<double> forces;
        dprojmatrices->setDMuniform(ct.getNelSpin());
        ct.dm_inner_steps = 60;
        double eks = mgmol->evaluateDMandEnergyAndForces(&orbitals, positions, anumbers, forces);
//        double eks = mgmol->evaluateEnergyAndForces(positions, anumbers, forces);

        /* TDDFT propagation */
        double t  = 0.0;
        double dt = ct.dt;
        int iters = 0, avg_iters = 0;

        /* Initialize Ionic Stepper for MD */
        std::shared_ptr<Ions> dions = mgmol->getIons();        

        ReplicatedMatrix X_real("X_real", ct.numst);
        X_real = dprojmatrices->dm();
        ReplicatedMatrix X_imag("X_imag", ct.numst);

// --- Diagnostic: pre-kick ground-state dipole moment ---
// Measures the dipole from the just-loaded, unperturbed ground-state DM,
// before updatePerturbationField/the time-stepping loop ever runs, so it's
// genuinely uncontaminated by the kick.
Vector3D mu_ground = computeDipoleMomentVector(orbitals, *dions, *dprojmatrices, false);
if (MPIdata::onpe0)
{
    std::cout << "Pre-kick ground-state dipole moment: ("
              << mu_ground[0] << ", " << mu_ground[1] << ", " << mu_ground[2]
              << ")" << std::endl;
}
// --- end diagnostic ---

        ReplicatedMatrix H_tot("H_tot", ct.numst);
        ReplicatedMatrix H_static("H_static", ct.numst);
        mgmol->computeHnl(&orbitals, H_static);

        double trace = 0., avg_energy = 0.;
        double trace_0 = X_real.trace();
        ReplicatedMatrix temp("temp", ct.numst);

        std::ofstream outputFile("outfile.txt");
        if (!outputFile.is_open())
        {
            std::cerr << "Error opening logging file!" << std::endl;
            return 1;
        }

        if (MPIdata::onpe0)
        {
            std::cout << "\nInitial State, t = 0.:" << std::endl;
            std::cout << "Initial trace: " << trace_0 << std::endl;
            std::cout << "Initial KS Energy: " << eks << std::endl;

            if (LOGGING)
            {
                outputFile << std::left
                           << std::setw(15) << "t" << ' '
                           << std::setw(12) << "CN_iters" << ' '
                           << std::setw(18) << "Tr(X)" << ' '
                           << std::setw(18) << "delta_Tr(X)" << ' '
                           << std::setw(18) << "E[XH]" << ' '
                           << std::setw(18) << "eks"
                           << '\n';
                outputFile << std::right;
            }
        }

        /* Initialize Ionic Stepper for MD */
//        std::shared_ptr<Ions> dions = mgmol->getIons();
        mgmol->mdInit(*dions);

        /* Setup initial Field (if any) and Hamiltonian before entering time loop */    
        /* Apply driving field, if any, before building H */
        mgmol->updatePerturbationField(t);
        /* Update Hamiltonian */
        mgmol->updateHFromHnl(&orbitals, H_static, H_tot);

        /* begin time-stepping */
        double tol = 1.0e-9;
        int maxits = 100;
        for (int step = 0; step < ct.num_MD_steps; step++)
        {
            t = step * dt;

            /* Propagate - Crank-Nicolson propagation */
            iters = propagate_density_matrix(X_real, X_imag, H_tot, dt, tol, maxits);
            avg_iters += iters;

            /* symmetrize to enforce Hermiticity */
            X_real.transpose(0.5, X_real, 0.5);
            X_imag.transpose(-0.5, X_imag, 0.5);

            dprojmatrices->setDM(X_real);

            /* Apply this step's driving field, if any, before rebuilding H */
            mgmol->updatePerturbationField((step+1) * dt);
            /* Update Hamiltonian */
            mgmol->updateHFromHnl(&orbitals, H_static, H_tot);

            /* update ion positions */
            mgmol->mdStep(orbitals, *dions);

            /* update total Kohn-Sham energy */
            eks = mgmol->finalEnergy(&orbitals);

            /* TDDFT: feed this step's KS energy into the stopping-power sanity-check trace */
            mgmol->recordStoppingEnergySample(eks);

            if (MPIdata::onpe0 && (step % PRINT_INTERVAL == 0))
            {
                trace = X_real.trace();
                double trace_diff = std::abs(trace - trace_0);

                temp.gemm('n', 'n', 1.0, X_real, H_tot, 0.);
                avg_energy = temp.trace();

                if (LOGGING)
                {
                    outputFile << std::scientific << std::setprecision(4)
                               << std::setw(5) << (t + dt) << ' '
                               << std::setw(10) << iters << ' ';
                    outputFile << std::scientific << std::setprecision(8)
                               << std::setw(18) << trace << ' '
                               << std::setw(18) << trace_diff << ' '
                               << std::setw(18) << avg_energy << ' '
                               << std::setw(18) << eks
                               << '\n';
                }
            }
        }

        mgmol->mdFinalize();

        /* Final summary */
        trace = X_real.trace();
        double trace_diff = std::abs(trace - trace_0);
        temp.gemm('n', 'n', 1.0, X_real, H_tot, 0.);
        avg_energy = temp.trace();

        eks = mgmol->finalEnergy(&orbitals);

        if (MPIdata::onpe0)
        {
            std::cout << "Final Statistics" << std::endl;
            std::cout << "\nPropagation complete: t = " << (ct.dt * ct.num_MD_steps)
                      << " a.u." << std::endl;
            std::cout << "Average CN_iters: " << (avg_iters / ct.num_MD_steps) << std::endl;
            std::cout << "Initial trace: " << trace_0 << std::endl;
            std::cout << "Final trace:   " << trace << std::endl;
            std::cout << "Total drift:   " << trace_diff << std::endl;
            std::cout << "E[XH] : " << avg_energy << std::endl;
            std::cout << "Final KS Energy : " << eks << std::endl;
            std::cout << "Forces :" << std::endl;
            for (auto it = forces.begin(); it != forces.end(); it += 3)
            {
                for (int i = 0; i < 3; i++) std::cout << "    " << *(it + i);
                std::cout << std::endl;
            }
        }

        mgmol->getAtomicPositions(positions);
        if (MPIdata::onpe0)
        {
            std::cout << "Positions:" << std::endl;
            auto ita = anumbers.begin();
            for (auto it = positions.begin(); it != positions.end(); it += 3)
            {
                std::cout << *ita;
                for (int i = 0; i < 3; i++) std::cout << "    " << *(it + i);
                std::cout << std::endl;
                ita++;
            }
        }

        outputFile.close();
        delete mgmol;
    }

    mgmol_finalize();

    mpirc = MPI_Finalize();
    if (mpirc != MPI_SUCCESS)
    {
        std::cerr << "MPI Finalize failed!!!" << std::endl;
    }

    time_t tt;
    time(&tt);
    if (onpe0) std::cout << " Run ended at " << ctime(&tt) << std::endl;

    return 0;
}

/* Crank-Nicolson evolution of the Density Matrix (real and imaginary parts) */
int propagate_density_matrix(ReplicatedMatrix& X_real, ReplicatedMatrix& X_imag,
    ReplicatedMatrix& H, double dt, double tol, int maxits)
{
    int iters = 0;
    double alpha = 0.5 * dt;
    double m_alpha = -alpha;
    double beta = 1.;
    double error_norm = 0.;
    double error_norm_real, error_norm_imag;

    ReplicatedMatrix comm_HX_r("comm_HX_r", H.m());
    ReplicatedMatrix comm_HX_i("comm_HX_i", H.m());

    ReplicatedMatrix work_r(X_real);
    ReplicatedMatrix work_i(X_imag);

    ReplicatedMatrix rhs_r(X_real);
    ReplicatedMatrix rhs_i(X_imag);
    rhs_r.gemm('n', 'n', alpha, H, X_imag, beta);
    rhs_i.gemm('n', 'n', alpha, X_real, H, beta);

    rhs_r.gemm('n', 'n', m_alpha, X_imag, H, beta);
    rhs_i.gemm('n', 'n', m_alpha, H, X_real, beta);

    for (iters = 0; iters < maxits; iters++)
    {
        compute_commutator(H, X_real, comm_HX_r);
        compute_commutator(H, X_imag, comm_HX_i);

        X_real = rhs_r;
        X_real.axpy(alpha, comm_HX_i);
        X_imag = rhs_i;
        X_imag.axpy(m_alpha, comm_HX_r);

        work_r -= X_real;
        work_i -= X_imag;
        error_norm_real = work_r.norm('F');
        error_norm_imag = work_i.norm('F');

        error_norm = std::sqrt(pow(error_norm_real, 2.) + pow(error_norm_imag, 2.));

        if (MPIdata::onpe0)
            std::cout << "iters = " << iters << ", error norm = " << error_norm << std::endl;

        if (error_norm < tol) break;

        work_r = X_real;
        work_i = X_imag;
    }
    return iters;
}

void compute_commutator(const ReplicatedMatrix& H, const ReplicatedMatrix& X,
    ReplicatedMatrix& op)
{
    op.clear();
    op.gemm('n', 'n', 1., H, X, 0.);
    op.gemm('n', 'n', -1., X, H, 1.);
}
// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "rom_workflows.h"
#include "Electrostatic.h"
#include <memory>
#include <stdexcept>
#include <string>

namespace CAROM
{

/*
    This is not implemented in libROM yet.
    A temporary placeholder until it is merged into libROM.
*/
void mult(const Matrix& A, const std::vector<int>& sample_row_A,
    const Matrix& B, Matrix& AB)
{
    CAROM_VERIFY(!B.distributed());
    CAROM_VERIFY(A.distributed() == AB.distributed());
    CAROM_VERIFY(A.numColumns() == B.numRows());
    CAROM_VERIFY(sample_row_A.size() <= A.numRows());

    const int num_rows = sample_row_A.size();
    const int num_cols = B.numColumns();
    AB.setSize(num_rows, num_cols);

    // Do the multiplication.
    const int Acol = A.numColumns();
    for (int r = 0; r < num_rows; ++r)
    {
        double* d_Arow = A.getData() + sample_row_A[r] * Acol;
        for (int c = 0; c < num_cols; ++c)
        {
            double* d_Aptr    = d_Arow;
            double* d_Bptr    = B.getData() + c;
            double result_val = 0.0;
            for (int ac = 0; ac < Acol; ++ac, d_Aptr++)
            {
                result_val += (*d_Aptr) * (*d_Bptr);
                d_Bptr += num_cols;
            }
            AB.item(r, c) = result_val;
        }
    }
}

}

template <typename... Args>
std::string string_format(const std::string& format, Args... args)
{
    int size_s = std::snprintf(nullptr, 0, format.c_str(), args...)
                 + 1; // Extra space for '\0'
    if (size_s <= 0)
    {
        throw std::runtime_error("Error during formatting.");
    }
    auto size = static_cast<size_t>(size_s);
    std::unique_ptr<char[]> buf(new char[size]);
    std::snprintf(buf.get(), size, format.c_str(), args...);
    return std::string(
        buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

template <class OrbitalsType>
void readRestartFiles(MGmolInterface* mgmol_)
{
    Control& ct              = *(Control::instance());
    Mesh* mymesh             = Mesh::instance();
    const pb::PEenv& myPEenv = mymesh->peenv();

    ROMPrivateOptions rom_options = ct.getROMOptions();
    /* type of variable we intend to run POD */
    ROMVariable rom_var = rom_options.variable;

    /* number of restart files, start/end indices */
    assert(rom_options.restart_file_minidx >= 0);
    assert(rom_options.restart_file_maxidx >= 0);
    const int minidx      = rom_options.restart_file_minidx;
    const int maxidx      = rom_options.restart_file_maxidx;
    const int num_restart = maxidx - minidx + 1;

    MGmol<OrbitalsType>* mgmol = static_cast<MGmol<OrbitalsType>*>(mgmol_);
    OrbitalsType* orbitals     = mgmol->getOrbitals();
    Potentials& pot            = mgmol->getHamiltonian()->potential();
    std::string filename;

    /* Determine basis prefix, dimension, and sample size */
    std::string basis_prefix = rom_options.basis_file;
    int dim;
    int totalSamples    = num_restart;
    const int chrom_num = orbitals->chromatic_number();
    switch (rom_var)
    {
        case ROMVariable::ORBITALS:
            basis_prefix += "_orbitals";
            dim = orbitals->getLocNumpt();
            /* if orbitals, each sample have chromatic number of wave functions
             */
            totalSamples *= orbitals->chromatic_number();
            break;

        case ROMVariable::POTENTIAL:
            basis_prefix += "_potential";
            dim = pot.size();
            break;

        default:
            ct.global_exit();
            break;
    }

    /* Initialize libROM classes */
    CAROM::Options svd_options(dim, totalSamples, 1);
    CAROM::BasisGenerator basis_generator(svd_options, false, basis_prefix);

    /* Collect the restart files */
    for (int k = minidx; k <= maxidx; k++)
    {
        filename = string_format(rom_options.restart_file_fmt, k);
        mgmol->loadRestartFile(filename);
        assert(dim == orbitals->getLocNumpt());
        assert(chrom_num == orbitals->chromatic_number());

        switch (rom_var)
        {
            case ROMVariable::ORBITALS:
                for (int i = 0; i < chrom_num; ++i)
                    basis_generator.takeSample(orbitals->getPsi(i));
                break;

            case ROMVariable::POTENTIAL:
                basis_prefix += "_potential";
                /* we save hartree potential */
                // TODO: consider create overloaded takeSample with const
                basis_generator.takeSample(
                    const_cast<double*>(pot.vh_rho().data()));
                break;
        }
    }
    basis_generator.writeSnapshot();
    basis_generator.endSamples();
}

template <class OrbitalsType>
void buildROMPoissonOperator(MGmolInterface* mgmol_)
{
    Control& ct              = *(Control::instance());
    Mesh* mymesh             = Mesh::instance();
    const pb::PEenv& myPEenv = mymesh->peenv();
    MGmol_MPI& mmpi          = *(MGmol_MPI::instance());
    const int rank           = mmpi.mypeGlobal();
    const int nprocs         = mmpi.size();

    ROMPrivateOptions rom_options = ct.getROMOptions();
    /* type of variable we intend to run POD */
    ROMVariable rom_var = rom_options.variable;
    if (rom_var != ROMVariable::POTENTIAL)
    {
        std::cerr << "buildROMPoissonOperator error: ROM variable must be "
                     "POTENTIAL to run this stage!\n"
                  << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 0);
    }

    /* Load Hartree potential basis matrix */
    std::string basis_file  = rom_options.basis_file;
    const int num_pot_basis = rom_options.num_potbasis;
    CAROM::BasisReader basis_reader(basis_file);
    CAROM::Matrix* pot_basis = basis_reader.getSpatialBasis(num_pot_basis);

    /* Load PoissonSolver pointer */
    MGmol<OrbitalsType>* mgmol = static_cast<MGmol<OrbitalsType>*>(mgmol_);
    Poisson* poisson           = mgmol->electrostat_->getPoissonSolver();

    /* GridFunc initialization inputs */
    const pb::Grid& grid(poisson->vh().grid());
    short bc[3];
    for (int d = 0; d < 3; d++)
        bc[d] = poisson->vh().bc(d);

    /* Initialize ROM matrix (undistributed) */
    CAROM::Matrix pot_rom(num_pot_basis, num_pot_basis, false);

    pb::GridFunc<POTDTYPE> col_gf(grid, bc[0], bc[1], bc[2]);
    pb::GridFunc<POTDTYPE> opcol_gf(grid, bc[0], bc[1], bc[2]);
    CAROM::Vector op_col(pot_basis->numRows(), true);
    CAROM::Vector rom_col(num_pot_basis, false);
    for (int c = 0; c < num_pot_basis; c++)
    {
        /* copy c-th column librom vector to GridFunc gf_col */
        CAROM::Vector* col = pot_basis->getColumn(c);
        col_gf.assign(col->getData(), 'd');

        /* apply Laplace operator */
        poisson->applyOperator(col_gf, opcol_gf);

        /* get librom view-vector of gf_opcol */
        opcol_gf.init_vect(op_col.getData(), 'd');

        /* Compute basis projection of the column */
        /* Resulting vector is undistributed */
        pot_basis->transposeMult(op_col, rom_col);

        /* libROM matrix is row-major, so data copy is necessary */
        for (int r = 0; r < num_pot_basis; r++)
            pot_rom(r, c) = rom_col(r);

        delete col;
    } // for (int c = 0; c < num_pot_basis; c++)

    /* DEIM hyperreduction */
    CAROM::Matrix pot_rhs_rom(num_pot_basis, num_pot_basis, false);
    std::vector<int> global_sampled_row(num_pot_basis),
        sampled_rows_per_proc(nprocs);
    DEIM(pot_basis, num_pot_basis, global_sampled_row, sampled_rows_per_proc,
        pot_rhs_rom, rank, nprocs);

    /* ROM rescaleTotalCharge operator */
    CAROM::Vector fom_ones(pot_basis->numRows(), true);
    CAROM::Vector rom_ones(num_pot_basis, false);
    fom_ones = mymesh->grid().vel(); // volume element
    pot_basis->transposeMult(fom_ones, rom_ones);

    /* Save ROM operator */
    // write the file from PE0 only
    if (MPIdata::onpe0)
    {
        std::string rom_oper = rom_options.pot_rom_file;
        CAROM::HDFDatabase h5_helper;
        h5_helper.create(rom_oper);
        h5_helper.putInteger("number_of_potential_basis", num_pot_basis);
        h5_helper.putDoubleArray("potential_rom_operator", pot_rom.getData(),
            num_pot_basis * num_pot_basis, false);

        /* save the inverse as well */
        pot_rom.inverse();
        h5_helper.putDoubleArray("potential_rom_inverse", pot_rom.getData(),
            num_pot_basis * num_pot_basis, false);

        /* save right-hand side hyper-reduction operator */
        h5_helper.putDoubleArray("potential_rhs_rom_inverse",
            pot_rhs_rom.getData(), num_pot_basis * num_pot_basis, false);

        /* save right-hand side rescaling operator */
        h5_helper.putDoubleArray(
            "potential_rhs_rescaler", rom_ones.getData(), num_pot_basis, false);

        h5_helper.close();
    }
}

template <class OrbitalsType>
void runPoissonROM(MGmolInterface* mgmol_)
{
    Control& ct              = *(Control::instance());
    Mesh* mymesh             = Mesh::instance();
    const pb::PEenv& myPEenv = mymesh->peenv();
    MGmol_MPI& mmpi          = *(MGmol_MPI::instance());
    const int rank           = mmpi.mypeGlobal();
    const int nprocs         = mmpi.size();

    ROMPrivateOptions rom_options = ct.getROMOptions();
    /* type of variable we intend to run POD */
    ROMVariable rom_var = rom_options.variable;
    if (rom_var != ROMVariable::POTENTIAL)
    {
        std::cerr << "runPoissonROM error: ROM variable must be POTENTIAL to "
                     "run this stage!\n"
                  << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 0);
    }

    /* Load Hartree potential basis matrix */
    std::string basis_file  = rom_options.basis_file;
    const int num_pot_basis = rom_options.num_potbasis;
    CAROM::BasisReader basis_reader(basis_file);
    CAROM::Matrix* pot_basis = basis_reader.getSpatialBasis(num_pot_basis);

    /* initialize rom operator variables */
    CAROM::Matrix pot_rom(num_pot_basis, num_pot_basis, false);
    CAROM::Matrix pot_rom_inv(num_pot_basis, num_pot_basis, false);
    CAROM::Matrix pot_rhs_rom(num_pot_basis, num_pot_basis, false);
    CAROM::Vector pot_rhs_rescaler(num_pot_basis, false);

    /* Load ROM operator */
    // read the file from PE0 only
    if (MPIdata::onpe0)
    {
        std::string rom_oper = rom_options.pot_rom_file;
        CAROM::HDFDatabase h5_helper;
        h5_helper.open(rom_oper, "r");
        int num_pot_basis_file = -1;
        h5_helper.getInteger("number_of_potential_basis", num_pot_basis_file);
        CAROM_VERIFY(num_pot_basis_file == num_pot_basis);

        h5_helper.getDoubleArray("potential_rom_operator", pot_rom.getData(),
            num_pot_basis * num_pot_basis, false);

        /* load the inverse as well */
        h5_helper.getDoubleArray("potential_rom_inverse", pot_rom_inv.getData(),
            num_pot_basis * num_pot_basis, false);

        /* load right-hand side hyper-reduction operator */
        h5_helper.getDoubleArray("potential_rhs_rom_inverse",
            pot_rhs_rom.getData(), num_pot_basis * num_pot_basis, false);

        /* load right-hand side rescaling operator */
        h5_helper.getDoubleArray("potential_rhs_rescaler",
            pot_rhs_rescaler.getData(), num_pot_basis, false);

        h5_helper.close();
    }
}

/* test routines */

template <class OrbitalsType>
void testROMPoissonOperator(MGmolInterface* mgmol_)
{
    Control& ct              = *(Control::instance());
    Mesh* mymesh             = Mesh::instance();
    const pb::PEenv& myPEenv = mymesh->peenv();

    ROMPrivateOptions rom_options = ct.getROMOptions();

    /* Load MGmol pointer and Potentials */
    MGmol<OrbitalsType>* mgmol = static_cast<MGmol<OrbitalsType>*>(mgmol_);
    Poisson* poisson           = mgmol->electrostat_->getPoissonSolver();
    Potentials& pot            = mgmol->getHamiltonian()->potential();
    const int dim              = pot.size();
    printf("pot size: %d\n", dim);

    /* GridFunc initialization inputs */
    const pb::Grid& grid(poisson->vh().grid());
    short bc[3];
    for (int d = 0; d < 3; d++)
        bc[d] = poisson->vh().bc(d);

    /* fictitious snapshot numbers */
    const int nsnapshot = 3;

    /* Set compensating charges to zero now */
    pb::GridFunc<POTDTYPE> rhoc(grid, bc[0], bc[1], bc[2]);
    rhoc = 0.0;

    /* Generate fictitious right-hand sides and snapshots */
    std::vector<std::vector<POTDTYPE>> rhs(nsnapshot), fom_sol(nsnapshot);
    for (int s = 0; s < nsnapshot; s++)
    {
        rhs[s].resize(dim);
        for (int d = 0; d < dim; d++)
            rhs[s][d] = ran0();

        /* average out for periodic bc */
        pb::GridFunc<POTDTYPE> rhs_gf(grid, bc[0], bc[1], bc[2]);
        rhs_gf.assign(rhs[s].data(), 'd');
        double avg = rhs_gf.get_average();
        rhs_gf -= avg;

        /* copy back to rhs */
        rhs_gf.init_vect(rhs[s].data(), 'd');

        poisson->solve(rhs_gf, rhoc);

        fom_sol[s].resize(dim);
        poisson->vh().init_vect(fom_sol[s].data(), 'd');

        /* check if the solution is correct */
        pb::GridFunc<POTDTYPE> res(grid, bc[0], bc[1], bc[2]);
        pb::GridFunc<POTDTYPE> sol_gf(grid, bc[0], bc[1], bc[2]);
        sol_gf.assign(fom_sol[s].data());
        /* apply Laplace operator */
        poisson->applyOperator(sol_gf, res);
        /* FD operator scales rhs by 4pi */
        res.axpy(-4. * M_PI, rhs_gf);
        printf("FOM res norm: %.3e\n", res.norm2());
    }

    /* Initialize libROM classes */
    std::string basis_prefix = "test_poisson";
    CAROM::Options svd_options(dim, nsnapshot, 1);
    CAROM::BasisGenerator basis_generator(svd_options, false, basis_prefix);

    /* Collect snapshots and train POD basis */
    for (int s = 0; s < nsnapshot; s++)
        basis_generator.takeSample(fom_sol[s].data());
    basis_generator.endSamples();

    /* Load POD basis. We use the maximum number of basis vectors. */
    const CAROM::Matrix* pot_basis = basis_generator.getSpatialBasis();

    /* Check if full projection preserves FOM solution */
    for (int c = 0; c < nsnapshot; c++)
    {
        CAROM::Vector* fom_sol_vec = nullptr;
        /* get librom view-vector of fom_sol */
        fom_sol_vec = new CAROM::Vector(
            fom_sol[c].data(), pot_basis->numRows(), true, false);

        CAROM::Vector* rom_proj    = pot_basis->transposeMult(*fom_sol_vec);
        CAROM::Vector* reconstruct = pot_basis->mult(*rom_proj);

        /* error on libROM side */
        CAROM::Vector* librom_error = reconstruct->minus(fom_sol_vec);
        printf("librom reconstruction error: %.3e\n", librom_error->norm());

        /* error on mgmol side */
        pb::GridFunc<POTDTYPE> recon_gf(grid, bc[0], bc[1], bc[2]);
        pb::GridFunc<POTDTYPE> fom_gf(grid, bc[0], bc[1], bc[2]);
        recon_gf.assign(reconstruct->getData(), 'd');
        fom_gf.assign(fom_sol[c].data(), 'd');
        recon_gf -= fom_gf;
        printf("mgmol reconstruction error: %.3e\n", recon_gf.norm2());

        delete fom_sol_vec;
        delete rom_proj;
        delete reconstruct;
        delete librom_error;
    }

    /* Check FOM axpy is equivalent to ROM axpy */
    for (int s = 0; s < nsnapshot; s++)
    {
        CAROM::Vector fom_res(pot_basis->numRows(), true);
        CAROM::Vector rom_res(nsnapshot, false);
        CAROM::Vector fom_rhs(pot_basis->numRows(), true);
        CAROM::Vector rom_rhs(nsnapshot, false);

        pb::GridFunc<POTDTYPE> res(grid, bc[0], bc[1], bc[2]);
        pb::GridFunc<POTDTYPE> fomsol_gf(grid, bc[0], bc[1], bc[2]);
        pb::GridFunc<POTDTYPE> mgmol_rhs(grid, bc[0], bc[1], bc[2]);
        fomsol_gf.assign(fom_sol[s].data(), 'd');
        mgmol_rhs.assign(rhs[s].data(), 'd');

        /* apply Laplace operator */
        poisson->applyOperator(fomsol_gf, res);

        /* get librom view-vector of fom_res */
        res.init_vect(fom_res.getData(), 'd');
        pot_basis->transposeMult(fom_res, rom_res);

        /* get librom view-vector of fom_rhs */
        mgmol_rhs.init_vect(fom_rhs.getData(), 'd');
        pot_basis->transposeMult(fom_rhs, rom_rhs);

        /* ROM residual: FD operator scales rhs by 4pi */
        rom_rhs *= 4. * M_PI;
        rom_res -= rom_rhs;
        printf("ROM res norm: %.3e\n", rom_res.norm());

        /* FOM residual: FD operator scales rhs by 4pi */
        res.axpy(-4. * M_PI, mgmol_rhs);
        printf("FOM res norm: %.3e\n", res.norm2());

        /* projection of the residual */
        res.init_vect(fom_res.getData(), 'd');
        CAROM::Vector* res_proj = pot_basis->transposeMult(fom_res);
        printf("FOM res projection norm: %.3e\n", res_proj->norm());

        delete res_proj;
    }

    /* Initialize Projection ROM matrix (undistributed) */
    CAROM::Matrix pot_rom(nsnapshot, nsnapshot, false);

    /* Build Projection of Poisson operator */
    for (int c = 0; c < nsnapshot; c++)
    {
        pb::GridFunc<POTDTYPE> col_gf(grid, bc[0], bc[1], bc[2]);
        pb::GridFunc<POTDTYPE> opcol_gf(grid, bc[0], bc[1], bc[2]);
        CAROM::Vector op_col(pot_basis->numRows(), true);

        /* copy c-th column librom vector to GridFunc gf_col */
        CAROM::Vector* col = pot_basis->getColumn(c);
        col_gf.assign(col->getData(), 'd');

        /* apply Laplace operator */
        poisson->applyOperator(col_gf, opcol_gf);

        /* get librom view-vector of gf_opcol */
        opcol_gf.init_vect(op_col.getData(), 'd');

        /* Compute basis projection of the column */
        /* Resulting vector is undistributed */
        CAROM::Vector* rom_col = pot_basis->transposeMult(op_col);

        /* libROM matrix is row-major, so data copy is necessary */
        for (int r = 0; r < nsnapshot; r++)
            pot_rom(r, c) = (*rom_col)(r);

        delete col;
        delete rom_col;
    } // for (int c = 0; c < num_pot_basis; c++)

    /* Inverse of the projection ROM matrix */
    CAROM::Matrix pot_rom_inv(pot_rom);
    pot_rom_inv.inverse();

    /* Check the inverse */
    CAROM::Matrix* identity = pot_rom_inv.mult(pot_rom);
    printf("pot_rom_inv * pot_rom = identity\n");
    for (int i = 0; i < nsnapshot; i++)
    {
        for (int j = 0; j < nsnapshot; j++)
            printf("%.3e\t", identity->item(i, j));
        printf("\n");
    }
    delete identity;

    /* Test with sample RHS. ROM must be able to 100% reproduce the FOM
     * solution. */
    std::vector<CAROM::Vector*> rom_sol(0), rom_rhs(0);
    std::vector<std::vector<POTDTYPE>> test_sol(nsnapshot);
    for (int s = 0; s < nsnapshot; s++)
    {
        /* get librom view-vector of rhs[s] */
        CAROM::Vector fom_rhs(rhs[s].data(), dim, true, false);

        /* project onto POD basis */
        rom_rhs.push_back(pot_basis->transposeMult(fom_rhs));

        /* FOM FD operator scales rhs by 4pi */
        *rom_rhs.back() *= 4. * M_PI;

        /* solve ROM */
        rom_sol.push_back(pot_rom_inv.mult(*rom_rhs.back()));

        /* check ROM solution */
        CAROM::Vector& res(*pot_rom.mult(*rom_sol.back()));
        res -= *rom_rhs.back();
        printf("rom res norm: %.3e\n", res.norm());

        /* initialize lift-up FOM solution */
        test_sol[s].resize(dim);
        /* get librom view-vector of test_sol[s] */
        CAROM::Vector test_sol_vec(test_sol[s].data(), dim, true, false);
        pot_basis->mult(*rom_sol.back(), test_sol_vec);
    }

    /* Compute relative errors */
    for (int s = 0; s < nsnapshot; s++)
    {
        pb::GridFunc<POTDTYPE> testsol_gf(grid, bc[0], bc[1], bc[2]);
        pb::GridFunc<POTDTYPE> fomsol_gf(grid, bc[0], bc[1], bc[2]);

        testsol_gf.assign(test_sol[s].data(), 'd');
        fomsol_gf.assign(fom_sol[s].data(), 'd');

        testsol_gf -= fomsol_gf;
        double rel_error = testsol_gf.norm2() / fomsol_gf.norm2();
        printf("%d-th sample relative error: %.3e\n", s, rel_error);

        if (rel_error > 1.0e-9) abort();
    }

    /* clean up pointers */
    for (int s = 0; s < nsnapshot; s++)
    {
        delete rom_sol[s];
        delete rom_rhs[s];
    }
}

template <class OrbitalsType>
void testROMRhoOperator(MGmolInterface* mgmol_)
{
    Control& ct              = *(Control::instance());
    Mesh* mymesh             = Mesh::instance();
    const int subdivx        = mymesh->subdivx();
    const pb::PEenv& myPEenv = mymesh->peenv();
    MGmol_MPI& mmpi          = *(MGmol_MPI::instance());
    const int rank           = mmpi.mypeGlobal();
    const int nprocs         = mmpi.size();

    // if (ct.isLocMode())
    //     printf("LocMode is On!\n");
    // else
    //     printf("LocMode is Off!\n");

    ROMPrivateOptions rom_options = ct.getROMOptions();

    /* Load MGmol pointer and Potentials */
    MGmol<OrbitalsType>* mgmol = static_cast<MGmol<OrbitalsType>*>(mgmol_);
    Poisson* poisson           = mgmol->electrostat_->getPoissonSolver();
    Potentials& pot            = mgmol->getHamiltonian()->potential();
    std::shared_ptr<Rho<OrbitalsType>> rho = NULL; // mgmol->getRho();
    const OrthoType ortho_type             = rho->getOrthoType();
    assert(ortho_type == OrthoType::Nonorthogonal);

    /* potential should have the same size as rho */
    const int dim = pot.size();

    /* number of restart files, start/end indices */
    assert(rom_options.restart_file_minidx >= 0);
    assert(rom_options.restart_file_maxidx >= 0);
    const int minidx      = rom_options.restart_file_minidx;
    const int maxidx      = rom_options.restart_file_maxidx;
    const int num_restart = maxidx - minidx + 1;

    /* Initialize libROM classes */
    /*
        In practice, we do not produce rho POD basis.
        This rho POD basis is for the sake of verification.
    */
    CAROM::Options svd_options(dim, num_restart, 1);
    svd_options.static_svd_preserve_snapshot = true;
    CAROM::BasisGenerator basis_generator(svd_options, false, "test_rho");

    /* Collect the restart files */
    std::string filename;
    for (int k = minidx; k <= maxidx; k++)
    {
        filename = string_format(rom_options.restart_file_fmt, k);
        mgmol->loadRestartFile(filename);
        basis_generator.takeSample(&rho->rho_[0][0]);
    }
    // basis_generator.writeSnapshot();
    const CAROM::Matrix rho_snapshots(*basis_generator.getSnapshotMatrix());
    basis_generator.endSamples();

    const CAROM::Matrix* rho_basis = basis_generator.getSpatialBasis();
    CAROM::Matrix* proj_rho        = rho_basis->transposeMult(rho_snapshots);

    /* DEIM hyperreduction */
    CAROM::Matrix rho_basis_inv(num_restart, num_restart, false);
    std::vector<int> global_sampled_row(num_restart),
        sampled_rows_per_proc(nprocs);
    DEIM(rho_basis, num_restart, global_sampled_row, sampled_rows_per_proc,
        rho_basis_inv, rank, nprocs);
    if (rank == 0)
    {
        int num_sample_rows = 0;
        for (int k = 0; k < sampled_rows_per_proc.size(); k++)
            num_sample_rows += sampled_rows_per_proc[k];
        printf("number of sampled row: %d\n", num_sample_rows);
    }

    /* get local sampled row */
    std::vector<int> offsets, sampled_row(sampled_rows_per_proc[rank]);
    int num_global_sample
        = CAROM::get_global_offsets(sampled_rows_per_proc[rank], offsets);
    for (int s = 0, gs = offsets[rank]; gs < offsets[rank + 1]; gs++, s++)
        sampled_row[s] = global_sampled_row[gs];

    /* load only the first restart file for now */
    const int test_idx = 2;

    filename = string_format(rom_options.restart_file_fmt, test_idx + minidx);
    /*
        currently, this does not update rho.
        computeRhoOnSamplePts computes with the new density matrix,
        while mgmol rho remains the same as the initial condition.
        Commenting line 516 gives a consistent result, both for the initial
       condition.
    */
    mgmol->loadRestartFile(filename);

    const int nrows = mymesh->locNumpt();
    // printf("mesh::locNumpt: %d\n", nrows);

    OrbitalsType* orbitals = mgmol->getOrbitals();
    // printf("orbitals::locNumpt: %d\n", orbitals->getLocNumpt());

    /* NOTE(kevin): we assume we only use ProjectedMatrices class */
    // ProjectedMatrices<dist_matrix::DistMatrix<double>> *proj_matrices =
    //     static_cast<ProjectedMatrices<dist_matrix::DistMatrix<double>>
    //     *>(orbitals->getProjMatrices());
    ProjectedMatricesInterface* proj_matrices = orbitals->getProjMatrices();
    proj_matrices->updateSubMatX();
    SquareLocalMatrices<MATDTYPE, memory_space_type>& localX(
        proj_matrices->getLocalX());

    // printf("localX nmat: %d\n", localX.nmat());
    // printf("localX n: %d\n", localX.n());
    // printf("localX m: %d\n", localX.m());

    bool dm_distributed = (localX.nmat() > 1);
    assert(!dm_distributed);

    /* copy density matrix */
    CAROM::Matrix dm(
        localX.getRawPtr(), localX.m(), localX.n(), dm_distributed, true);

    // /* random density matrix */
    // /* NOTE(kevin): Due to rescaleTotalCharge, the result is slightly
    // inconsistent. */ CAROM::Matrix dm(localX.m(), localX.n(), dm_distributed,
    // true); dm = 0.0; dist_matrix::DistMatrix<double>
    // dm_mgmol(proj_matrices->dm()); for (int i = 0; i < dm_mgmol.m(); i++)
    // {
    //     for (int j = 0; j < dm_mgmol.m(); j++)
    //     {
    //         if (dm_mgmol.getVal(i, j) == 0.0)
    //             continue;

    //         dm(i, j) = dm_mgmol.getVal(i, j) * (0.975 + 0.05 * ran0());
    //         dm_mgmol.setVal(i, j, dm(i, j));
    //     }
    // }

    /* update rho first */
    // rho->computeRho(*orbitals, dm_mgmol);
    // rho->update(*orbitals);

    const int chrom_num = orbitals->chromatic_number();
    CAROM::Matrix psi(dim, chrom_num, true);
    for (int c = 0; c < chrom_num; c++)
    {
        ORBDTYPE* d_psi = orbitals->getPsi(c);
        for (int d = 0; d < dim; d++)
            psi.item(d, c) = *(d_psi + d);
    }

    CAROM::Matrix rom_psi(chrom_num, chrom_num, false);
    for (int i = 0; i < chrom_num; i++)
        for (int j = 0; j < chrom_num; j++)
            rom_psi(i, j) = (i == j) ? 1 : 0;

    /* this will be resized in computeRhoOnSamplePts */
    CAROM::Vector sample_rho(1, true);

    computeRhoOnSamplePts(dm, psi, rom_psi, sampled_row, sample_rho);

    for (int s = 0; s < sampled_row.size(); s++)
    {
        const double error = abs(rho->rho_[0][sampled_row[s]] - sample_rho(s));
        if (error > 1.0e-4)
            printf("rank %d, rho[%d]: %.5e, sample_rho: %.5e, librom_snapshot: "
                   "%.5e\n",
                rank, sampled_row[s], rho->rho_[0][sampled_row[s]],
                sample_rho(s), rho_snapshots(sampled_row[s], test_idx));
        CAROM_VERIFY(error < 1.0e-4);
    }

    sample_rho.gather();

    CAROM::Vector* rom_rho = rho_basis_inv.mult(sample_rho);
    for (int d = 0; d < rom_rho->dim(); d++)
    {
        if ((rank == 0)
            && (abs(proj_rho->item(d, test_idx) - rom_rho->item(d)) > 1.0e-3))
            printf("rom_rho error: %.3e\n",
                abs(proj_rho->item(d, test_idx) - rom_rho->item(d)));
        CAROM_VERIFY(
            abs(proj_rho->item(d, test_idx) - rom_rho->item(d)) < 1.0e-3);
    }

    CAROM::Vector* fom_rho = rho_basis->mult(*rom_rho);

    CAROM_VERIFY(fom_rho->dim() == rho->rho_[0].size());
    for (int d = 0; d < fom_rho->dim(); d++)
        CAROM_VERIFY(abs(fom_rho->item(d) - rho->rho_[0][d]) < 1.0e-4);

    delete rom_rho;
    delete fom_rho;
}

/*
    dm: density matrix converted to CAROM::Matrix
    phi_basis: POD basis matrix of orbitals, or orbitals themselves
    rom_phi: ROM coefficients of POD Basis. If phi_basis is orbitals themselves,
   then simply an identity. local_idx: sampled local grid indices on this
   processor. sampled_rho: FOM density on sampled grid points.
*/
void computeRhoOnSamplePts(const CAROM::Matrix& dm,
    const CAROM::Matrix& phi_basis, const CAROM::Matrix& rom_phi,
    const std::vector<int>& local_idx, CAROM::Vector& sampled_rho)
{
    assert(!dm.distributed());
    assert(sampled_rho.distributed() == phi_basis.distributed());

    // this will be resized.
    CAROM::Matrix sampled_phi(1, 1, phi_basis.distributed());

    CAROM::mult(phi_basis, local_idx, rom_phi, sampled_phi);

    /* same product as in computeRhoSubdomainUsingBlas3 */
    CAROM::Matrix product(1, 1, sampled_phi.distributed());
    sampled_phi.mult(dm, product);

    sampled_rho.setSize(sampled_phi.numRows());
    double* d_product = product.getData();
    double* d_phi     = sampled_phi.getData();
    for (int d = 0; d < sampled_rho.dim(); d++)
    {
        double val = 0.0;
        /* CAROM Matrices are row-major */
        for (int c = 0; c < sampled_phi.numColumns(); c++, d_product++, d_phi++)
            val += (*d_product) * (*d_phi);

        sampled_rho(d) = val;
    }

    /*
        TODO(kevin): need to figure out what these functions do.
        and probably need to make ROM-equivalent functions with another
       hyper-reduction?
    */
    // gatherSpin();

    /*
        rescaleTotalCharge is handled after hyperreduction.
    */
    // rescaleTotalCharge();
}

template <class OrbitalsType>
void testROMIonDensity(MGmolInterface* mgmol_)
{
    /* random number generator */
    static std::random_device
        rd; // Will be used to obtain a seed for the random number engine
    static std::mt19937 gen(
        rd()); // Standard mersenne_twister_engine seeded with rd(){}
    static std::uniform_real_distribution<> dis(0.0, 1.0);

    Control& ct              = *(Control::instance());
    Mesh* mymesh             = Mesh::instance();
    const pb::Grid& mygrid   = mymesh->grid();
    const int subdivx        = mymesh->subdivx();
    const pb::PEenv& myPEenv = mymesh->peenv();
    MGmol_MPI& mmpi          = *(MGmol_MPI::instance());
    const int rank           = mmpi.mypeGlobal();
    const int nprocs         = mmpi.size();

    ROMPrivateOptions rom_options = ct.getROMOptions();

    /* Load MGmol pointer and Potentials */
    MGmol<OrbitalsType>* mgmol = static_cast<MGmol<OrbitalsType>*>(mgmol_);
    Poisson* poisson           = mgmol->electrostat_->getPoissonSolver();
    Potentials& pot            = mgmol->getHamiltonian()->potential();
    const int dim              = pot.size();
    std::shared_ptr<Ions> ions = mgmol->getIons();

    /* get the extent of global domain */
    const double origin[3]
        = { mygrid.origin(0), mygrid.origin(1), mygrid.origin(2) };
    const double lattice[3] = { mygrid.ll(0), mygrid.ll(1), mygrid.ll(2) };
    if (rank == 0)
    {
        printf("origin: (%.3e, %.3e, %.3e)\n", origin[0], origin[1], origin[2]);
        printf("lattice: (%.3e, %.3e, %.3e)\n", lattice[0], lattice[1],
            lattice[2]);
    }

    /* get global atomic numbers */
    const int num_ions = ions->getNumIons();
    std::vector<short> atnumbers(num_ions);
    ions->getAtomicNumbers(atnumbers);

    assert(!(mgmol->electrostat_->isDielectric()));
    assert(pot.getBackgroundCharge() <= 0.0);

    const int num_snap = 3;

    /* 3 fictitious ion configurations */
    std::vector<std::vector<double>> cfgs(num_snap);
    for (int idx = 0; idx < num_snap; idx++)
    {
        cfgs[idx].resize(3 * num_ions);
        if (rank == 0)
            for (int k = 0; k < num_ions; k++)
                for (int d = 0; d < 3; d++)
                    cfgs[idx][3 * k + d] = origin[d] + lattice[d] * dis(gen);

        mmpi.bcastGlobal(cfgs[idx].data(), 3 * num_ions, 0);
    }

    /* Artificial ions object to avoid repeated setPositions */
    Ions* new_ions;
    /* Collect fictitious ion density based on each configuration */
    std::vector<std::vector<POTDTYPE>> fom_rhoc(num_snap);
    /* Sanity check for overlappingVL_ions */
    std::vector<std::vector<std::vector<double>>> fom_overlap_ions(num_snap);
    for (int idx = 0; idx < num_snap; idx++)
    {
        /* set ion positions */
        new_ions = new Ions(cfgs[idx], atnumbers, lattice, ions->getSpecies());

        /* save overlapping ions for sanity check */
        fom_overlap_ions[idx].resize(new_ions->overlappingVL_ions().size());
        for (int k = 0; k < new_ions->overlappingVL_ions().size(); k++)
        {
            fom_overlap_ions[idx][k].resize(3);
            for (int d = 0; d < 3; d++)
                fom_overlap_ions[idx][k][d]
                    = new_ions->overlappingVL_ions()[k]->position(d);
        }

        /* compute resulting ion density */
        /* NOTE: we exclude rescaling for the sake of verification */
        mgmol->setupPotentials(*new_ions);

        fom_rhoc[idx].resize(dim);
        mgmol->electrostat_->getRhoc()->init_vect(fom_rhoc[idx].data(), 'd');

        delete new_ions;
    }

    /* Initialize libROM classes */
    /*
        In practice, we do not produce rhoc POD basis.
        This rhoc POD basis is for the sake of verification.
    */
    CAROM::Options svd_options(dim, num_snap, 1);
    svd_options.static_svd_preserve_snapshot = true;
    CAROM::BasisGenerator basis_generator(svd_options, false, "test_rhoc");
    for (int idx = 0; idx < num_snap; idx++)
        basis_generator.takeSample(&fom_rhoc[idx][0]);

    const CAROM::Matrix rhoc_snapshots(*basis_generator.getSnapshotMatrix());
    basis_generator.endSamples();

    const CAROM::Matrix* rhoc_basis = basis_generator.getSpatialBasis();
    CAROM::Matrix* proj_rhoc        = rhoc_basis->transposeMult(rhoc_snapshots);

    /* DEIM hyperreduction */
    CAROM::Matrix rhoc_basis_inv(num_snap, num_snap, false);
    std::vector<int> global_sampled_row(num_snap),
        sampled_rows_per_proc(nprocs);
    DEIM(rhoc_basis, num_snap, global_sampled_row, sampled_rows_per_proc,
        rhoc_basis_inv, rank, nprocs);
    if (rank == 0)
    {
        int num_sample_rows = 0;
        for (int k = 0; k < sampled_rows_per_proc.size(); k++)
            num_sample_rows += sampled_rows_per_proc[k];
        printf("number of sampled row: %d\n", num_sample_rows);
    }

    /* get local sampled row */
    std::vector<int> offsets, sampled_row(sampled_rows_per_proc[rank]);
    int num_global_sample
        = CAROM::get_global_offsets(sampled_rows_per_proc[rank], offsets);
    for (int s = 0, gs = offsets[rank]; gs < offsets[rank + 1]; gs++, s++)
        sampled_row[s] = global_sampled_row[gs];

    /* test one solution */
    std::uniform_int_distribution<> distrib(0, num_snap - 1);
    int test_idx = distrib(gen);
    mmpi.bcastGlobal(&test_idx);
    if (rank == 0) printf("test index: %d\n", test_idx);

    /* set ion positions */
    new_ions = new Ions(cfgs[test_idx], atnumbers, lattice, ions->getSpecies());

    /* Sanity check for overlapping ions */
    CAROM_VERIFY(fom_overlap_ions[test_idx].size()
                 == new_ions->overlappingVL_ions().size());
    for (int k = 0; k < new_ions->overlappingVL_ions().size(); k++)
        for (int d = 0; d < 3; d++)
            CAROM_VERIFY(abs(fom_overlap_ions[test_idx][k][d]
                             - new_ions->overlappingVL_ions()[k]->position(d))
                         < 1.0e-12);

    /* set up potentials */
    mgmol->setupPotentials(*new_ions);

    /* eval ion density on sample grid points */
    std::vector<RHODTYPE> sampled_rhoc(sampled_row.size());
    pot.evalIonDensityOnSamplePts(*new_ions, sampled_row, sampled_rhoc);

    // For now, we relax the threshold to allow the slight difference of the
    // values at the sampled indices. This is because the rescaling procedure
    // after getting the radial data on mesh requires global information, thus
    // cannot be done in the ROM level only with sampled values. After we have
    // the DEIM reconstruction, we will do rescaling and revisit this.
    for (int d = 0; d < sampled_row.size(); d++)
    {
        printf("rank %d, fom rhoc[%d]: %.3e, rom rhoc: %.3e\n", rank,
            sampled_row[d], fom_rhoc[test_idx][sampled_row[d]],
            sampled_rhoc[d]);
        CAROM_VERIFY(
            abs(fom_rhoc[test_idx][sampled_row[d]] - sampled_rhoc[d]) < 1.0e-2);
    }

    delete new_ions;
}

template void readRestartFiles<LocGridOrbitals<ORBDTYPE>>(
    MGmolInterface* mgmol_);
template void readRestartFiles<ExtendedGridOrbitals<ORBDTYPE>>(
    MGmolInterface* mgmol_);

template void buildROMPoissonOperator<LocGridOrbitals<ORBDTYPE>>(
    MGmolInterface* mgmol_);
template void buildROMPoissonOperator<ExtendedGridOrbitals<ORBDTYPE>>(
    MGmolInterface* mgmol_);

template void runPoissonROM<LocGridOrbitals<ORBDTYPE>>(MGmolInterface* mgmol_);
template void runPoissonROM<ExtendedGridOrbitals<ORBDTYPE>>(
    MGmolInterface* mgmol_);

template void testROMPoissonOperator<LocGridOrbitals<ORBDTYPE>>(
    MGmolInterface* mgmol_);
template void testROMPoissonOperator<ExtendedGridOrbitals<ORBDTYPE>>(
    MGmolInterface* mgmol_);

template void testROMRhoOperator<LocGridOrbitals<ORBDTYPE>>(
    MGmolInterface* mgmol_);
template void testROMRhoOperator<ExtendedGridOrbitals<ORBDTYPE>>(
    MGmolInterface* mgmol_);

template void testROMIonDensity<LocGridOrbitals<ORBDTYPE>>(
    MGmolInterface* mgmol_);
template void testROMIonDensity<ExtendedGridOrbitals<ORBDTYPE>>(
    MGmolInterface* mgmol_);

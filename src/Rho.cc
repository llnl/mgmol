// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "Rho.h"
#include "Control.h"
#include "ExtendedGridOrbitals.h"
#include "LocGridOrbitals.h"
#include "MPIdata.h"
#include "Mesh.h"
#include "ProjectedMatrices.h"
#include "ReplicatedMatrix.h"
#include "SquareLocalMatrices.h"
#include "magma_singleton.h"
#include "memory_space.h"
#include "mputils.h"
#include "numerical_kernels.h"

#ifdef MGMOL_USE_SCALAPACK
#include "SubMatrices.h"
#endif

template <class OrbitalsType>
Timer Rho<OrbitalsType>::update_tm_("Rho::update");
template <class OrbitalsType>
Timer Rho<OrbitalsType>::compute_tm_("Rho::compute");
template <class OrbitalsType>
Timer Rho<OrbitalsType>::compute_blas_tm_("Rho::compute_usingBlas");

#ifdef HAVE_MAGMA
template <typename ScalarType>
using MemoryDev = MemorySpace::Memory<ScalarType, MemorySpace::Device>;
#endif

template <class OrbitalsType>
Rho<OrbitalsType>::Rho()
    : orbitals_type_(OrthoType::UNDEFINED),
      iterative_index_(-10),
      verbosity_level_(0) // default value
{
    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    myspin_         = mmpi.myspin();
    nspin_          = mmpi.nspin();

    Mesh* mymesh           = Mesh::instance();
    const pb::Grid& mygrid = mymesh->grid();
    np_                    = mygrid.size();

    rho_.resize(nspin_);
    for (short is = 0; is < nspin_; is++)
    {
        rho_[is].resize(np_);
        memset(&rho_[is][0], 0, np_ * sizeof(RHODTYPE));
    }

    assert(nspin_ == 1 || nspin_ == 2);
    assert(myspin_ == 0 || myspin_ == 1);
}

template <class OrbitalsType>
void Rho<OrbitalsType>::setup(const OrthoType orbitals_type,
    const std::vector<std::vector<int>>& orbitals_indexes)
{
    if (verbosity_level_ > 2 && onpe0)
        (*MPIdata::sout) << " Rho<OrbitalsType>::setup()" << std::endl;

    orbitals_type_ = orbitals_type;

    orbitals_indexes_ = orbitals_indexes;

    iterative_index_ = -1;
}

template <class OrbitalsType>
void Rho<OrbitalsType>::axpyRhoc(const double alpha, RHODTYPE* rhoc)
{
    double factor = (nspin_ > 1) ? 0.5 * alpha : alpha;
    LinearAlgebraUtils<MemorySpace::Host>::MPaxpy(
        np_, factor, &rhoc[0], &rho_[myspin_][0]);
}

template <class OrbitalsType>
void Rho<OrbitalsType>::update(OrbitalsType& current_orbitals)
{
    const ProjectedMatricesInterface& proj_matrices(
        *(current_orbitals.getProjMatrices()));

    assert(current_orbitals.getIterativeIndex() >= 0);
    assert(proj_matrices.getDMMatrixIndex() >= 0);

    update_tm_.start();

    if (verbosity_level_ > 1 && onpe0)
        (*MPIdata::sout) << "Rho::update()..." << std::endl;

    iterative_index_ = ((1 + current_orbitals.getIterativeIndex()) % 100)
                       + (proj_matrices.getDMMatrixIndex() % 100) * 100;

#ifdef PRINT_OPERATIONS
    if (onpe0)
        (*MPIdata::sout) << "Rho<OrbitalsType>::update(), iterative_index_="
                         << iterative_index_ << std::endl;
#endif

    computeRho(current_orbitals);

    assert(iterative_index_ >= 0);

    update_tm_.stop();
}

// note: rho can be negative because of added background charge
template <class OrbitalsType>
double Rho<OrbitalsType>::computeTotalCharge()
{
    const int nspin = (int)rho_.size();

    double tcharge = 0.;
    for (short ispin = 0; ispin < nspin; ispin++)
    {
        const RHODTYPE* const prho = &rho_[ispin][0];
        for (int idx = 0; idx < np_; idx++)
        {
            assert(prho[idx] < 1.e6);
            tcharge += (double)prho[idx];
        }
    }

    // reduce over spin communicator since charge from other spin is already
    // included
    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    mmpi.allreduce(&tcharge, 1, MPI_SUM);

    Mesh* mymesh           = Mesh::instance();
    const pb::Grid& mygrid = mymesh->grid();
    tcharge *= mygrid.vel();
#ifdef DEBUG
    if (mmpi.instancePE0())
        (*MPIdata::sout) << fixed << setprecision(8) << " myspin=" << myspin_
                         << ", Total charge = " << tcharge << std::endl;
#endif

    return tcharge;
}

template <class OrbitalsType>
void Rho<OrbitalsType>::rescaleTotalCharge()
{
    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    // Check total charge
    const double tcharge = computeTotalCharge();

    const int nspin = (int)rho_.size();

    Control& ct = *(Control::instance());
    double nel  = ct.getNel();
    if (tcharge > 0.)
    {
        double t1 = nel / tcharge;

        if (mmpi.PE0() && fabs(t1 - 1.) > 0.001)
        {
            (*MPIdata::sout)
                << " Rho<OrbitalsType>::rescaleTotalCharge(), charge = "
                << tcharge << std::endl;
            (*MPIdata::sout) << " Rescaling factor: " << t1 << std::endl;
            (*MPIdata::sout) << " Num. electrons: " << nel << std::endl;
        }
        if (fabs(t1 - 1.) > 0.02)
        {
            std::cerr << "Error on density too large to continue. Abort."
                      << std::endl;
            mmpi.abort();
        }

        for (int ispin = 0; ispin < nspin; ispin++)
            LinearAlgebraUtils<MemorySpace::Host>::MPscal(
                np_, t1, &rho_[ispin][0]);
    }
#ifdef DEBUG
    Mesh* mymesh = Mesh::instance();
    // Check total charge again
    double ttcharge = 0.;
    for (int ispin = 0; ispin < nspin; ispin++)
        for (int idx = 0; idx < np_; idx++)
        {
            ttcharge += (double)rho_[ispin][idx];
        }

    mmpi.allreduce(&ttcharge, 1, MPI_SUM);
    const pb::Grid& mygrid = mymesh->grid();
    ttcharge *= mygrid.vel();

    if (mmpi.instancePE0())
        (*MPIdata::sout) << std::fixed << std::setprecision(8)
                         << " Total charge after rescaling: " << ttcharge
                         << std::endl;
#endif
}

template <class OrbitalsType>
void Rho<OrbitalsType>::accumulateCharge(const double alpha, const short ix_max,
    const ORBDTYPE* const psii, const ORBDTYPE* const psij,
    RHODTYPE* const plrho)
{
    for (int ix = 0; ix < ix_max; ix++)
        plrho[ix] += (RHODTYPE)(alpha * (double)psii[ix] * (double)psij[ix]);
}

template <class OrbitalsType>
void Rho<OrbitalsType>::computeRhoSubdomain(const int iloc_init,
    const int iloc_end, const OrbitalsType& orbitals,
    const std::vector<double>& occ)
{
    assert(orbitals_type_ != OrthoType::UNDEFINED);
    if (verbosity_level_ > 2 && onpe0)
        (*MPIdata::sout)
            << "Rho<OrbitalsType>::computeRhoSubdomain, diagonal case..."
            << std::endl;

    Mesh* mymesh = Mesh::instance();

    const int loc_numpt = mymesh->locNumpt();
    const int n_colors  = orbitals_indexes_[0].size();

    RHODTYPE* const prho = &rho_[myspin_][0];

    for (int iloc = iloc_init; iloc < iloc_end; iloc++)
    {
        const int istart              = iloc * loc_numpt;
        std::vector<int>& loc_indexes = orbitals_indexes_[iloc];

        RHODTYPE* const lrho = prho + istart;

        // Loop over states and accumulate charge
        for (int icolor = 0; icolor < n_colors; icolor++)
        {
            const ORBDTYPE* const psi = orbitals.getPsi(icolor, iloc);
            const int st              = loc_indexes[icolor];
            assert(st < (int)occ.size());
            if (st >= 0)
            {
                const double t1 = 2. * (double)occ[st];
                for (int idx = 0; idx < loc_numpt; idx++)
                {
                    const double alpha = (double)psi[idx];
                    lrho[idx] += (RHODTYPE)(t1 * alpha * alpha);
                }
            }
        }
    }
}

template <class OrbitalsType>
void Rho<OrbitalsType>::computeRho(OrbitalsType& orbitals)
{
    ProjectedMatricesInterface& proj_matrices(*(orbitals.getProjMatrices()));

    computeRho(orbitals, proj_matrices);
}

template <class OrbitalsType>
void Rho<OrbitalsType>::resetValues()
{
    Mesh* mymesh        = Mesh::instance();
    const int subdivx   = mymesh->subdivx();
    const int loc_numpt = mymesh->locNumpt();

    memset(&rho_[myspin_][0], 0, subdivx * loc_numpt * sizeof(RHODTYPE));
}

template <class OrbitalsType>
void Rho<OrbitalsType>::computeRho(
    OrbitalsType& orbitals, ProjectedMatricesInterface& proj_matrices)
{
    assert(rho_.size() > 0);
    assert(rho_[myspin_].size() > 0);

    Mesh* mymesh      = Mesh::instance();
    const int subdivx = mymesh->subdivx();
    Control& ct       = *(Control::instance());

    resetValues();

    if (orbitals_type_ == OrthoType::Eigenfunctions
        || (orbitals_type_ == OrthoType::Orthonormal && ct.fullyOccupied()))
    {
        std::vector<double> occ(orbitals.numst());
        proj_matrices.getOccupations(occ);
        computeRhoSubdomain(0, subdivx, orbitals, occ);
    }
    else
    {
        proj_matrices.updateSubMatX();

        if (std::is_same<OrbitalsType, LocGridOrbitals<ORBDTYPE>>::value)
        {
            SquareLocalMatrices<MATDTYPE, memory_space_type>& localX(
                (orbitals.projMatrices())->getLocalX());

            // if (verbosity_level_ > 1 && onpe0)
            //    (*MPIdata::sout) << "Mask DM..." << endl;
            localX.applySymmetricMask(orbitals_indexes_);
        }
        computeRhoSubdomainUsingBlas3(0, subdivx, orbitals);
    }

    gatherSpin();

    rescaleTotalCharge();
}

template <class OrbitalsType>
template <class MatrixType>
void Rho<OrbitalsType>::computeRho(OrbitalsType& orbitals1,
    OrbitalsType& orbitals2, const MatrixType& dm11, const MatrixType& dm12,
    const MatrixType& /*dm21*/, const MatrixType& dm22)
{
    assert(orbitals_type_ == OrthoType::Nonorthogonal);

    Mesh* mymesh      = Mesh::instance();
    const int subdivx = mymesh->subdivx();

    resetValues();

    // 11 diagonal block
    ProjectedMatrices<MatrixType>* projmatrices1
        = dynamic_cast<ProjectedMatrices<MatrixType>*>(
            orbitals1.getProjMatrices());
    projmatrices1->updateSubMatX(dm11);
    computeRhoSubdomainUsingBlas3(0, subdivx, orbitals1);

    // 22 diagonal block
    ProjectedMatrices<MatrixType>* projmatrices2
        = dynamic_cast<ProjectedMatrices<MatrixType>*>(
            orbitals2.getProjMatrices());
    projmatrices2->updateSubMatX(dm22);
    computeRhoSubdomainUsingBlas3(0, subdivx, orbitals2);

    MatrixType dm(dm12);
    dm.scal(2.); // use symmetry to reduce work
    projmatrices1->updateSubMatX(dm);
    computeRhoSubdomainUsingBlas3(0, subdivx, orbitals1, orbitals2);

    gatherSpin();

    rescaleTotalCharge();
}

template <class OrbitalsType>
void Rho<OrbitalsType>::computeRhoSubdomainUsingBlas3(const int iloc_init,
    const int iloc_end, const OrbitalsType& orbitals1,
    const OrbitalsType& orbitals2)
{
    assert(orbitals1.getLda() == orbitals2.getLda());
    assert(orbitals1.chromatic_number() == orbitals2.chromatic_number());

    compute_blas_tm_.start();

    Mesh* mymesh    = Mesh::instance();
    const int nrows = mymesh->locNumpt();

    RHODTYPE* const prho = &rho_[myspin_][0];

    SquareLocalMatrices<MATDTYPE, memory_space_type>& localX(
        (orbitals1.projMatrices())->getLocalX());

    const int ld      = orbitals1.getLda();
    const int ncols   = orbitals1.chromatic_number();
    ORBDTYPE* product = new ORBDTYPE[nrows * ncols];

    for (int iloc = iloc_init; iloc < iloc_end; iloc++)
    {
        const int istart = iloc * nrows;

        RHODTYPE* const lrho = &prho[istart];

        const MATDTYPE* const mat = localX.getSubMatrix(iloc);
        ORBDTYPE* phi1            = orbitals1.getPsi(0, iloc);

        // O(N^3) part
#ifdef HAVE_MAGMA
        // If we have magma, we move all the data on the device
        std::unique_ptr<ORBDTYPE[], void (*)(ORBDTYPE*)> product_dev(
            MemoryDev<ORBDTYPE>::allocate(nrows * ncols),
            MemoryDev<ORBDTYPE>::free);

        LinearAlgebraUtils<MemorySpace::Device>::MPgemmNN(nrows, ncols, ncols,
            1., phi1, ld, mat, ncols, 0., product_dev.get(), nrows);

        // Move the data back on the host
        MemorySpace::copy_to_host(product_dev, nrows * ncols, product);
#else // MAGMA is not installed
        LinearAlgebraUtils<MemorySpace::Host>::MPgemmNN(
            nrows, ncols, ncols, 1., phi1, ld, mat, ncols, 0., product, nrows);
#endif

        // O(N^2) part: if we have MAGMA and we can offload with OpenMP, we
        // perform the operation on the device. Otherwise, we do it on the host.
#ifdef HAVE_OPENMP_OFFLOAD
        ORBDTYPE* product_alias = product_dev.get();
        ORBDTYPE* phi2_alias    = orbitals2.getPsi(0, iloc);
        // Copy lrho to the device
        std::unique_ptr<RHODTYPE[], void (*)(RHODTYPE*)> lrho_dev(
            MemoryDev<RHODTYPE>::allocate(nrows), MemoryDev<RHODTYPE>::free);
        MemorySpace::copy_to_dev(lrho, nrows, lrho_dev);
        RHODTYPE* const lrho_alias = lrho_dev.get();
#else
        ORBDTYPE* product_alias = product;
        ORBDTYPE* phi2          = orbitals2.getPsi(0, iloc);
        using memory_space_type = typename OrbitalsType::memory_space_type;
        ORBDTYPE* phi2_alias    = MemorySpace::Memory<ORBDTYPE,
            memory_space_type>::allocate_host_view(ld * ncols + nrows);
        MemorySpace::Memory<ORBDTYPE, memory_space_type>::copy_view_to_host(
            phi2, ld * ncols + nrows, phi2_alias);
        RHODTYPE* const lrho_alias = lrho;
#endif
        MGMOL_PARALLEL_FOR_COLLAPSE(2, lrho_alias, product_alias, phi2_alias)
        for (int j = 0; j < ncols; ++j)
        {
            for (int i = 0; i < nrows; ++i)
            {
#pragma omp atomic update
                lrho_alias[i]
                    += product_alias[j * nrows + i] * phi2_alias[j * ld + i];
            }
        }
#if defined(HAVE_MAGMA) && defined(HAVE_OPENMP_OFFLOAD)
        // Move lrho back to the host
        MemorySpace::copy_to_host(lrho_alias, nrows, lrho);
#else
        MemorySpace::Memory<ORBDTYPE, memory_space_type>::free_host_view(
            phi2_alias);
#endif
    }

    delete[] product;

    compute_blas_tm_.stop();
}

template <class OrbitalsType>
template <class MatrixType>
void Rho<OrbitalsType>::computeRho(OrbitalsType& orbitals, const MatrixType& dm)
{
    assert(orbitals_type_ == OrthoType::Nonorthogonal);

    iterative_index_++;

    Mesh* mymesh      = Mesh::instance();
    const int subdivx = mymesh->subdivx();

    resetValues();

    ProjectedMatrices<MatrixType>* projmatrices
        = dynamic_cast<ProjectedMatrices<MatrixType>*>(
            orbitals.getProjMatrices());
    projmatrices->updateSubMatX(dm);

    computeRhoSubdomainUsingBlas3(0, subdivx, orbitals);

    gatherSpin();

    rescaleTotalCharge();
}

template <class OrbitalsType>
void Rho<OrbitalsType>::init(const RHODTYPE* const rhoc)
{
    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    int ione        = 1;

    // Initialize the charge density
    if (verbosity_level_ > 2 && mmpi.instancePE0())
        (*MPIdata::sout) << "Rho: Initialize electronic density value with rhoc"
                         << std::endl;
    for (unsigned int i = 0; i < rho_.size(); i++)
    {
        Tcopy(&np_, rhoc, &ione, &rho_[i][0], &ione);
        if (rho_.size() == 2)
            LinearAlgebraUtils<MemorySpace::Host>::MPscal(
                np_, 0.5, &rho_[i][0]);
    }
    iterative_index_ = 0;

    rescaleTotalCharge();
}

template <class OrbitalsType>
void Rho<OrbitalsType>::initUniform()
{
    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    // Initialize the charge density
    if (mmpi.instancePE0())
        (*MPIdata::sout) << " Initialize electronic density with uniform value"
                         << std::endl;
    for (unsigned int i = 0; i < rho_.size(); i++)
    {
        for (int j = 0; j < np_; j++)
            rho_[i][j] = 1.;
        if (rho_.size() == 2)
            LinearAlgebraUtils<MemorySpace::Host>::MPscal(
                np_, 0.5, &rho_[i][0]);
    }
    iterative_index_ = 0;

    rescaleTotalCharge();
}

// read rho and potentials form a hdf5 file
template <class OrbitalsType>
int Rho<OrbitalsType>::readRestart(HDFrestart& file)
{
    Control& ct = *(Control::instance());
    if (onpe0 && ct.verbose > 0)
        (*MPIdata::sout) << "Try to read density" << std::endl;

    // Read the Density
    file.read_1func_hdf5(&rho_[myspin_][0], "Density");
    iterative_index_ = 0;

    return 0;
}

template <class OrbitalsType>
template <typename T2>
double Rho<OrbitalsType>::dotWithRho(const T2* const func) const
{
    MGmol_MPI& mmpi = *(MGmol_MPI::instance());

    double val = LinearAlgebraUtils<MemorySpace::Host>::MPdot(
        np_, &rho_[mmpi.myspin()][0], func);

    double esum = 0.;
    mmpi.allreduce(&val, &esum, 1, MPI_SUM);
    val = esum;

    mmpi.allreduceSpin(&val, &esum, 1, MPI_SUM);
    val = esum;

    return val;
}

template <class OrbitalsType>
void Rho<OrbitalsType>::gatherSpin()
{
    if (nspin_ < 2) return;

    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    // if(mmpi.instancePE0())cout<<"Rho<OrbitalsType>::gatherSpin(),
    // myspin_="<<myspin_<<endl;
    mmpi.exchangeDataSpin(&rho_[myspin_][0], &rho_[(myspin_ + 1) % 2][0], np_);
}

template <class OrbitalsType>
void Rho<OrbitalsType>::printTimers(std::ostream& os)
{
    update_tm_.print(os);
    compute_tm_.print(os);
    compute_blas_tm_.print(os);
}

template class Rho<LocGridOrbitals<ORBDTYPE>>;
template class Rho<ExtendedGridOrbitals<ORBDTYPE>>;

template double Rho<LocGridOrbitals<ORBDTYPE>>::dotWithRho<double>(
    const double* const func) const;
template double Rho<ExtendedGridOrbitals<ORBDTYPE>>::dotWithRho<double>(
    const double* const func) const;

#ifdef MGMOL_USE_SCALAPACK
template void Rho<ExtendedGridOrbitals<ORBDTYPE>>::computeRho<
    dist_matrix::DistMatrix<double>>(ExtendedGridOrbitals<ORBDTYPE>&,
    ExtendedGridOrbitals<ORBDTYPE>&, const dist_matrix::DistMatrix<double>&,
    const dist_matrix::DistMatrix<double>&,
    const dist_matrix::DistMatrix<double>&,
    const dist_matrix::DistMatrix<double>&);
template void Rho<ExtendedGridOrbitals<ORBDTYPE>>::computeRho<
    dist_matrix::DistMatrix<double>>(
    ExtendedGridOrbitals<ORBDTYPE>&, const dist_matrix::DistMatrix<double>&);
template void
Rho<LocGridOrbitals<ORBDTYPE>>::computeRho<dist_matrix::DistMatrix<double>>(
    LocGridOrbitals<ORBDTYPE>&, const dist_matrix::DistMatrix<double>&);
#endif
#ifdef MGMOL_USE_MIXEDP
template double Rho<LocGridOrbitals<ORBDTYPE>>::dotWithRho<float>(
    const float* const func) const;
#endif
template void Rho<ExtendedGridOrbitals<ORBDTYPE>>::computeRho<ReplicatedMatrix>(
    ExtendedGridOrbitals<ORBDTYPE>&, const ReplicatedMatrix&);
template void Rho<ExtendedGridOrbitals<ORBDTYPE>>::computeRho<ReplicatedMatrix>(
    ExtendedGridOrbitals<ORBDTYPE>&, ExtendedGridOrbitals<ORBDTYPE>&,
    const ReplicatedMatrix&, const ReplicatedMatrix&, const ReplicatedMatrix&,
    const ReplicatedMatrix&);
template void Rho<LocGridOrbitals<ORBDTYPE>>::computeRho<ReplicatedMatrix>(
    LocGridOrbitals<ORBDTYPE>&, const ReplicatedMatrix&);

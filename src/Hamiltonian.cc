// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "Hamiltonian.h"
#include "Control.h"
#include "ExtendedGridOrbitals.h"
#include "LocGridOrbitals.h"
#include "Mesh.h"
#include "Potentials.h"
#include "ProjectedMatrices.h"
#include "ReplicatedMatrix.h"

template <class T>
Hamiltonian<T>::Hamiltonian()
{
    itindex_ = -1;
    lapOper_ = nullptr;
    hlphi_   = nullptr;
    pot_     = new Potentials();
}

template <class T>
Hamiltonian<T>::~Hamiltonian()
{
    if (hlphi_ != nullptr) delete hlphi_;
    if (lapOper_ != nullptr) delete lapOper_;
    delete pot_;
}

template <class T>
void Hamiltonian<T>::setup(const pb::Grid& myGrid, const int lap_type)
{
    if (lapOper_ != nullptr) delete lapOper_;
    lapOper_ = LapFactory<ORBDTYPE>::createLap(myGrid, lap_type);
}

template <class T>
const T& Hamiltonian<T>::applyLocal(T& phi, const bool force)
{
    assert(phi.getIterativeIndex() >= 0);
    assert(pot_->getIterativeIndex() >= 0);

    if (hlphi_ == nullptr) hlphi_ = new T("Hphi", phi, false);
    if (!hlphi_->isCompatibleWith(phi))
    {
        delete hlphi_;
        itindex_ = -1;
        hlphi_   = new T("Hphi", phi, false);
    }
    const int new_index
        = 100 * phi.getIterativeIndex() + pot_->getIterativeIndex();
#ifdef DEBUG
    if (onpe0)
    {
        (*MPIdata::sout) << "Hamiltonian<T>::applyLocal(), new_index ="
                         << new_index << std::endl;
        (*MPIdata::sout) << "Hamiltonian<T>::applyLocal(), itindex_  ="
                         << itindex_ << std::endl;
    }
#endif
    if (force || new_index != itindex_)
    {
        applyLocal(phi.chromatic_number(), phi, *hlphi_);

        itindex_ = new_index;
#ifdef PRINT_OPERATIONS
    }
    else
    {
        if (onpe0)
            (*MPIdata::sout)
                << "Hamiltonian<T>::hlphi up to date, itindex_=" << itindex_
                << ", Potential index=" << pot_->getIterativeIndex()
                << std::endl;
#endif
    }
    return *hlphi_;
}

template <class T>
void Hamiltonian<T>::applyLocal(const int ncolors, T& phi, T& hphi)
{
#ifdef PRINT_OPERATIONS
    if (onpe0)
        (*MPIdata::sout) << "Hamiltonian<T>::applyLocal() for " << ncolors
                         << " states" << std::endl;
#endif

    const Control& ct      = *(Control::instance());
    Mesh* mymesh           = Mesh::instance();
    const pb::Grid& mygrid = mymesh->grid();

    const POTDTYPE* const vtot = pot_->vtot();

    phi.setDataWithGhosts();
    phi.trade_boundaries();

    // start timer after filling ghost values
    apply_Hloc_tm_.start();

    auto first_value       = phi.getPsi(0)[0];
    using OrbitalsDataType = decltype(first_value);

    using memory_space_type = typename T::memory_space_type;
    using MultiVector = pb::GridFuncVector<OrbitalsDataType, memory_space_type>;

    if (ct.lap_type != 2)
    {
        // need phi with ghosts
        MultiVector& gfvphi(*phi.getPtDataWGhosts());

        // gfvw1 = potential * phi
        pb::GridFunc<POTDTYPE> gfpot(
            mygrid, ct.bcWF[0], ct.bcWF[1], ct.bcWF[2]);
        gfpot.assign(vtot);
        gfpot.trade_boundaries();
        const std::vector<std::vector<int>>& gid(phi.getOverlappingGids());
        MultiVector tmp1(mygrid, ct.bcWF[0], ct.bcWF[1], ct.bcWF[2], gid);
        tmp1.pointwiseProduct(gfvphi, gfpot);

        MultiVector tmp2(mygrid, ct.bcWF[0], ct.bcWF[1], ct.bcWF[2], gid);

        if (ct.Mehrstellen())
        {
            // work1 = B*V*psi
            tmp1.applyRHS(0, tmp2);
        }
        else
        {
            tmp2.copyFrom(tmp1);
        }
        // tmp1 = -Lap*phi
        gfvphi.applyLap(ct.lap_type, tmp1);
        // tmp2 = -Lap*phi + B*V*psi
        tmp2.axpy((OrbitalsDataType)1., tmp1);
        // set hpsi data without ghosts
        hphi.setPsi(tmp2);
    }
    else
    {
        // This loop is not thread safe as GridFunc ghost values filling
        // MPI calls may conflicts (all use the same tag)
        // #pragma omp parallel for
        for (int i = 0; i < ncolors; i++)
        {
            using memory_space_type = typename T::memory_space_type;
            auto ihphi              = hphi.getPsi(i);
            unsigned int const size = hphi.getNumpt();
            OrbitalsDataType* ihphi_host_view
                = MemorySpace::Memory<OrbitalsDataType,
                    memory_space_type>::allocate_host_view(size);
            MemorySpace::Memory<OrbitalsDataType,
                memory_space_type>::copy_view_to_host(hphi.getPsi(i), size,
                ihphi_host_view);

            lapOper_->applyWithPot(
                phi.getFuncWithGhosts(i), vtot, ihphi_host_view);

            MemorySpace::Memory<OrbitalsDataType,
                memory_space_type>::copy_view_to_dev(ihphi_host_view, size,
                ihphi);
            MemorySpace::Memory<OrbitalsDataType,
                memory_space_type>::free_host_view(ihphi_host_view);
        }
    }

    apply_Hloc_tm_.stop();
}

template <class T>
void Hamiltonian<T>::applyDeltaPot(const T& phi, T& hphi)
{
    const std::vector<POTDTYPE>& dv(pot_->dv());

    phi.applyDiagonalOp(dv, hphi);
}

// add to hij the elements <phi1|Hloc|phi2>
// corresponding to the local part of the Hamiltonian
#ifdef MGMOL_USE_SCALAPACK
template <>
template <>
void Hamiltonian<LocGridOrbitals<ORBDTYPE>>::addHlocal2matrix(
    LocGridOrbitals<ORBDTYPE>& phi1, LocGridOrbitals<ORBDTYPE>& phi2,
    dist_matrix::DistMatrix<double>& hij, const bool force)
{
    applyLocal(phi2, force);

#ifdef PRINT_OPERATIONS
    if (onpe0)
        (*MPIdata::sout) << "Hamiltonian<T>::addHlocal2matrix()" << std::endl;
#endif

    phi1.addDotWithNcol2Matrix(*hlphi_, hij);
}

template <>
template <>
void Hamiltonian<ExtendedGridOrbitals<ORBDTYPE>>::addHlocal2matrix(
    ExtendedGridOrbitals<ORBDTYPE>& phi1, ExtendedGridOrbitals<ORBDTYPE>& phi2,
    dist_matrix::DistMatrix<double>& hij, const bool force)
{
    applyLocal(phi2, force);

#ifdef PRINT_OPERATIONS
    if (onpe0)
        (*MPIdata::sout) << "Hamiltonian<T>::addHlocal2matrix()" << std::endl;
#endif

    // hij.print(std::cout, 0, 0, 5, 5);

    phi1.addDotWithNcol2Matrix(*hlphi_, hij);

    // hij.print(std::cout, 0, 0, 5, 5);
}
#endif

template <>
template <>
void Hamiltonian<ExtendedGridOrbitals<ORBDTYPE>>::addHlocal2matrix(
    ExtendedGridOrbitals<ORBDTYPE>& phi1, ExtendedGridOrbitals<ORBDTYPE>& phi2,
    ReplicatedMatrix& hij, const bool force)
{
    applyLocal(phi2, force);

#ifdef PRINT_OPERATIONS
    if (onpe0)
        (*MPIdata::sout) << "Hamiltonian<T>::addHlocal2matrix() at line "
                         << __LINE__ << std::endl;
#endif

    phi1.addDotWithNcol2Matrix(*hlphi_, hij);
}

template <>
template <>
void Hamiltonian<LocGridOrbitals<ORBDTYPE>>::addHlocal2matrix(
    LocGridOrbitals<ORBDTYPE>& phi1, LocGridOrbitals<ORBDTYPE>& phi2,
    ReplicatedMatrix& hij, const bool force)
{
    (void)phi1;
    (void)phi2;
    (void)hij;

    applyLocal(phi2, force);

    // phi1.addDotWithNcol2Matrix(*hlphi_, hij);
    std::cerr << "Not implemented!" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}

template <class T>
void Hamiltonian<T>::addHlocalij(
    T& phi1, T& phi2, ProjectedMatricesInterface* proj_matrices)
{
    applyLocal(phi2);

#ifdef PRINT_OPERATIONS
    if (onpe0)
        (*MPIdata::sout) << "Hamiltonian<T>::addHLocalij() at line " << __LINE__
                         << std::endl;
#endif

    addHlocalij(phi1, proj_matrices);
}

template <class T>
void Hamiltonian<T>::addHlocalij(
    T& phi1, ProjectedMatricesInterface* proj_matrices)
{
    SquareLocalMatrices<MATDTYPE, MemorySpace::Host> slm(
        phi1.subdivx(), phi1.chromatic_number());

    phi1.computeLocalProduct(*hlphi_, slm);

    proj_matrices->setLocalMatrixElementsHl(slm);

    proj_matrices->consolidateH();
}

template <>
template <>
void Hamiltonian<LocGridOrbitals<ORBDTYPE>>::addHlocal2matrix(
    LocGridOrbitals<ORBDTYPE>& phi1, LocGridOrbitals<ORBDTYPE>& phi2,
    VariableSizeMatrix<sparserow>& mat, const bool force)
{
    Control& ct = *(Control::instance());

    applyLocal(phi2, force);

#ifdef PRINT_OPERATIONS
    if (onpe0)
        (*MPIdata::sout) << "Hamiltonian<T>::addHLocalij() at line " << __LINE__
                         << std::endl;
#endif

    SquareLocalMatrices<MATDTYPE, MemorySpace::Host> ss(
        phi1.subdivx(), phi1.chromatic_number());

    phi1.computeLocalProduct(*hlphi_, ss);

    mat.insertMatrixElements(ss, phi1.getOverlappingGids(), ct.numst);
}

template class Hamiltonian<LocGridOrbitals<ORBDTYPE>>;
template class Hamiltonian<ExtendedGridOrbitals<ORBDTYPE>>;

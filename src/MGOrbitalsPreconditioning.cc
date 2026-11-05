// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "MGOrbitalsPreconditioning.h"

#include "Control.h"
#include "ExtendedGridOrbitals.h"
#include "LocGridOrbitals.h"
#include "LocalizationRegions.h"
#include "MasksSet.h"
#include "Mesh.h"
#include "Potentials.h"
#include "Preconditioning.h"
#include "ProjectedMatricesInterface.h"

template <class OrbitalsType, typename PDataType>
MGOrbitalsPreconditioning<OrbitalsType, PDataType>::MGOrbitalsPreconditioning(
    const short mg_levels, const pb::LapType lap_type)
    : mg_levels_(mg_levels), lap_type_(lap_type), is_set_(false)
{
    Control& ct(*(Control::instance()));
    Mesh* mymesh = Mesh::instance();
    const pb::Grid& mygrid(mymesh->grid());

    precond_
        = std::make_shared<Preconditioning<PDataType>>(lap_type_, mg_levels_,
            ct.mg_npresmoothing_, ct.mg_npostsmoothing_, mygrid, ct.bcWF);
}

template <class OrbitalsType, typename PDataType>
MGOrbitalsPreconditioning<OrbitalsType, PDataType>::~MGOrbitalsPreconditioning()
{
    assert(is_set_);
    assert(precond_);
}

template <class OrbitalsType, typename PDataType>
void MGOrbitalsPreconditioning<OrbitalsType, PDataType>::setup(
    OrbitalsType& orbitals, MasksSet* currentMasks,
    const std::shared_ptr<LocalizationRegions>& lrs)
{
    assert(!is_set_);

    Control& ct(*(Control::instance()));
    Mesh* mymesh = Mesh::instance();
    const pb::Grid& mygrid(mymesh->grid());

    if (currentMasks != nullptr)
    {
        // set masks in GridFuncVector class
        map2masks_
            = std::make_shared<Map2Masks>(currentMasks, lrs->getOverlapGids());
        pb::GridFuncVector<PDataType, memory_space_type>::setMasks(
            map2masks_.get());
    }

    precond_->setup(orbitals.getOverlappingGids());

    assert(orbitals.chromatic_number()
           == static_cast<int>(orbitals.getOverlappingGids()[0].size()));

    gfv_work1_
        = std::make_shared<pb::GridFuncVector<PDataType, memory_space_type>>(
            mygrid, ct.bcWF[0], ct.bcWF[1], ct.bcWF[2],
            orbitals.getOverlappingGids());

    gfv_work2_
        = std::make_shared<pb::GridFuncVector<PDataType, memory_space_type>>(
            mygrid, ct.bcWF[0], ct.bcWF[1], ct.bcWF[2],
            orbitals.getOverlappingGids());

    if (sizeof(ORBDTYPE) != sizeof(PDataType))
        gfv_work3_
            = std::make_shared<pb::GridFuncVector<ORBDTYPE, memory_space_type>>(
                mygrid, ct.bcWF[0], ct.bcWF[1], ct.bcWF[2],
                orbitals.getOverlappingGids());

    is_set_ = true;

    assert(gfv_work2_);
}

template <class OrbitalsType, typename PDataType>
void MGOrbitalsPreconditioning<OrbitalsType, PDataType>::precond(
    OrbitalsType& orbitals)
{
    assert(is_set_);
    assert(precond_);
    assert(gamma_ > 0.);
    assert(gfv_work1_);

#ifdef PRINT_OPERATIONS
    if (onpe0) (*MPIdata::sout) << "T::precond_mg()..." << endl;
#endif
    precond_tm_.start();

    // initialize gfv_work2_ with data from orbitals
    if (sizeof(ORBDTYPE) == sizeof(PDataType))
    {
        orbitals.setDataWithGhosts(gfv_work2_.get());
    }
    else
    {
        // Convert to data with ghosts first, then convert to different
        // precision. This is more efficient in practice than doing precision
        // conversion in setDataWithGhosts
        orbitals.setDataWithGhosts(gfv_work3_.get());

        gfv_work2_->copyFrom(*gfv_work3_);
    }

    gfv_work1_->resetData();
    gfv_work1_->axpy((PDataType)gamma_, *gfv_work2_);

    // block-implemented preconditioner
    precond_->mg(*gfv_work1_, *gfv_work2_, lap_type_, 0);

    if (sizeof(ORBDTYPE) == sizeof(PDataType))
    {
        orbitals.setPsi(*gfv_work1_);
    }
    else
    {
        // Convert to orbitals precision first
        gfv_work3_->copyFrom(*gfv_work1_);

        // set orbitals to GridFuncVector second
        orbitals.setPsi(*gfv_work3_);
    }

#ifdef PRINT_OPERATIONS
    if (onpe0)
        (*MPIdata::sout)
            << "MGOrbitalsPreconditioning<OrbitalsType,PDataType>::"
               "precond_mg() done"
            << endl;
#endif
    precond_tm_.stop();
}

template <class OrbitalsType, typename PDataType>
void MGOrbitalsPreconditioning<OrbitalsType, PDataType>::setGamma(
    const pb::Lap<ORBDTYPE>& lapOper, const Potentials& pot,
    const short mg_levels, ProjectedMatricesInterface* proj_matrices)
{
    assert(precond_);
    assert(is_set_);

    const double small_eig = proj_matrices->getLowestEigenvalue();
    double diag            = lapOper.invDiagEl();
    double vmax            = pot.max();

    // diag * 4^{N_level+1}
    // gamma = inverse of the largest eigenvalue for the low frequency error
    gamma_ = diag;
    for (short ln = 0; ln <= mg_levels; ln++)
    {
        gamma_ *= 4.;
    }
    gamma_ = 1.0 / (2.0 / gamma_ + fabs(vmax - small_eig));
#ifdef DEBUG
    Control& ct(*(Control::instance()));
    if (onpe0 && ct.verbose > 2)
        (*MPIdata::sout) << " time step for low frequencies corrections = "
                         << gamma_ << endl;
#endif
}

template <class OrbitalsType, typename PDataType>
void MGOrbitalsPreconditioning<OrbitalsType, PDataType>::printTimers(
    std::ostream& os)
{
    precond_tm_.print(os);
}

template class MGOrbitalsPreconditioning<LocGridOrbitals<ORBDTYPE>, float>;
template class MGOrbitalsPreconditioning<LocGridOrbitals<ORBDTYPE>, double>;
template class MGOrbitalsPreconditioning<ExtendedGridOrbitals<ORBDTYPE>, float>;
template class MGOrbitalsPreconditioning<ExtendedGridOrbitals<ORBDTYPE>,
    double>;

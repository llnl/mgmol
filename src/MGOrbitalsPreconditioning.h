// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#ifndef MGMOL_MGOrbitalsPreconditioning_H
#define MGMOL_MGOrbitalsPreconditioning_H

#include "GridFuncVector.h"
#include "Lap.h"
#include "Map2Masks.h"
#include "OrbitalsPreconditioning.h"
#include "Preconditioning.h"

#include <memory>

// class Masks4Orbitals;
// class MasksSet;
class ProjectedMatricesInterface;
class Potentials;
// class LocalizationRegions;

template <class OrbitalsType, typename PDataType>
class MGOrbitalsPreconditioning : public OrbitalsPreconditioning<OrbitalsType>
{
private:
#ifdef HAVE_MAGMA
    using memory_space_type = MemorySpace::Device;
#else
    using memory_space_type = MemorySpace::Host;
#endif

    std::shared_ptr<Preconditioning<PDataType>> precond_;

    // work arrays with preconditioner precision
    std::shared_ptr<pb::GridFuncVector<PDataType, memory_space_type>>
        gfv_work1_;
    std::shared_ptr<pb::GridFuncVector<PDataType, memory_space_type>>
        gfv_work2_;

    // tmp work array for case ORBDTYPE!=PDataType
    std::shared_ptr<pb::GridFuncVector<ORBDTYPE, memory_space_type>> gfv_work3_;

    short mg_levels_;

    pb::LapType lap_type_;

    bool is_set_;

    // coefficient for preconditioning
    double gamma_;

    // timers
    static Timer precond_tm_;

    std::shared_ptr<Map2Masks> map2masks_;

public:
    MGOrbitalsPreconditioning(
        const short mg_levels, const pb::LapType lap_type);

    ~MGOrbitalsPreconditioning();

    void setup(OrbitalsType& orbitals, MasksSet*,
        const std::shared_ptr<LocalizationRegions>&) override;
    void precond(OrbitalsType& orbitals) override;
    void setGamma(const pb::Lap<ORBDTYPE>& lapOper, const Potentials& pot,
        const short mg_levels, ProjectedMatricesInterface* proj_matrices);
    static void printTimers(std::ostream& os);
};

template <class OrbitalsType, typename PDataType>
Timer MGOrbitalsPreconditioning<OrbitalsType, PDataType>::precond_tm_(
    "MGOrbitalsPreconditioning::precond");

#endif

// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#ifndef MGMOL_OrbitalsExtrapolationFACTORY_H
#define MGMOL_OrbitalsExtrapolationFACTORY_H

#include "MGmol_abort.h"
#include "OrbitalsExtrapolationOrder2.h"
#include "OrbitalsExtrapolationOrder3.h"

template <class T>
class OrbitalsExtrapolationFactory
{
public:
    static OrbitalsExtrapolation<T>* create(const WFExtrapolationType type)
    {
        OrbitalsExtrapolation<T>* orbitals_extrapol;
        switch (type)
        {
            case WFExtrapolationType::Order2:
                orbitals_extrapol = new OrbitalsExtrapolationOrder2<T>();
                break;
            case WFExtrapolationType::Order3:
                orbitals_extrapol = new OrbitalsExtrapolationOrder3<T>();
                break;
            default:
                mgmol_abort(
                    "OrbitalsExtrapolation* create() --- option invalid");
        }
        return orbitals_extrapol;
    }
};

#endif

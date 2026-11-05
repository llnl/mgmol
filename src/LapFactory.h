// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE
#ifndef MGMOL_LAPFACTORY_H
#define MGMOL_LAPFACTORY_H

#include "Laph2.h"
#include "Laph4.h"
#include "Laph4M.h"
#include "Laph4MP.h"
#include "Laph6.h"
#include "Laph8.h"
#include "MGmol_MPI.h"

template <class T>
class LapFactory
{
public:
    static pb::Lap<T>* createLap(const pb::Grid& grid, const pb::LapType type)
    {
        pb::Lap<T>* lap = nullptr;
        switch (type)
        {
            case pb::LapType::Mehrstellen1:
                lap = new pb::Laph4M<T>(grid);
                break;
            case pb::LapType::FD2nd:
                lap = new pb::Laph2<T>(grid);
                break;
            case pb::LapType::FD4th:
                lap = new pb::Laph4<T>(grid);
                break;
            case pb::LapType::FD6th:
                lap = new pb::Laph6<T>(grid);
                break;
            case pb::LapType::FD8th:
                lap = new pb::Laph8<T>(grid);
                break;
            case pb::LapType::Mehrstellen2:
                lap = new pb::Laph4MP<T>(grid);
                break;
            default:
                MGmol_MPI& mmpi = *(MGmol_MPI::instance());
                if (mmpi.instancePE0())
                    std::cerr << "LapFactory::createLap() --- option invalid:"
                              << std::endl;
                mmpi.abort();
        }
        return lap;
    }
};

#endif

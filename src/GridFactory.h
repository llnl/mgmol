// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#ifndef MGMOL_GRIDFACTORY_H
#define MGMOL_GRIDFACTORY_H

#include "Grid.h"
#include "Lap.h"
#include "MPIdata.h"
#include "Poisson.h"

class GridFactory
{
public:
    static pb::Grid* createGrid(const unsigned ngpts[3], const double origin[3],
        const double lattice[3], const pb::LapType lap_type,
        const bool diel_flag, const pb::PEenv& myPEenv)
    {
        int nghosts = 0;
        switch (lap_type)
        {
            case pb::LapType::Mehrstellen1:
                nghosts = diel_flag ? 2 : 1;
                break;
            case pb::LapType::FD2nd:
                nghosts = 1;
                break;
            case pb::LapType::FD4th:
                nghosts = 2;
                break;
            case pb::LapType::FD6th:
                nghosts = 3;
                break;
            case pb::LapType::FD8th:
                nghosts = 4;
                break;
            case pb::LapType::Mehrstellen2:
                nghosts = diel_flag ? 2 : 1;
                break;
            default:
                (*MPIdata::serr)
                    << "GridFactory::createGrid() --- option invalid."
                    << std::endl;
                exit(2);
        }
        return (new pb::Grid(origin, lattice, ngpts, myPEenv, nghosts));
    }
};

#endif

// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#ifndef ROM_WORKFLOWS_H
#define ROM_WORKFLOWS_H

#include "Control.h"
#include "ExtendedGridOrbitals.h"
#include "LocGridOrbitals.h"
#include "MGmol.h"
#include "MGmol_MPI.h"
#include "MPIdata.h"
#include "Mesh.h"
#include "Potentials.h"
#include "ProjectedMatrices.h"
#include "mgmol_run.h"
#include "tools.h"

#include <cassert>
#include <fenv.h>
#include <iostream>
#include <iterator>
#include <sys/cdefs.h>
#include <time.h>
#include <vector>

#include <mpi.h>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include "librom.h"
#include "utils/HDFDatabase.h"
#include "utils/mpi_utils.h"

template <class OrbitalsType>
void readRestartFiles(MGmolInterface* mgmol_);

template <class OrbitalsType>
void buildROMPoissonOperator(MGmolInterface* mgmol_);

template <class OrbitalsType>
void runPoissonROM(MGmolInterface* mgmol_);

template <class OrbitalsType>
void testROMPoissonOperator(MGmolInterface* mgmol_);

template <class OrbitalsType>
void testROMRhoOperator(MGmolInterface* mgmol_);

template <class OrbitalsType>
void testROMIonDensity(MGmolInterface* mgmol_);

void computeRhoOnSamplePts(const CAROM::Matrix& dm,
    const CAROM::Matrix& phi_basis, const CAROM::Matrix& rom_phi,
    const std::vector<int>& local_idx, CAROM::Vector& sampled_rho);

#endif // ROM_WORKFLOWS_H

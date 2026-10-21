// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE
#include "Ions.h"
#include "ProjectedMatrices.h"

template <class OrbitalsType, class MatrixType>
void computeDipoleMoment(const OrbitalsType& orbitals, Ions& ions,
    ProjectedMatrices<MatrixType>& projmatrices);

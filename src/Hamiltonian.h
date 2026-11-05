// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#ifndef MGMOL_HAMILTONIAN_H
#define MGMOL_HAMILTONIAN_H

#include "LapFactory.h"
#include "ProjectedMatricesInterface.h"
#include "Timer.h"
#include "VariableSizeMatrix.h"

class Potentials;

template <class OrbitalsType>
class Hamiltonian
{
    pb::Lap<ORBDTYPE>* lapOper_;
    Potentials* pot_;
    OrbitalsType* hlphi_;
    int itindex_;

    static Timer apply_Hloc_tm_;

public:
    static Timer apply_Hloc_tm() { return apply_Hloc_tm_; }

    Hamiltonian();
    ~Hamiltonian();

    void setup(const pb::Grid& myGrid, const pb::LapType lap_type);

    Potentials& potential() { return *pot_; }
    void setHlOutdated() { itindex_ = -1; }
    pb::Lap<ORBDTYPE>* lapOper() { return lapOper_; }

    const OrbitalsType& applyLocal(OrbitalsType& phi, const bool force = false);
    void applyLocal(const int nstates, OrbitalsType& phi, OrbitalsType& hphi);

    /*!
     * Apply potential difference dv to phi
     */
    void applyDeltaPot(const OrbitalsType& phi, OrbitalsType& hphi);

    template <class MatrixType>
    void addHlocal2matrix(OrbitalsType& orbitals1, OrbitalsType& orbitals2,
        MatrixType& mat, const bool force);
    void addHlocalij(OrbitalsType& orbitals1, OrbitalsType& orbitals2,
        ProjectedMatricesInterface*);
    void addHlocalij(OrbitalsType& orbitals1, ProjectedMatricesInterface*);
};
// Instantiate static variable here to avoid clang warnings
template <class OrbitalsType>
Timer Hamiltonian<OrbitalsType>::apply_Hloc_tm_("Hamiltonian::apply_Hloc");
#endif

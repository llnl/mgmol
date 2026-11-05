// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#ifndef MGMOL_PCG_SOLVER_DIEL_H_
#define MGMOL_PCG_SOLVER_DIEL_H_

#include "Lap.h"
#include "PB.h"
#include "PBh2.h"
#include "PBh4.h"
#include "PBh4M.h"
#include "PBh4MP.h"
#include "PBh6.h"
#include "PBh8.h"
#include "Poisson.h"

#include <vector>

template <class T, typename ScalarType>
class PCGSolver_Diel
{

private:
    std::vector<pb::Grid*> grid_;
    short bc_[3];
    // operators
    T oper_;
    std::vector<T*> pc_oper_;
    const pb::LapType lap_type_;
    std::vector<pb::GridFunc<ScalarType>*> gf_work_;
    std::vector<pb::GridFunc<ScalarType>*> gf_rcoarse_;
    std::vector<pb::GridFunc<ScalarType>*> gf_newv_;
    // solver params
    int maxiters_;
    double tol_;
    double final_residual_;
    double residual_reduction_;
    // precon params
    short nu1_;
    short nu2_;
    short max_nlevels_;
    short nlevels_;
    void setupPrecon();

    void clear();

    void preconSolve(pb::GridFunc<ScalarType>& gf_v,
        const pb::GridFunc<ScalarType>& gf_f, const short level = 0);

public:
    PCGSolver_Diel(T& oper, const pb::LapType lap_type, const short px,
        const short py, const short pz)
        : oper_(oper), lap_type_(lap_type)
    {
        maxiters_           = 10; // default
        nu1_                = 2; // default
        nu2_                = 2; // default
        tol_                = 1.e-16;
        max_nlevels_        = 10;
        final_residual_     = -1.;
        residual_reduction_ = -1.;

        // boundary conditions
        bc_[0] = px;
        bc_[1] = py;
        bc_[2] = pz;
    };

    void setup(const short nu1, const short nu2, const short max_sweeps,
        const double tol, const short max_nlevels)
    {
        maxiters_    = max_sweeps;
        nu1_         = nu1;
        nu2_         = nu2;
        tol_         = tol;
        max_nlevels_ = max_nlevels;
    }

    bool solve(
        pb::GridFunc<ScalarType>& gf_phi, pb::GridFunc<ScalarType>& gf_rhs);

    bool solve(pb::GridFunc<ScalarType>& gf_phi,
        pb::GridFunc<ScalarType>& gf_rhs, pb::GridFunc<ScalarType>& gf_rhod,
        pb::GridFunc<ScalarType>& gf_vks);

    double getFinalResidual() const { return final_residual_; }
    double getResidualReduction() const { return residual_reduction_; }

    // Destructor
    ~PCGSolver_Diel() { clear(); }
};

#endif

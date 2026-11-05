// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#ifndef MGMOL_PCG_SOLVER_H
#define MGMOL_PCG_SOLVER_H

#include "Lap.h"
#include "Poisson.h"

#include <vector>

template <class OperatorType, typename ScalarDataType, typename PrecondDataType>
class PCGSolver
{
private:
    std::vector<pb::Grid*> grid_;
    short bc_[3];
    bool fully_periodic_;

    // operator to solve for
    OperatorType oper_;

    const PoissonFDtype precond_lap_type_;

    // preconditioner operator for each MG level
    std::vector<pb::Lap<PrecondDataType>*> precond_oper_;
    std::vector<pb::GridFunc<PrecondDataType>*> gf_work_;
    std::vector<pb::GridFunc<PrecondDataType>*> gf_rcoarse_;
    std::vector<pb::GridFunc<PrecondDataType>*> gf_newv_;

    // solver parameters
    int maxiters_;
    double tol_;
    double final_residual_;
    double residual_reduction_;

    // preconditioner parameters
    short nu1_;
    short nu2_;
    short max_nlevels_;
    short nlevels_;
    bool is_precond_setup_;

    void preconSolve(pb::GridFunc<PrecondDataType>& gf_v,
        const pb::GridFunc<PrecondDataType>& gf_f, const short level = 0);
    void setupPrecon();
    void clear();

public:
    PCGSolver(OperatorType& oper, const PoissonFDtype precond_lap_type,
        const short px, const short py, const short pz)
        : oper_(oper),
          precond_lap_type_(precond_lap_type),
          maxiters_(10),
          tol_(1.e-16),
          final_residual_(-1.),
          residual_reduction_(-1.),
          nu1_(2),
          nu2_(2),
          max_nlevels_(10),
          is_precond_setup_(false)
    {
        // boundary conditions
        bc_[0]          = px;
        bc_[1]          = py;
        bc_[2]          = pz;
        fully_periodic_ = ((bc_[0] == 1) && (bc_[1] == 1) && (bc_[2] == 1));
    };

    void setup(const short nu1, const short nu2, const short max_sweeps,
        const double tol, const short max_nlevels)
    {
        maxiters_    = max_sweeps;
        nu1_         = nu1;
        nu2_         = nu2;
        tol_         = tol;
        max_nlevels_ = max_nlevels;
        setupPrecon();
    }

    bool solve(pb::GridFunc<ScalarDataType>& gf_phi,
        const pb::GridFunc<ScalarDataType>& gf_rhs);

    /*!
     * Interface for raw pointers
     */
    bool solve(ScalarDataType* phi, ScalarDataType* rhs, const char dis);

    double getFinalResidual() const { return final_residual_; }
    double getResidualReduction() const { return residual_reduction_; }

    // Destructor
    ~PCGSolver() { clear(); }
};

#endif

// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#ifndef MGMOL_ELECTROSTATIC_H
#define MGMOL_ELECTROSTATIC_H

#include "Control.h"
#include "GridFunc.h"
#include "Poisson.h"
#include "PoissonSolverFactory.h"
#include "Rho.h"
#include "Timer.h"

class Ions;
class Potentials;

class Electrostatic
{
    PoissonFDtype laptype_;
    short bc_[3];
    pb::Grid* pbGrid_;
    bool diel_flag_;

    Poisson* poisson_solver_;

    double Evh_rho_;
    double Evh_rhoc_;
    double Evhold_rho_;
    double eepsilon_;

    pb::GridFunc<RHODTYPE>* grhoc_;
    pb::GridFunc<RHODTYPE>* grhod_;

    int iterative_index_;

    static Timer solve_tm_;

public:
    Electrostatic(PoissonFDtype lap_type, const short bc[3],
        const double screening_const = 0.);
    ~Electrostatic();
    static Timer solve_tm() { return solve_tm_; }

    const bool isDielectric() { return diel_flag_; }
    pb::GridFunc<RHODTYPE>* getRhoc() { return grhoc_; }
    Poisson* getPoissonSolver() { return poisson_solver_; }

    void setup(const short max_sweeps);
    void setupPB(const double e0, const double rho0, const double drho0,
        Potentials& pot);

    void setupRhoc(RHODTYPE* rhoc);
    void fillFuncAroundIons(const Ions& ions);
    template <class T>
    void computeVh(const Ions& ions, Rho<T>& rho, Potentials& pot);
    template <class T>
    void computeVh(const pb::GridFunc<POTDTYPE>& vhinit, const Ions& ions,
        Rho<T>& rho, Potentials& pot);
    void setupInitialVh(const std::vector<POTDTYPE>&);
    void setupInitialVh(const pb::GridFunc<POTDTYPE>&);
    template <class T>
    void computeVhRho(Rho<T>& rho);
    void resetSolution() { poisson_solver_->resetVh(); }

    const pb::GridFunc<POTDTYPE>& getVh() const;

    double evhRho() const { return Evh_rho_; }
    double evhRhoc() const { return Evh_rhoc_; }
    double eEpsilon() const { return eepsilon_; }
};

#endif

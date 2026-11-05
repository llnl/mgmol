// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#ifndef MGMOL_HARTREE_CG_H
#define MGMOL_HARTREE_CG_H

#include "Control.h"
#include "PCGSolver.h"
#include "Poisson.h"

#include <memory>

template <class OperatorType, typename ScalarType, typename PDataType>
class Hartree_CG : public Poisson
{
private:
    std::shared_ptr<PCGSolver<OperatorType, ScalarType, PDataType>>
        poisson_solver_;

public:
    // Constructor
    Hartree_CG(const pb::Grid& grid, const short bc[3]) : Poisson(grid, bc)
    {
        OperatorType oper(Poisson::grid_);
        Control& ct = *(Control::instance());
        poisson_solver_
            = std::make_shared<PCGSolver<OperatorType, ScalarType, PDataType>>(
                oper, ct.getPoissonFDtype(), bc[0], bc[1], bc[2]);
    };

    // Destructor
    ~Hartree_CG() override { }

    void setup(const short nu1, const short nu2, const short max_sweeps,
        const double tol, const short max_nlevels,
        const bool gather_coarse_level = true) override
    {
        (void)gather_coarse_level;
        poisson_solver_->setup(nu1, nu2, max_sweeps, tol, max_nlevels);
    }

    void solve(const pb::GridFunc<ScalarType>& rho,
        const pb::GridFunc<ScalarType>& rhoc) override;
};

#endif

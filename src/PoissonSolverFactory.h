// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE
#ifndef MGMOL_PoissonSolverFactory
#define MGMOL_PoissonSolverFactory

#include "Control.h"
#include "Hartree.h"
#include "Hartree_CG.h"
#include "Laph2.h"
#include "Laph4.h"
#include "Laph4M.h"
#include "Laph4MP.h"
#include "Laph6.h"
#include "Laph8.h"
#include "MGmol_MPI.h"
#include "Mesh.h"
#include "PBdiel.h"
#include "PBdiel_CG.h"
#include "ShiftedHartree.h"
#include "ShiftedLaph4M.h"

/*!
 * Create Hartree_CG solver templated on data type and preconditioner data type
 */
template <typename ScalarType, typename PDataType>
Poisson* createHartreeCG(
    pb::LapType lap_type, const pb::Grid& myGrid, const short bc[3])
{
    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    if (mmpi.instancePE0())
    {
        std::cout << "create HartreeCG with precision "
                  << 8 * sizeof(ScalarType) << std::endl;
        std::cout << "HartreeCG with preconditioner in precision "
                  << 8 * sizeof(PDataType) << std::endl;
    }
    Poisson* poisson_solver = nullptr;

    switch (lap_type)
    {
        case pb::LapType::Mehrstellen1:
            poisson_solver
                = new Hartree_CG<pb::Laph4M<ScalarType>, ScalarType, PDataType>(
                    myGrid, bc);
            break;
        case pb::LapType::FD2nd:
            poisson_solver
                = new Hartree_CG<pb::Laph2<ScalarType>, ScalarType, PDataType>(
                    myGrid, bc);
            break;
        case pb::LapType::FD4th:
            poisson_solver
                = new Hartree_CG<pb::Laph4<ScalarType>, ScalarType, PDataType>(
                    myGrid, bc);
            break;
        case pb::LapType::FD6th:
            poisson_solver
                = new Hartree_CG<pb::Laph6<ScalarType>, ScalarType, PDataType>(
                    myGrid, bc);
            break;
        case pb::LapType::FD8th:
            poisson_solver
                = new Hartree_CG<pb::Laph8<ScalarType>, ScalarType, PDataType>(
                    myGrid, bc);
            break;
        case pb::LapType::Mehrstellen2:
            poisson_solver = new Hartree_CG<pb::Laph4MP<ScalarType>, ScalarType,
                PDataType>(myGrid, bc);
            break;
        default:
            std::cerr << "createHartreeCG(), Undefined option: "
                      << static_cast<int>(lap_type) << std::endl;
    }
    return poisson_solver;
}

/*!
 * Main factory
 */
class PoissonSolverFactory
{
public:
    /*!
     * return specific Poisson solver needed to solve Hartree problem
     */
    static Poisson* create(const pb::Grid& myGrid, pb::LapType lap_type,
        const short bc[3], const double screening_const)
    {
        Poisson* poisson_solver = nullptr;

        Control& ct = *(Control::instance());
        if (ct.MGPoissonSolver()) // use MG for Poisson Solver
        {
            if (screening_const > 0.)
            {
                switch (lap_type)
                {
                    case pb::LapType::Mehrstellen1:
                        poisson_solver
                            = new ShiftedHartree<pb::ShiftedLaph4M<double>>(
                                myGrid, bc, screening_const);
                        break;
                    default:
                        std::cerr << "PoissonSolverFactory, shifted, Undefined "
                                     "option: "
                                  << static_cast<int>(lap_type) << std::endl;
                }
            }
            else
            {
                switch (lap_type)
                {
                    case pb::LapType::Mehrstellen1:
                        poisson_solver
                            = new Hartree<pb::Laph4M<double>>(myGrid, bc);
                        break;
                    case pb::LapType::FD2nd:
                        poisson_solver
                            = new Hartree<pb::Laph2<double>>(myGrid, bc);
                        break;
                    case pb::LapType::FD4th:
                        poisson_solver
                            = new Hartree<pb::Laph4<double>>(myGrid, bc);
                        break;
                    case pb::LapType::FD6th:
                        poisson_solver
                            = new Hartree<pb::Laph6<double>>(myGrid, bc);
                        break;
                    case pb::LapType::FD8th:
                        poisson_solver
                            = new Hartree<pb::Laph8<double>>(myGrid, bc);
                        break;
                    case pb::LapType::Mehrstellen2:
                        poisson_solver
                            = new Hartree<pb::Laph4MP<double>>(myGrid, bc);
                        break;
                    default:
                        std::cerr << "PoissonSolverFactory, Undefined option: "
                                  << static_cast<int>(lap_type) << std::endl;
                }
            }
        }
        else // use PCG for Poisson Solver
        {
            if (screening_const > 0.)
            {
                switch (lap_type)
                {
                    case pb::LapType::Mehrstellen1:
                        poisson_solver
                            = new ShiftedHartree<pb::ShiftedLaph4M<double>>(
                                myGrid, bc, screening_const);
                        break;
                    default:
                        std::cerr
                            << "PoissonSolverFactory, with screening_const, "
                               "Undefined option: "
                            << static_cast<int>(lap_type) << std::endl;
                }
            }
            else
            {
                const short precision = ct.poisson_pc_data_;
                if (precision == 32)
                    poisson_solver
                        = createHartreeCG<double, float>(lap_type, myGrid, bc);
                else if (precision == 64)
                    poisson_solver
                        = createHartreeCG<double, double>(lap_type, myGrid, bc);
                else
                {
                    std::cerr
                        << "PoissonSolverFactory: Unknown precision option "
                        << precision << std::endl;
                }
            }
        }

        return poisson_solver;
    }

    static Poisson* createDiel(pb::Grid& pbGrid, pb::LapType lap_type,
        const short bc[3], const double e0, const double rho0,
        const double drho0)
    {
        Poisson* poisson_solver = nullptr;

        Control& ct = *(Control::instance());
        if (ct.MGPoissonSolver()) // use MG for Poisson Solver
        {
            switch (lap_type)
            {
                case pb::LapType::Mehrstellen1:
                    poisson_solver = new PBdiel<pb::PBh4M<double>>(
                        pbGrid, bc, e0, rho0, drho0);
                    break;
                case pb::LapType::FD2nd:
                    poisson_solver = new PBdiel<pb::PBh2<double>>(
                        pbGrid, bc, e0, rho0, drho0);
                    break;
                case pb::LapType::FD4th:
                    poisson_solver = new PBdiel<pb::PBh4<double>>(
                        pbGrid, bc, e0, rho0, drho0);
                    break;
                case pb::LapType::FD6th:
                    poisson_solver = new PBdiel<pb::PBh6<double>>(
                        pbGrid, bc, e0, rho0, drho0);
                    break;
                case pb::LapType::FD8th:
                    poisson_solver = new PBdiel<pb::PBh8<double>>(
                        pbGrid, bc, e0, rho0, drho0);
                    break;
                case pb::LapType::Mehrstellen2:
                    poisson_solver = new PBdiel<pb::PBh4MP<double>>(
                        pbGrid, bc, e0, rho0, drho0);
                    break;
                default:
                    std::cerr << "createDiel(), Undefined option" << std::endl;
            }
        }
        else // use PCG for Poisson Solver
        {
            switch (lap_type)
            {
                case pb::LapType::Mehrstellen1:
                    poisson_solver = new PBdiel_CG<pb::PBh4M<double>>(
                        pbGrid, bc, e0, rho0, drho0);
                    break;
                case pb::LapType::FD2nd:
                    poisson_solver = new PBdiel_CG<pb::PBh2<double>>(
                        pbGrid, bc, e0, rho0, drho0);
                    break;
                case pb::LapType::FD4th:
                    poisson_solver = new PBdiel_CG<pb::PBh4<double>>(
                        pbGrid, bc, e0, rho0, drho0);
                    break;
                case pb::LapType::FD6th:
                    poisson_solver = new PBdiel_CG<pb::PBh6<double>>(
                        pbGrid, bc, e0, rho0, drho0);
                    break;
                case pb::LapType::FD8th:
                    poisson_solver = new PBdiel_CG<pb::PBh8<double>>(
                        pbGrid, bc, e0, rho0, drho0);
                    break;
                case pb::LapType::Mehrstellen2:
                    poisson_solver = new PBdiel_CG<pb::PBh4MP<double>>(
                        pbGrid, bc, e0, rho0, drho0);
                    break;
                default:
                    std::cerr << "createDiel(), Undefined option" << std::endl;
            }
        }
        return poisson_solver;
    }
};

#endif

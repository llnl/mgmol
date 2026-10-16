// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "ExtendedGridOrbitals.h"
#include "Ions.h"
#include "LocGridOrbitals.h"
#include "ProjectedMatrices.h"
#include "ReplicatedMatrix.h"
#include "XYZOps.h"

#include <vector>

template <class OrbitalsType, class MatrixType>
void computeDipoleTensors(const OrbitalsType& orbitals, MatrixType& Dx,
    MatrixType& Dy, MatrixType& Dz)
{
    const int n = orbitals.numst();
    std::vector<std::vector<double>> xyz(3);
    for (int i = 0; i < 3; i++)
        xyz[i].resize(n * n);

    // compute the local contributions to the Dipole matrix
    XYZOps<OrbitalsType>::compute(orbitals, xyz);

    Dx.assign(xyz[0].data(), n);
    Dy.assign(xyz[1].data(), n);
    Dz.assign(xyz[2].data(), n);
}

template <class OrbitalsType, class MatrixType>
Vector3D computeDipoleMomentVector(const OrbitalsType& orbitals, Ions& ions,
    ProjectedMatrices<MatrixType>& projmatrices, bool print_stats=true)
{
    Control& ct = *(Control::instance());
    MGmol_MPI& mmpi(*(MGmol_MPI::instance()));

    // first compute electronic contribution (negative)
    ReplicatedMatrix Dx("Dx", orbitals.numst());
    ReplicatedMatrix Dy("Dy", orbitals.numst());
    ReplicatedMatrix Dz("Dz", orbitals.numst());

    computeDipoleTensors(orbitals, Dx, Dy, Dz);

    double dx = -1. * projmatrices.getExpectation(Dx);
    double dy = -1. * projmatrices.getExpectation(Dy);
    double dz = -1. * projmatrices.getExpectation(Dz);
    if (mmpi.instancePE0()) std::cout << std::endl;
    if (print_stats && mmpi.instancePE0() && ct.verbose > 0)
        std::cout << "Electronic contribution to dipole = " << dx << ", " << dy
                  << ", " << dz << std::endl;

    // add contribution from the ions (positive)
    double di[3] = { 0., 0., 0. };
    const std::vector<Ion*>& local_ions(ions.local_ions());
    for (const auto& ion : local_ions)
    {
        const double charge = ion->getZion();
        const double x      = ion->position(0);
        const double y      = ion->position(1);
        const double z      = ion->position(2);

        di[0] += charge * x;
        di[1] += charge * y;
        di[2] += charge * z;
    }

    mmpi.allreduce(&di[0], 3, MPI_SUM);
    if (print_stats && mmpi.instancePE0() && ct.verbose > 0)
        std::cout << "Ionic contribution to dipole      = " << di[0] << ", "
                  << di[1] << ", " << di[2] << std::endl;

    dx += di[0];
    dy += di[1];
    dz += di[2];

    if (print_stats && mmpi.instancePE0())
    {
        std::cout << "Dipole moment                     = " << dx << ", " << dy
                  << ", " << dz << " a.u." << std::endl;
        const double au2D = 2.542;
        std::cout << "Dipole moment                     = " << dx * au2D << ", "
                  << dy * au2D << ", " << dz * au2D << " Debye" << std::endl
                  << std::endl;
    }

    return Vector3D(dx, dy, dz);
}

template <class OrbitalsType, class MatrixType>
void computeDipoleMoment(const OrbitalsType& orbitals, Ions& ions,
    ProjectedMatrices<MatrixType>& projmatrices)
{
    computeDipoleMomentVector(orbitals, ions, projmatrices);
}

template void
computeDipoleMoment<ExtendedGridOrbitals<ORBDTYPE>, ReplicatedMatrix>(
    const ExtendedGridOrbitals<ORBDTYPE>& orbitals, Ions& ions,
    ProjectedMatrices<ReplicatedMatrix>&);
template void computeDipoleMoment<LocGridOrbitals<ORBDTYPE>, ReplicatedMatrix>(
    const LocGridOrbitals<ORBDTYPE>& orbitals, Ions& ions,
    ProjectedMatrices<ReplicatedMatrix>&);
template Vector3D
computeDipoleMomentVector<ExtendedGridOrbitals<ORBDTYPE>, ReplicatedMatrix>(
    const ExtendedGridOrbitals<ORBDTYPE>& orbitals, Ions& ions,
    ProjectedMatrices<ReplicatedMatrix>&, bool print_stats);
template Vector3D computeDipoleMomentVector<LocGridOrbitals<ORBDTYPE>, ReplicatedMatrix>(
    const LocGridOrbitals<ORBDTYPE>& orbitals, Ions& ions,
    ProjectedMatrices<ReplicatedMatrix>&, bool print_stats);
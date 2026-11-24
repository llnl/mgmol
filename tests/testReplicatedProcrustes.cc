// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE
#include "ReplicatedMatrix.h"

#include "catch.hpp"

#include <cmath>
#include <iostream>
#include <mpi.h>
#include <vector>

namespace
{
const int n                     = 4;
const std::vector<double> sigma = { 3., 2., 0.7, 0.3 };

// n x n matrices stored in column-major order
typedef std::vector<double> Mat;

void print(std::string name, const Mat& a)
{
    printf("%s\n", name.c_str());
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            printf("  %le", a[i + n * j]);
        printf("\n");
    }
}

Mat identity(const int n)
{
    Mat a(n * n, 0.);
    for (int i = 0; i < n; i++)
        a[i + n * i] = 1.;
    return a;
}

// rotation of angle "angle" in the plane (i,j)
Mat givens(const int n, const int i, const int j, const double angle)
{
    Mat a        = identity(n);
    a[i + n * i] = std::cos(angle);
    a[j + n * j] = std::cos(angle);
    a[i + n * j] = -std::sin(angle);
    a[j + n * i] = std::sin(angle);
    return a;
}

Mat multiply(const int n, const Mat& a, const Mat& b)
{
    Mat c(n * n, 0.);
    for (int j = 0; j < n; j++)
        for (int k = 0; k < n; k++)
            for (int i = 0; i < n; i++)
                c[i + n * j] += a[i + n * k] * b[k + n * j];
    return c;
}

Mat transpose(const int n, const Mat& a)
{
    Mat t(n * n);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            t[j + n * i] = a[i + n * j];
    return t;
}

Mat diagonal(const int n, const std::vector<double>& d)
{
    Mat a(n * n, 0.);
    for (int i = 0; i < n; i++)
        a[i + n * i] = d[i];
    return a;
}

double maxDiff(const Mat& a, const Mat& b)
{
    double diff = 0.;
    for (size_t i = 0; i < a.size(); i++)
        diff = std::max(diff, std::fabs(a[i] - b[i]));
    return diff;
}

// orthogonal matrices used to build q = u * diag(sigma) * v^T
Mat getU()
{
    return multiply(n, givens(n, 0, 1, 0.3),
        multiply(n, givens(n, 2, 3, 0.5), givens(n, 1, 2, 0.7)));
}

Mat getV()
{
    return multiply(n, givens(n, 0, 2, 0.4),
        multiply(n, givens(n, 1, 3, 0.9), givens(n, 0, 1, 0.2)));
}
} // namespace

TEST_CASE("Check singular value decomposition of ReplicatedMatrix",
    "[replicated_gesvd]")
{
    ReplicatedMatrix::setMPIcomm(MPI_COMM_WORLD);

    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    const Mat u = getU();
    const Mat v = getV();
    const Mat q
        = multiply(n, u, multiply(n, diagonal(n, sigma), transpose(n, v)));

    ReplicatedMatrix a("A", n);
    a.init(q.data(), n);
    ReplicatedMatrix uo("U", n);
    ReplicatedMatrix vto("VT", n);

    std::vector<double> s(n);
    a.gesvd('V', 'V', s, uo, vto);

    if (mpi_rank == 0) printf("Computed singular values:\n");
    for (int i = 0; i < n; i++)
    {
        if (mpi_rank == 0) printf("s = %le\n", s[i]);
        CHECK(s[i] == Approx(sigma[i]).epsilon(1.e-10));
    }

    Mat uc(n * n);
    Mat vtc(n * n);
    uo.get(uc.data(), n);
    vto.get(vtc.data(), n);

    // u * diag(s) * vt reproduces the original matrix
    const Mat rebuilt = multiply(n, uc, multiply(n, diagonal(n, s), vtc));
    CHECK(maxDiff(rebuilt, q) < 1.e-10);

    // u and vt are orthogonal
    CHECK(maxDiff(multiply(n, transpose(n, uc), uc), identity(n)) < 1.e-10);
    CHECK(maxDiff(multiply(n, vtc, transpose(n, vtc)), identity(n)) < 1.e-10);
}

TEST_CASE(
    "Check Procrustes transform of ReplicatedMatrix", "[replicated_procrustes]")
{
    ReplicatedMatrix::setMPIcomm(MPI_COMM_WORLD);

    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    // build two orthonormal matrices
    const Mat u1 = givens(n, 0, 1, 0.2);
    const Mat u2 = givens(n, 1, 3, 0.7);

    const Mat q = multiply(n, transpose(n, u2), u1);
    ReplicatedMatrix a("a", n);
    a.init(q.data(), n);

    // replace a with its Procrustes transform
    ReplicatedMatrix yyt("yyt", n);
    getProcrustesTransform(a, yyt);

    // verify that u2 * a == u1
    Mat qc(n * n);
    a.get(qc.data(), n);
    Mat u2p(n * n);
    u2p = multiply(n, u2, qc);

    if (mpi_rank == 0)
    {
        print("u1", u1);
        print("u2", u2);
        print("u2p", u2p);
    }
    fflush(stdout);
    std::cout.flush();

    CHECK(maxDiff(u1, u2p) < 1.e-10);
}

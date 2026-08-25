// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

// Adapted from D3vector.h,v 1.11 2002/06/26 22:53:45 fgygi

#ifndef MGMOL_Vector3D_H
#define MGMOL_Vector3D_H

#include <cassert>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <vector>

const double tol = 1.e-12;

class Vector3D
{
    double x_[3];

public:
    // explicit constructor to avoid implicit conversion from double to Vector3D
    explicit Vector3D(
        const double xv = 0, const double yv = 0, const double zv = 0)
    {
        x_[0] = xv;
        x_[1] = yv;
        x_[2] = zv;
    }

    Vector3D(const Vector3D& v)
    {
        x_[0] = v.x_[0];
        x_[1] = v.x_[1];
        x_[2] = v.x_[2];
    }

    Vector3D& operator=(const Vector3D& v)
    {
        x_[0] = v.x_[0];
        x_[1] = v.x_[1];
        x_[2] = v.x_[2];

        return *this;
    }

    Vector3D sign()
    {
        Vector3D result;

        for (int i = 0; i < 3; i++)
        {
            result[i] = (double)std::signbit(x_[i]);
            result[i] = (result[i] * -1. + 0.5) * 2.;
        }

        return result;
    }

    void assign(const double xv, const double yv, const double zv)
    {
        x_[0] = xv;
        x_[1] = yv;
        x_[2] = zv;
    }

    double& operator[](const short i) { return x_[i]; }

    const double& operator[](const short i) const { return x_[i]; }

    bool operator==(const Vector3D& rhs) const
    {
        return std::abs(x_[0] - rhs.x_[0]) < tol
               && std::abs(x_[1] - rhs.x_[1]) < tol
               && std::abs(x_[2] - rhs.x_[2]) < tol;
    }

    bool operator!=(const Vector3D& rhs) const
    {
        return std::abs(x_[0] - rhs.x_[0]) >= tol
               || std::abs(x_[1] - rhs.x_[1]) >= tol
               || std::abs(x_[2] - rhs.x_[2]) >= tol;
    }

    bool operator<(const Vector3D& rhs) const
    {
        return (10000. * x_[0] + 100. * x_[1] + x_[2]
                < 10000. * rhs.x_[0] + 100. * rhs.x_[1] + rhs.x_[2]);
    }

    Vector3D& operator+=(const Vector3D& rhs)
    {
        x_[0] += rhs.x_[0];
        x_[1] += rhs.x_[1];
        x_[2] += rhs.x_[2];
        return *this;
    }

    Vector3D& axpy(const double alpha, const Vector3D& rhs)
    {
        x_[0] += alpha * rhs.x_[0];
        x_[1] += alpha * rhs.x_[1];
        x_[2] += alpha * rhs.x_[2];
        return *this;
    }

    Vector3D& operator-=(const Vector3D& rhs)
    {
        x_[0] -= rhs.x_[0];
        x_[1] -= rhs.x_[1];
        x_[2] -= rhs.x_[2];
        return *this;
    }

    Vector3D& operator*=(const double& rhs)
    {
        x_[0] *= rhs;
        x_[1] *= rhs;
        x_[2] *= rhs;
        return *this;
    }

    Vector3D& operator*=(const Vector3D& rhs)
    {
        x_[0] *= rhs.x_[0];
        x_[1] *= rhs.x_[1];
        x_[2] *= rhs.x_[2];
        return *this;
    }

    Vector3D& operator/=(const double& rhs)
    {
        const double alpha = 1. / rhs;
        x_[0] *= alpha;
        x_[1] *= alpha;
        x_[2] *= alpha;
        return *this;
    }

    void sub_and_assign(const Vector3D& a, const Vector3D& b)
    {
        x_[0] = a.x_[0] - b.x_[0];
        x_[1] = a.x_[1] - b.x_[1];
        x_[2] = a.x_[2] - b.x_[2];
    }

    double minimage(
        const Vector3D& w, const Vector3D& l, const short bc[3]) const;
    Vector3D vminimage(
        const Vector3D& w, const Vector3D& l, const short bc[3]) const;

    friend const Vector3D operator+(const Vector3D& lhs, const Vector3D& rhs)
    {
        return Vector3D(lhs) += rhs;
    }

    friend const Vector3D operator-(const Vector3D& a, const Vector3D& b)
    {
        return Vector3D(a) -= b;
    }

    friend Vector3D operator-(const Vector3D& a) // unary minus
    {
        return Vector3D(-a.x_[0], -a.x_[1], -a.x_[2]);
    }

    friend Vector3D operator*(const double& a, const Vector3D& b)
    {
        return Vector3D(b) *= a;
    }

    friend Vector3D operator*(const Vector3D& a, const double& b)
    {
        return Vector3D(a) *= b;
    }

    friend Vector3D operator/(const Vector3D& a, const double& b)
    {
        return Vector3D(a) /= b;
    }

    // scalar product
    friend double operator*(const Vector3D& a, const Vector3D& b)
    {
        return a.x_[0] * b.x_[0] + a.x_[1] * b.x_[1] + a.x_[2] * b.x_[2];
    }

    friend Vector3D operator^(const Vector3D& a, const Vector3D& b)
    {
        return Vector3D(a.x_[1] * b.x_[2] - a.x_[2] * b.x_[1],
            a.x_[2] * b.x_[0] - a.x_[0] * b.x_[2],
            a.x_[0] * b.x_[1] - a.x_[1] * b.x_[0]);
    }

    friend Vector3D rotate(const Vector3D& x, const Vector3D& w)
    {
        if (length(x) < tol) return x; // x has zero length
        double theta = length(w); // rotate by zero
        if (theta < tol) return x;
        Vector3D ew = normalized(w);
        Vector3D v  = w ^ x;
        if (length(v) < tol) return x; // x is parallel to the rotation axis
        v          = normalized(v);
        Vector3D u = v ^ ew;
        double p   = x * u;
        return (x * ew) * ew + p * cos(theta) * u + p * sin(theta) * v;
    }

    friend double length(const Vector3D& a)
    {
        return sqrt(a.x_[0] * a.x_[0] + a.x_[1] * a.x_[1] + a.x_[2] * a.x_[2]);
    }

    friend double norm2(const Vector3D& a)
    {
        return a.x_[0] * a.x_[0] + a.x_[1] * a.x_[1] + a.x_[2] * a.x_[2];
    }

    friend Vector3D normalized(const Vector3D& a) 
    { 
        double len = length(a);
        assert(len != 0); // should we care about near zero vectors here? - DOK
        return a / len; 
    }

    friend std::ostream& operator<<(std::ostream& os, const Vector3D& v)
    {
        os << v.x_[0] << " " << v.x_[1] << " " << v.x_[2];
        return os;
    }

    friend std::istream& operator>>(std::istream& is, Vector3D& v)
    {
        is >> v.x_[0] >> v.x_[1] >> v.x_[2];
        return is;
    }

    void bcast(MPI_Comm comm) { MPI_Bcast(&x_[0], 3, MPI_DOUBLE, 0, comm); }

    friend void bcastvv3d(std::vector<Vector3D>& vv, MPI_Comm comm);
};
#endif

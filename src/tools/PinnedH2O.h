// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#ifndef PINNED_H2O_H
#define PINNED_H2O_H

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

class PinnedH2O
{

public:
    PinnedH2O();
    ~PinnedH2O() = default;

    void rotate(std::vector<double>& positions, std::vector<short>& anumbers);
    void transpose_rotate(std::vector<double>& positions,
        std::vector<short>& anumbers, std::vector<double>& forces);
    void print(std::ostream& os)
    {
        os << "Bondlengths = " << bondlength1 << ", " << bondlength2
           << " Bohrs; "
           << "Bondangle = " << bondangle * 180.0 / M_PI << " degrees."
           << std::endl;
    }

private:
    double calculate_bondlength(
        const double atom1[3], const double atom2[3]) const;
    double calculate_bondangle(const double atom1[3], const double atom2[3],
        const double atom3[3]) const;
    void rotation_matrix(
        const double axis[3], double angle, double matrix[3][3]) const;
    void normalize(double vec[3]) const;
    void cross(const double a[3], const double b[3], double result[3]) const;
    void apply_rotation(
        const double matrix[3][3], const double vec[3], double result[3]) const;
    void apply_transpose_rotation(
        const double matrix[3][3], const double vec[3], double result[3]) const;

    double bondlength1;
    double bondlength2;
    double bondangle;

    bool flipped_bond;
    int O1_idx;
    int H1_idx;
    int H2_idx;
    double planar_rotation_angle;
    double out_of_plane_rotation_matrix[3][3];
};

#endif // PINNED_H2O_H

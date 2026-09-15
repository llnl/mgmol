// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "PinnedH2O.h"
#include <iostream>

PinnedH2O::PinnedH2O()
    : flipped_bond(false),
      O1_idx(-1),
      H1_idx(-1),
      H2_idx(-1),
      planar_rotation_angle(0.0)
{
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            out_of_plane_rotation_matrix[i][j] = 0.0;
        }
    }
}

double PinnedH2O::calculate_bondlength(
    const double atom1[3], const double atom2[3]) const
{
    return sqrt(pow(atom1[0] - atom2[0], 2) + pow(atom1[1] - atom2[1], 2)
                + pow(atom1[2] - atom2[2], 2));
}

double PinnedH2O::calculate_bondangle(
    const double atom1[3], const double atom2[3], const double atom3[3]) const
{
    double vector1[3]
        = { atom1[0] - atom2[0], atom1[1] - atom2[1], atom1[2] - atom2[2] };
    double vector2[3]
        = { atom3[0] - atom2[0], atom3[1] - atom2[1], atom3[2] - atom2[2] };

    double dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
                         + vector1[2] * vector2[2];
    double magnitude_product
        = sqrt(pow(vector1[0], 2) + pow(vector1[1], 2) + pow(vector1[2], 2))
          * sqrt(pow(vector2[0], 2) + pow(vector2[1], 2) + pow(vector2[2], 2));
    double angle = acos(dot_product / magnitude_product);

    return angle;
}

void PinnedH2O::rotation_matrix(
    const double axis[3], double angle, double matrix[3][3]) const
{
    double cos_theta = cos(angle);
    double sin_theta = sin(angle);
    double ux = axis[0], uy = axis[1], uz = axis[2];

    matrix[0][0] = cos_theta + ux * ux * (1 - cos_theta);
    matrix[0][1] = ux * uy * (1 - cos_theta) - uz * sin_theta;
    matrix[0][2] = ux * uz * (1 - cos_theta) + uy * sin_theta;

    matrix[1][0] = uy * ux * (1 - cos_theta) + uz * sin_theta;
    matrix[1][1] = cos_theta + uy * uy * (1 - cos_theta);
    matrix[1][2] = uy * uz * (1 - cos_theta) - ux * sin_theta;

    matrix[2][0] = uz * ux * (1 - cos_theta) - uy * sin_theta;
    matrix[2][1] = uz * uy * (1 - cos_theta) + ux * sin_theta;
    matrix[2][2] = cos_theta + uz * uz * (1 - cos_theta);
}

void PinnedH2O::normalize(double vec[3]) const
{
    double norm = sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
    vec[0] /= norm;
    vec[1] /= norm;
    vec[2] /= norm;
}

void PinnedH2O::cross(
    const double a[3], const double b[3], double result[3]) const
{
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
}

void PinnedH2O::apply_rotation(
    const double matrix[3][3], const double vec[3], double result[3]) const
{
    result[0]
        = matrix[0][0] * vec[0] + matrix[0][1] * vec[1] + matrix[0][2] * vec[2];
    result[1]
        = matrix[1][0] * vec[0] + matrix[1][1] * vec[1] + matrix[1][2] * vec[2];
    result[2]
        = matrix[2][0] * vec[0] + matrix[2][1] * vec[1] + matrix[2][2] * vec[2];
}

void PinnedH2O::apply_transpose_rotation(
    const double matrix[3][3], const double vec[3], double result[3]) const
{
    result[0]
        = matrix[0][0] * vec[0] + matrix[1][0] * vec[1] + matrix[2][0] * vec[2];
    result[1]
        = matrix[0][1] * vec[0] + matrix[1][1] * vec[1] + matrix[2][1] * vec[2];
    result[2]
        = matrix[0][2] * vec[0] + matrix[1][2] * vec[1] + matrix[2][2] * vec[2];
}

void PinnedH2O::rotate(
    std::vector<double>& positions, std::vector<short>& anumbers)
{
    for (int i = 0; i < 3; i++)
    {
        if (positions[3 * i] == 0.0 && positions[3 * i + 1] == 0.0
            && positions[3 * i + 2] == 0.0)
        {
            O1_idx = i;
            break;
        }
    }
    if (O1_idx == -1) return;

    H1_idx = (O1_idx + 1) % 3;
    H2_idx = (O1_idx + 2) % 3;

    double O1[3] = { positions[3 * O1_idx], positions[3 * O1_idx + 1],
        positions[3 * O1_idx + 2] };
    double H1[3] = { positions[3 * H1_idx], positions[3 * H1_idx + 1],
        positions[3 * H1_idx + 2] };
    double H2[3] = { positions[3 * H2_idx], positions[3 * H2_idx + 1],
        positions[3 * H2_idx + 2] };

    bondlength1 = calculate_bondlength(H1, O1);
    bondlength2 = calculate_bondlength(H2, O1);
    bondangle   = calculate_bondangle(H1, O1, H2);

    double H1_temp[3], H2_temp[3];
    double H1_rotated[3], H2_rotated[3];

    double plane_normal[3];
    cross(H2, H1, plane_normal);
    normalize(plane_normal);
    double target_plane_normal[3] = { 0, 0, 1 };
    double dot_product            = plane_normal[0] * target_plane_normal[0]
                         + plane_normal[1] * target_plane_normal[1]
                         + plane_normal[2] * target_plane_normal[2];
    double angle_to_align
        = std::acos(std::min(std::max(dot_product, -1.0), 1.0));
    double axis_to_align[3];
    if (abs(dot_product) > 1.0 - 1e-8)
    {
        axis_to_align[0] = 1.0;
        axis_to_align[1] = 0.0;
        axis_to_align[2] = 0.0;
    }
    else
    {
        cross(plane_normal, target_plane_normal, axis_to_align);
        normalize(axis_to_align);
    }

    rotation_matrix(
        axis_to_align, angle_to_align, out_of_plane_rotation_matrix);
    apply_rotation(out_of_plane_rotation_matrix, H1, H1_temp);
    apply_rotation(out_of_plane_rotation_matrix, H2, H2_temp);

    double theta1         = std::atan2(H1_temp[1], H1_temp[0]);
    planar_rotation_angle = -theta1 + bondangle / 2.0;
    double planar_rotation_matrix[3][3];
    rotation_matrix(
        target_plane_normal, planar_rotation_angle, planar_rotation_matrix);
    apply_rotation(planar_rotation_matrix, H1_temp, H1_rotated);
    apply_rotation(planar_rotation_matrix, H2_temp, H2_rotated);

    flipped_bond = (bondlength1 < bondlength2);
    if (flipped_bond)
    {
        H1_rotated[1] *= -1.0;
        H2_rotated[1] *= -1.0;
        std::swap(H1_rotated, H2_rotated);
        std::swap(bondlength1, bondlength2);
    }

    positions[0] = H2_rotated[0];
    positions[1] = H2_rotated[1];
    positions[2] = H2_rotated[2];
    positions[3] = 0.0;
    positions[4] = 0.0;
    positions[5] = 0.0;
    positions[6] = H1_rotated[0];
    positions[7] = H1_rotated[1];
    positions[8] = H1_rotated[2];

    anumbers[0] = 1;
    anumbers[1] = 8;
    anumbers[2] = 1;
}

void PinnedH2O::transpose_rotate(std::vector<double>& positions,
    std::vector<short>& anumbers, std::vector<double>& forces)
{
    double H2_rotated[3] = { positions[0], positions[1], positions[2] };
    double O1_rotated[3] = { positions[3], positions[4], positions[5] };
    double H1_rotated[3] = { positions[6], positions[7], positions[8] };

    double f_H2_rotated[3] = { forces[0], forces[1], forces[2] };
    double f_O1_rotated[3] = { forces[3], forces[4], forces[5] };
    double f_H1_rotated[3] = { forces[6], forces[7], forces[8] };

    if (flipped_bond)
    {
        H1_rotated[1] *= -1.0;
        H2_rotated[1] *= -1.0;
        f_O1_rotated[1] *= -1.0;
        f_H1_rotated[1] *= -1.0;
        f_H2_rotated[1] *= -1.0;
    }

    double planar_rotation_matrix[3][3];
    double target_plane_normal[3] = { 0, 0, 1 };
    rotation_matrix(
        target_plane_normal, planar_rotation_angle, planar_rotation_matrix);

    double H1_temp[3], H2_temp[3];
    apply_transpose_rotation(planar_rotation_matrix, H1_rotated, H1_temp);
    apply_transpose_rotation(planar_rotation_matrix, H2_rotated, H2_temp);

    double f_O1_temp[3], f_H1_temp[3], f_H2_temp[3];
    apply_transpose_rotation(planar_rotation_matrix, f_O1_rotated, f_O1_temp);
    apply_transpose_rotation(planar_rotation_matrix, f_H1_rotated, f_H1_temp);
    apply_transpose_rotation(planar_rotation_matrix, f_H2_rotated, f_H2_temp);

    double H1_restored[3], H2_restored[3];
    apply_transpose_rotation(
        out_of_plane_rotation_matrix, H1_temp, H1_restored);
    apply_transpose_rotation(
        out_of_plane_rotation_matrix, H2_temp, H2_restored);

    double f_O1_restored[3], f_H1_restored[3], f_H2_restored[3];
    apply_transpose_rotation(
        out_of_plane_rotation_matrix, f_O1_temp, f_O1_restored);
    apply_transpose_rotation(
        out_of_plane_rotation_matrix, f_H1_temp, f_H1_restored);
    apply_transpose_rotation(
        out_of_plane_rotation_matrix, f_H2_temp, f_H2_restored);

    positions[3 * H2_idx]     = H2_restored[0];
    positions[3 * H2_idx + 1] = H2_restored[1];
    positions[3 * H2_idx + 2] = H2_restored[2];
    positions[3 * O1_idx]     = 0.0;
    positions[3 * O1_idx + 1] = 0.0;
    positions[3 * O1_idx + 2] = 0.0;
    positions[3 * H1_idx]     = H1_restored[0];
    positions[3 * H1_idx + 1] = H1_restored[1];
    positions[3 * H1_idx + 2] = H1_restored[2];

    anumbers[H2_idx] = 1;
    anumbers[O1_idx] = 8;
    anumbers[H1_idx] = 1;

    forces[3 * H2_idx]     = f_H2_restored[0];
    forces[3 * H2_idx + 1] = f_H2_restored[1];
    forces[3 * H2_idx + 2] = f_H2_restored[2];
    forces[3 * O1_idx]     = f_O1_restored[0];
    forces[3 * O1_idx + 1] = f_O1_restored[1];
    forces[3 * O1_idx + 2] = f_O1_restored[2];
    forces[3 * H1_idx]     = f_H1_restored[0];
    forces[3 * H1_idx + 1] = f_H1_restored[1];
    forces[3 * H1_idx + 2] = f_H1_restored[2];
}

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

using namespace std;

double calculate_bondlength(const double atom1[3], const double atom2[3])
{
    return sqrt(pow(atom1[0] - atom2[0], 2) + pow(atom1[1] - atom2[1], 2)
                + pow(atom1[2] - atom2[2], 2));
}

double calculate_bondangle(const double atom1[3], const double atom2[3],
    const double atom3[3], bool radian)
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

    if (!radian)
    {
        angle = angle * 180.0 / M_PI;
    }
    return angle;
}

void rotation_matrix(const double axis[3], double angle, double matrix[3][3])
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

void normalize(double vec[3])
{
    double norm = sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
    vec[0] /= norm;
    vec[1] /= norm;
    vec[2] /= norm;
}

void cross(const double a[3], const double b[3], double result[3])
{
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
}

void apply_rotation(
    const double matrix[3][3], const double vec[3], double result[3])
{
    result[0]
        = matrix[0][0] * vec[0] + matrix[0][1] * vec[1] + matrix[0][2] * vec[2];
    result[1]
        = matrix[1][0] * vec[0] + matrix[1][1] * vec[1] + matrix[1][2] * vec[2];
    result[2]
        = matrix[2][0] * vec[0] + matrix[2][1] * vec[1] + matrix[2][2] * vec[2];
}

void apply_transpose_rotation(
    const double matrix[3][3], const double vec[3], double result[3])
{
    result[0]
        = matrix[0][0] * vec[0] + matrix[1][0] * vec[1] + matrix[2][0] * vec[2];
    result[1]
        = matrix[0][1] * vec[0] + matrix[1][1] * vec[1] + matrix[2][1] * vec[2];
    result[2]
        = matrix[0][2] * vec[0] + matrix[1][2] * vec[1] + matrix[2][2] * vec[2];
}

int main()
{
    double O1[3] = { 0.00, 0.00, 0.00 };
    double H1[3] = { -0.45, -1.48, -0.97 };
    double H2[3] = { -0.45, 1.42, -1.07 };

    double bondlength1 = calculate_bondlength(H1, O1);
    double bondlength2 = calculate_bondlength(H2, O1);
    double bondangle   = calculate_bondangle(H1, O1, H2, false);

    cout << "Original system" << endl;
    cout << "H1 = (" << H1[0] << ", " << H1[1] << ", " << H1[2] << ")" << endl;
    cout << "H2 = (" << H2[0] << ", " << H2[1] << ", " << H2[2] << ")" << endl;
    cout << "Bondlength of O1-H1 = " << bondlength1 << endl;
    cout << "Bondlength of O1-H2 = " << bondlength2 << endl;
    cout << "Angle between O1-H1 and O1-H2 = " << bondangle << endl;

    double H1_temp[3], H2_temp[3];
    double H1_rotated[3], H2_rotated[3];

    double plane_normal[3];
    cross(H2, H1, plane_normal);
    normalize(plane_normal);
    double target_plane_normal[3] = { 0, 0, 1 };
    double axis_to_align[3];
    cross(plane_normal, target_plane_normal, axis_to_align);
    normalize(axis_to_align);
    double dot_product = plane_normal[0] * target_plane_normal[0]
                         + plane_normal[1] * target_plane_normal[1]
                         + plane_normal[2] * target_plane_normal[2];
    double angle_to_align = acos(min(max(dot_product, -1.0), 1.0));
    double out_of_plane_rotation_matrix[3][3];
    rotation_matrix(
        axis_to_align, angle_to_align, out_of_plane_rotation_matrix);
    apply_rotation(out_of_plane_rotation_matrix, H1, H1_temp);
    apply_rotation(out_of_plane_rotation_matrix, H2, H2_temp);

    double theta1                = atan2(H1_temp[1], H1_temp[0]);
    double planar_rotation_angle = -theta1 + bondangle / 360.0 * M_PI;
    double planar_rotation_matrix[3][3];
    rotation_matrix(
        target_plane_normal, planar_rotation_angle, planar_rotation_matrix);
    apply_rotation(planar_rotation_matrix, H1_temp, H1_rotated);
    apply_rotation(planar_rotation_matrix, H2_temp, H2_rotated);

    if (bondlength1 < bondlength2)
    {
        swap(H1_rotated, H2_rotated);
        H1_rotated[1] *= -1.0;
        H2_rotated[1] *= -1.0;
    }

    double bondlength1_rotated = calculate_bondlength(H1_rotated, O1);
    double bondlength2_rotated = calculate_bondlength(H2_rotated, O1);
    double bondangle_rotated
        = calculate_bondangle(H1_rotated, O1, H2_rotated, false);

    cout << "Reference system (in z=0 plane, symmetric about x=0 axis, with "
            "longer bondlength in Q1)"
         << endl;
    cout << "H1 = (" << H1_rotated[0] << ", " << H1_rotated[1] << ", "
         << H1_rotated[2] << ")" << endl;
    cout << "H2 = (" << H2_rotated[0] << ", " << H2_rotated[1] << ", "
         << H2_rotated[2] << ")" << endl;
    cout << "Bondlength of O1-H1 = " << bondlength1_rotated << endl;
    cout << "Bondlength of O1-H2 = " << bondlength2_rotated << endl;
    cout << "Angle between O1-H1 and O1-H2 = " << bondangle_rotated << endl;

    double H1_restored[3], H2_restored[3];

    if (bondlength1 < bondlength2)
    {
        swap(H1_rotated, H2_rotated);
        H1_rotated[1] *= -1.0;
        H2_rotated[1] *= -1.0;
    }

    apply_transpose_rotation(planar_rotation_matrix, H1_rotated, H1_temp);
    apply_transpose_rotation(planar_rotation_matrix, H2_rotated, H2_temp);

    apply_transpose_rotation(
        out_of_plane_rotation_matrix, H1_temp, H1_restored);
    apply_transpose_rotation(
        out_of_plane_rotation_matrix, H2_temp, H2_restored);

    double bondlength1_restored = calculate_bondlength(H1_restored, O1);
    double bondlength2_restored = calculate_bondlength(H2_restored, O1);
    double bondangle_restored
        = calculate_bondangle(H1_restored, O1, H2_restored, false);

    cout << "Restored system" << endl;
    cout << "H1 = (" << H1_restored[0] << ", " << H1_restored[1] << ", "
         << H1_restored[2] << ")" << endl;
    cout << "H2 = (" << H2_restored[0] << ", " << H2_restored[1] << ", "
         << H2_restored[2] << ")" << endl;
    cout << "Bondlength of O1-H1 = " << bondlength1_restored << endl;
    cout << "Bondlength of O1-H2 = " << bondlength2_restored << endl;
    cout << "Angle between O1-H1 and O1-H2 = " << bondangle_restored << endl;

    return 0;
}

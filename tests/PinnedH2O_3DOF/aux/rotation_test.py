import numpy as np

O1 = np.array([0.00, 0.00, 0.00])
H1 = np.array([-0.45, -1.48, -0.97])
H2 = np.array([-0.45, 1.42, -1.07])

def calculate_bondlength(atom1, atom2):
    return np.linalg.norm(atom1 - atom2)

def calculate_bondangle(atom1, atom2, atom3, radian):
    vector1 = atom1 - atom2
    vector2 = atom3 - atom2
    dot_product = np.dot(vector1, vector2)
    magnitude_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    angle = np.arccos(dot_product / magnitude_product)
    if not radian:
        angle = np.degrees(angle)
    return angle

bondlength1 = calculate_bondlength(H1, O1)
bondlength2 = calculate_bondlength(H2, O1)
bondangle = calculate_bondangle(H1, O1, H2, False)

print('Original system')
print(f'H1 = {H1}')
print(f'H2 = {H2}')
print(f'Bondlength of O1-H1 = {bondlength1}')
print(f'Bondlength of O1-H2 = {bondlength2}')
print(f'Angle between O1-H1 and O1-H2 = {bondangle}')

def rotation_matrix(axis, angle):
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    ux, uy, uz = axis
    return np.array([
        [cos_theta + ux**2 * (1 - cos_theta), ux * uy * (1 - cos_theta) - uz * sin_theta, ux * uz * (1 - cos_theta) + uy * sin_theta],
        [uy * ux * (1 - cos_theta) + uz * sin_theta, cos_theta + uy**2 * (1 - cos_theta), uy * uz * (1 - cos_theta) - ux * sin_theta],
        [uz * ux * (1 - cos_theta) - uy * sin_theta, uz * uy * (1 - cos_theta) + ux * sin_theta, cos_theta + uz**2 * (1 - cos_theta)]
    ])

H1_rotated, H2_rotated = H1, H2

plane_normal = np.cross(H2, H1)
plane_normal = plane_normal / np.linalg.norm(plane_normal)

target_plane_normal = np.array([0, 0, 1])
axis_to_align = np.cross(plane_normal, target_plane_normal)
axis_to_align /= np.linalg.norm(axis_to_align)
angle_to_align = np.arccos(np.clip(np.dot(plane_normal, target_plane_normal), -1.0, 1.0))
out_of_plane_rotation_matrix = rotation_matrix(axis_to_align, angle_to_align)
H1_rotated = np.dot(out_of_plane_rotation_matrix, H1)
H2_rotated = np.dot(out_of_plane_rotation_matrix, H2)

theta1 = np.arctan2(H1_rotated[1], H1_rotated[0]) 
theta2 = np.arctan2(H2_rotated[1], H2_rotated[0])
planar_rotation_angle = -theta1 + np.radians(bondangle) / 2 
planar_rotation_matrix = rotation_matrix(target_plane_normal, planar_rotation_angle)
H1_rotated = np.dot(planar_rotation_matrix, H1_rotated)
H2_rotated = np.dot(planar_rotation_matrix, H2_rotated)

if bondlength1 < bondlength2:
    H1_rotated, H2_rotated = H2_rotated, H1_rotated 
    H1_rotated[1] *= -1
    H2_rotated[1] *= -1

bondlength1_rotated = calculate_bondlength(H1_rotated, O1)
bondlength2_rotated = calculate_bondlength(H2_rotated, O1)
bondangle_rotated = calculate_bondangle(H1_rotated, O1, H2_rotated, False)

print('Reference system (z=0 plane, symmetric about x=0 axis, with longer bondlength in Q1)')
print(f'H1 = {H1_rotated}')
print(f'H2 = {H2_rotated}')
print(f'Bondlength of O1-H1 = {bondlength1_rotated}')
print(f'Bondlength of O1-H2 = {bondlength2_rotated}')
print(f'Angle between O1-H1 and O1-H2 = {bondangle_rotated}')

H1_restored, H2_restored = H1_rotated, H2_rotated

if bondlength1 < bondlength2:
    H1_rotated, H2_rotated = H2_rotated, H1_rotated 
    H1_rotated[1] *= -1
    H2_rotated[1] *= -1

H1_restored = np.dot(planar_rotation_matrix.T, H1_restored)
H2_restored = np.dot(planar_rotation_matrix.T, H2_restored)

H1_restored = np.dot(out_of_plane_rotation_matrix.T, H1_restored)
H2_restored = np.dot(out_of_plane_rotation_matrix.T, H2_restored)

bondlength1_restored = calculate_bondlength(H1_restored, O1)
bondlength2_restored = calculate_bondlength(H2_restored, O1)
bondangle_restored = calculate_bondangle(H1_restored, O1, H2_restored, False)

print('Restored system')
print(f'H1 = {H1_restored}')
print(f'H2 = {H2_restored}')
print(f'Bondlength of O1-H1 = {bondlength1_restored}')
print(f'Bondlength of O1-H2 = {bondlength2_restored}')
print(f'Angle between O1-H1 and O1-H2 = {bondangle_restored}')

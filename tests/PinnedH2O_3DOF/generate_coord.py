import numpy as np
import os

O1 = np.array([0.00, 0.00, 0.00])

ref_bondlength = 1.83
ref_bondangle = 104.5

# factors and increments for bond lengths and bond angle
bondlength_factor = np.linspace(0.95, 1.05, 11)
bondangle_increment = np.linspace(-5, 5, 11)

# output directory
output_dir = "data"

# generation
os.makedirs(output_dir, exist_ok=True)

for d_bondangle in bondangle_increment:
    bondangle = ref_bondangle + d_bondangle
    x = ref_bondlength * np.cos(np.radians(bondangle / 2))
    y = ref_bondlength * np.sin(np.radians(bondangle / 2))
    for i, f_bondlength1 in enumerate(bondlength_factor):
        for f_bondlength2 in bondlength_factor[:(i+1)]:
            H1 = np.array([f_bondlength1*x, f_bondlength1*y, 0.0])
            H2 = np.array([f_bondlength2*x, -f_bondlength2*y, 0.0])
            filename = f"{output_dir}/coords_{f_bondlength1:.2f}_{f_bondlength2:.2f}_{d_bondangle}.in"
            with open(filename, "w") as file:
                file.write(f"O1  1   {O1[0]:.2f}    {O1[1]:.2f}    {O1[2]:.2f}  0\n")
                file.write(f"H1  2   {H1[0]:.2f}    {H1[1]:.2f}    {H1[2]:.2f}  1\n")
                file.write(f"H2  2   {H2[0]:.2f}    {H2[1]:.2f}    {H2[2]:.2f}  1\n")

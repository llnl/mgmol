import glob
import re
import matplotlib.pyplot as plt
import numpy as np

ex_test = 0.00005
ey_test = 0.00005
ez_test = 0.00005
ex_init = 0.00000
ey_init = 0.00000
ez_init = 0.00000
e_min_rom = 0.00000
e_max_rom = 0.00010
e_num_increment_rom = 2
r_min = 6
r_max = 18

d_init = []
d_rom = []
valid_r = []
r_values = list(range(r_min, r_max + 1))

for r in r_values:
  ref_filename = f"ref_results_{r}_{ex_test:.5f}_{ey_test:.5f}_{ez_test:.5f}.out"
  online_filename = (
      f"online_results_{r}_{ex_test:.5f}_{ey_test:.5f}_{ez_test:.5f}"
      f"_{ex_init:.5f}_{ey_init:.5f}_{ez_init:.5f}"
      f"_{e_min_rom:.5f}_{e_max_rom:.5f}_{e_num_increment_rom}.out"
  )
  print(ref_filename)
  print(online_filename)

  ref_files = glob.glob(ref_filename)
  online_files = glob.glob(online_filename)

  if not ref_files or not online_files:
    continue

  ref_dipole = None
  with open(ref_files[0], "r") as f:
    for line in f:
      if "Dipole moment" in line and "Debye" in line:
        match = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        if len(match) >= 3:
          ref_dipole = np.array([float(match[0]), float(match[1]), float(match[2])])
          break

  online_dipoles = []
  with open(online_files[0], "r") as f:
    for line in f:
      if "Dipole moment" in line and "Debye" in line:
        match = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        if len(match) >= 3:
          online_dipoles.append(
              np.array([float(match[0]), float(match[1]), float(match[2])])
          )

  init_dipole = online_dipoles[0]
  rom_dipole = online_dipoles[1]

  norm_ref = np.linalg.norm(ref_dipole)

  err_init = np.linalg.norm(ref_dipole - init_dipole) / norm_ref
  err_rom = np.linalg.norm(ref_dipole - rom_dipole) / norm_ref

  d_init.append(err_init)
  d_rom.append(err_rom)

plot_filename = (
    f"dipole_moments_difference_{ex_test:.5f}_{ey_test:.5f}_{ez_test:.5f}"
    f"_{ex_init:.5f}_{ey_init:.5f}_{ez_init:.5f}"
    f"_{e_min_rom:.5f}_{e_max_rom:.5f}_{e_num_increment_rom}.png"
)
print(plot_filename)

plt.figure(figsize=(8, 5))
plt.plot(
    r_values,
    d_init,
    marker="o",
    linestyle="-",
    label="Initial vs Reference ($d_{init}$)",
)
plt.plot(
    r_values,
    d_rom,
    marker="s",
    linestyle="--",
    label="ROM vs Reference ($d_{rom}$)",
)

plt.xlabel("$r$", fontsize=12)
plt.ylabel("Relative Norm Difference", fontsize=12)
plt.title("Comparison of Dipole Moment Relative Errors", fontsize=14)
plt.xticks(valid_r)
plt.yscale("log")
plt.grid(True, which="both", linestyle=":", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig(plot_filename)

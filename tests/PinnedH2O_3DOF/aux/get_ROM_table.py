import subprocess
import re

bondlength_num_increments = (2, 5, 10)
bondangle_num_increments = (2, 5, 10)

pattern = r"For energy fraction: \d+\.\d+, take first (\d+) of \d+ basis vectors"

print("\\begin{tabular}{|c|c||c|c|c|c|c|c|c|}")
print("\\hline")
print("$N_L$ & $N_\\theta$ & $\\varepsilon = 10^{-1}$ & $\\varepsilon = 10^{-2}$ & $\\varepsilon = 10^{-3}$ & $\\varepsilon = 10^{-4}$ & $\\varepsilon = 10^{-5}$ & Snapshots \\\\")
print("\\hline")

for _, N_L in enumerate(bondlength_num_increments): 
    for _, N_theta in enumerate(bondangle_num_increments): 
        snapshots = 2*(N_L+1)*(N_L+2)*(N_theta+1)
        grep_command = f"grep 'take first' basis_{N_L}_{N_theta}_PinnedH2O_3DOF.out"
        result = subprocess.run(grep_command, shell=True, capture_output=True, text=True)
        matches = re.findall(pattern, result.stdout)
        energy_fractions = {
            "0.9": matches[0],
            "0.99": matches[1],
            "0.999": matches[2],
            "0.9999": matches[3],
            "0.99999": matches[4],
        }
        line = f"{N_L} & {N_theta} & {energy_fractions['0.9']} & {energy_fractions['0.99']} & {energy_fractions['0.999']} & {energy_fractions['0.9999']} & {energy_fractions['0.99999']} & {snapshots} \\\\"
        print(line)

print("\\hline")
print("\\end{tabular}")

import subprocess
import re

print("\\begin{tabular}{|c||c|c|c|c|c|c|c|}")
print("\\hline")
print("$k$ & $\\varepsilon = 10^{-1}$ & $\\varepsilon = 10^{-2}$ & $\\varepsilon = 10^{-3}$ & $\\varepsilon = 10^{-4}$ & $\\varepsilon = 10^{-5}$ & Snapshots \\\\")
print("\\hline")

pattern = r"For energy fraction: \d+\.\d+, take first (\d+) of \d+ basis vectors"
for t in range(10):
    k = 50*(t+1)
    snapshots = 4*k
    grep_command = f"grep 'take first' basis_1_{k}_Pinned_H2O.out"
    result = subprocess.run(grep_command, shell=True, capture_output=True, text=True)
    matches = re.findall(pattern, result.stdout)
    energy_fractions = {
        "0.9": matches[0],
        "0.99": matches[1],
        "0.999": matches[2],
        "0.9999": matches[3],
        "0.99999": matches[4],
    }
    line = f"{k} & {energy_fractions['0.9']} & {energy_fractions['0.99']} & {energy_fractions['0.999']} & {energy_fractions['0.9999']} & {energy_fractions['0.99999']} & {snapshots} \\\\"
    print(line)

print("\\hline")
print("\\end{tabular}")

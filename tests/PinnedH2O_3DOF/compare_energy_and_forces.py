import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse

parser = argparse.ArgumentParser(description="Calculate and plot differences in energies and forces.")

parser.add_argument("--N_l", type=int, default=2, help="Sampling frequency of bond length points")
parser.add_argument("--N_theta", type=int, default=2, help="Sampling frequency of bond angle points")
parser.add_argument("--rdim", type=int, default=18, help="Dimension of the ROM basis")

args = parser.parse_args()

N_l = args.N_l
N_theta = args.N_theta
rdim = args.rdim

bondlength_min = 0.95
bondlength_max = 1.05
bondlength_num_increments = 10

bondangle_min = -5.0
bondangle_max = 5.0
bondangle_num_increments = 10

output_dir = f'results_{N_l}_{N_theta}_{rdim}'

def process_data(s_l1, s_l2, s_theta, N_l, N_theta, rdim):
    try:
        data_dir = f'data/{s_l1}_{s_l2}_{s_theta}'
        offline_file = os.path.join(data_dir, 'offline_PinnedH2O.out')
        log_file = f'{output_dir}/{s_l1}_{s_l2}_{s_theta}.log'

        f_O1_fom = None
        f_H1_fom = None
        f_H2_fom = None

        if not os.path.exists(offline_file):
            print(f"Error: File not found: {offline_file}")
            return None

        with open(offline_file, 'r') as f:
            eks_lines = [line for line in f if " SC ENERGY      [Ha]" in line]
            if eks_lines:
                Eks_fom = float(eks_lines[-1].split()[-1])
            else:
                print(f"Error: Could not find 'SC ENERGY    [Ha]' line in {offline_file}")
                return None

        with open(offline_file, 'r') as f:
            for line in f:
                if "O1" in line:
                    f_O1_fom = np.array(line.split()[6:9], dtype=float)
                elif "H1" in line:
                    f_H1_fom = np.array(line.split()[5:8], dtype=float)
                elif "H2" in line:
                    f_H2_fom = np.array(line.split()[5:8], dtype=float)

                if f_O1_fom is not None and f_H1_fom is not None and f_H2_fom is not None:
                    break # Exit the loop once we found all lines

        if f_O1_fom is None or f_H1_fom is None or f_H2_fom is None:
            print(f"Error: Could not find O1, H1, or H2 lines in {offline_file}")
            return None

        f_H2_rom = None
        f_O1_rom = None
        f_H1_rom = None

        if not os.path.exists(log_file):
            print(f"Error: File not found: {log_file}")
            return None

        with open(log_file, 'r') as f:
            lines = f.readlines()

        eks_index = -1
        for i in range(len(lines) - 1, -1, -1):  # Search from the end
            if "Eks:" in lines[i]:
                eks_index = i
                break

        if eks_index == -1:
            print(f"Error: Could not find 'Eks:' line in {log_file}")
            return None

        Eks_rom = float(lines[eks_index].split()[-1])  # Get the last element

        force_index = -1
        for i in range(len(lines) - 1, -1, -1):  # Search from the end
            if "Forces:" in lines[i]:
                force_index = i
                break

        if force_index == -1:
            print(f"Error: Could not find 'Forces:' line in {log_file}")
            return None

        if force_index + 3 >= len(lines):
            print(f"Error: Not enough lines after 'Forces:' in {log_file}")
            return None

        f_H2_rom = np.array(lines[force_index + 1].split(), dtype=float)
        f_O1_rom = np.array(lines[force_index + 2].split(), dtype=float)
        f_H1_rom = np.array(lines[force_index + 3].split(), dtype=float)

        return Eks_fom, f_O1_fom, f_H1_fom, f_H2_fom, Eks_rom, f_O1_rom, f_H1_rom, f_H2_rom

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

Eks_diff_reproductive = []
f_O1_diff_reproductive = []
f_H1_diff_reproductive = []
f_H2_diff_reproductive = []

Eks_diff_predictive = []
f_O1_diff_predictive = []
f_H1_diff_predictive = []
f_H2_diff_predictive = []

for i in range(bondlength_num_increments + 1):
    bondlength_one = round(bondlength_min + i * (bondlength_max - bondlength_min) / bondlength_num_increments, 2)
    for j in range(i + 1):
        bondlength_two = round(bondlength_min + j * (bondlength_max - bondlength_min) / bondlength_num_increments, 2)
        for k in range(bondangle_num_increments + 1):
            bondangle = round(bondangle_min + k * (bondangle_max - bondangle_min) / bondangle_num_increments, 1)

            s_l1 = f"{bondlength_one:.2f}"
            s_l2 = f"{bondlength_two:.2f}"
            s_theta = f"{bondangle:.1f}"
            tag = f'{s_l1}_{s_l2}_{s_theta}'

            output_filename = f'{output_dir}/energy_and_forces_{tag}.out'

            with open(output_filename, 'w') as outfile:
                print("s_l1:", s_l1, file=outfile)
                print("s_l2:", s_l2, file=outfile)
                print("s_theta:", s_theta, file=outfile)
                print("N_l:", N_l, file=outfile)
                print("N_theta:", N_theta, file=outfile)
                print("rdim:", rdim, file=outfile)

                results = process_data(s_l1, s_l2, s_theta, N_l, N_theta, rdim)

                if results:
                    Eks_fom, f_O1_fom, f_H1_fom, f_H2_fom, Eks_rom, f_O1_rom, f_H1_rom, f_H2_rom = results

                    print("Eks_fom:", Eks_fom, file=outfile)
                    print("Eks_rom:", Eks_rom, file=outfile)

                    print("f_O1_fom:", f_O1_fom, file=outfile)
                    print("f_O1_rom:", f_O1_rom, file=outfile)

                    print("f_H1_fom:", f_H1_fom, file=outfile)
                    print("f_H1_rom:", f_H1_rom, file=outfile)

                    print("f_H2_fom:", f_H2_fom, file=outfile)
                    print("f_H2_rom:", f_H2_rom, file=outfile)

                    def calculate_differences(fom, rom, name):
                        abs_diff = np.linalg.norm(fom - rom)
                        rel_diff = abs_diff / np.linalg.norm(fom) if np.linalg.norm(fom)!= 0 else float('inf')
                        print(f"Absolute difference in {name}:", abs_diff, file=outfile)
                        print(f"Relative difference in {name}:", rel_diff, file=outfile)
                        return abs_diff

                    Eks_diff = calculate_differences(Eks_fom, Eks_rom, "Eks")
                    f_O1_diff = calculate_differences(f_O1_fom, f_O1_rom, "f_O1")
                    f_H1_diff = calculate_differences(f_H1_fom, f_H1_rom, "f_H1")
                    f_H2_diff = calculate_differences(f_H2_fom, f_H2_rom, "f_H2")

                    if i * N_l % bondlength_num_increments == 0 and j * N_l % bondlength_num_increments == 0 and k * N_theta % bondangle_num_increments == 0:
                        Eks_diff_reproductive.append(Eks_diff)
                        f_O1_diff_reproductive.append(f_O1_diff)
                        f_H1_diff_reproductive.append(f_H1_diff)
                        f_H2_diff_reproductive.append(f_H2_diff)
                    else:
                        Eks_diff_predictive.append(Eks_diff)
                        f_O1_diff_predictive.append(f_O1_diff)
                        f_H1_diff_predictive.append(f_H1_diff)
                        f_H2_diff_predictive.append(f_H2_diff)
                else:
                    print("Error occurred during data processing. Differences cannot be calculated.", file=outfile)

def plot_histogram(data, quantity, test_case):
    plt.figure(figsize=(8, 6)) 
    plt.hist(data, bins=20, color='skyblue', edgecolor='black')

    if quantity == "Eks":
        quantity_name = "absolute difference in total energy"
    elif quantity.startswith("f_"):
        quantity_name = f"magnitude of difference in force on {quantity[2:]}"
    else:
        raise ValueError("Invalid input quantity")

    plt.title(f'Histogram of {quantity_name}')
    plt.xlabel('Difference')
    plt.ylabel('Frequency')

    min_val, max_val = np.min(data), np.max(data)
    plt.xlim(min_val, max_val)
    num_ticks = 8 
    xticks = np.linspace(min_val, max_val, num_ticks)
    plt.xticks(xticks)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0)) 
    plt.gca().xaxis.set_major_formatter(formatter)

    total_count = len(data)
    mean_val = np.mean(data)
    max_val = np.max(data)
    stats_text = (f"Total {test_case} cases: {total_count}\n"
                  f"Mean: {mean_val:.3e}")
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

    plt.tight_layout() 
    plt.savefig(f"{output_dir}/{quantity}_difference_histogram_{test_case}.png")

plot_histogram(Eks_diff_reproductive, "Eks", "reproductive")
plot_histogram(f_O1_diff_reproductive, "f_O1", "reproductive")
plot_histogram(f_H1_diff_reproductive, "f_H1", "reproductive")
plot_histogram(f_H2_diff_reproductive, "f_H2", "reproductive")

plot_histogram(Eks_diff_predictive, "Eks", "predictive")
plot_histogram(f_O1_diff_predictive, "f_O1", "predictive")
plot_histogram(f_H1_diff_predictive, "f_H1", "predictive")
plot_histogram(f_H2_diff_predictive, "f_H2", "predictive")

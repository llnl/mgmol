#!/bin/bash

rm -rf PinnedH2O_3DOF_assembled_results
mkdir PinnedH2O_3DOF_assembled_results

for d in results_*; do
  echo "Processing $d"
  rm $d/energy_and_forces_*.out
  rm $d/*.png

  suffix=$(echo "$d" | cut -d'_' -f2,3,4)
  N_l=$(echo "$suffix" | cut -d'_' -f1)
  N_theta=$(echo "$suffix" | cut -d'_' -f2)
  rdim=$(echo "$suffix" | cut -d'_' -f3)
  python3 compare_energy_and_forces.py --N_l "$N_l" --N_theta "$N_theta" --rdim "$rdim"

  cp "$d/Eks_difference_histogram_reproductive.png" "PinnedH2O_3DOF_assembled_results/Eks_difference_histogram_${suffix}_reproductive.png"
  cp "$d/f_H1_difference_histogram_reproductive.png" "PinnedH2O_3DOF_assembled_results/f_H1_difference_histogram_${suffix}_reproductive.png"
  cp "$d/f_H2_difference_histogram_reproductive.png" "PinnedH2O_3DOF_assembled_results/f_H2_difference_histogram_${suffix}_reproductive.png"
  cp "$d/f_O1_difference_histogram_reproductive.png" "PinnedH2O_3DOF_assembled_results/f_O1_difference_histogram_${suffix}_reproductive.png"

  cp "$d/Eks_difference_histogram_predictive.png" "PinnedH2O_3DOF_assembled_results/Eks_difference_histogram_${suffix}_predictive.png"
  cp "$d/f_H1_difference_histogram_predictive.png" "PinnedH2O_3DOF_assembled_results/f_H1_difference_histogram_${suffix}_predictive.png"
  cp "$d/f_H2_difference_histogram_predictive.png" "PinnedH2O_3DOF_assembled_results/f_H2_difference_histogram_${suffix}_predictive.png"
  cp "$d/f_O1_difference_histogram_predictive.png" "PinnedH2O_3DOF_assembled_results/f_O1_difference_histogram_${suffix}_predictive.png"
done

tar -cvf PinnedH2O_3DOF_assembled_results.tar PinnedH2O_3DOF_assembled_results
mv PinnedH2O_3DOF_assembled_results.tar /p/lustre2/cheung26/scp_local

echo "Finished assembling results."

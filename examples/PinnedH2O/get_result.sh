#filename="offline_PinnedH2O" # FOM
#filename="rom39_PinnedH2O" # ROM compare MD
#filename="39_force_PinnedH2O" # ROM compare force

#filename="PinnedH2O_test2_ref" # FOM
filename="PinnedH2O_rom_3DOF_test2_2_2_34" # ROM PinnedH2O 3DOF MD

# Extracting kinetic energy, total energy, temperature from MGmgol output log
awk '/Kinetic/ {print $3}' $filename.out > ke_$filename.txt
awk '/Kinetic/ {print $5}' $filename.out >temp_$filename.txt
awk '/Total/ {print $3}' $filename.out > te_$filename.txt

# Extracting H1, H2, F1, F2 from MGmgol output log
# if FOM, these files contain the FOM results
# if compare MD, these files contain the results with projected orbitals
awk '/O1 / {print $4, $5, $6}' $filename.out > O1_$filename.txt
awk '/H1 / {print $3, $4, $5}' $filename.out > H1_$filename.txt
awk '/H2 / {print $3, $4, $5}' $filename.out > H2_$filename.txt
awk '/O1 / {print $7, $8, $9}' $filename.out > f_O1_$filename.txt
awk '/H1 / {print $6, $7, $8}' $filename.out > f_H1_$filename.txt
awk '/H2 / {print $6, $7, $8}' $filename.out > f_H2_$filename.txt

# if compare force, files with "_fom" contain the FOM results
# files with "_rom" contain the results with projected orbitals
if [[ "$filename" == *"force_"* ]]; then
    sed -n '1~2p' H1_$filename.out > H1_rom$filename.txt
    sed -n '1~2p' H2_$filename.out > H2_rom$filename.txt
    sed -n '1~2p' f_H1_$filename.out > f_H1_rom$filename.txt
    sed -n '1~2p' f_H2_$filename.out > f_H2_rom$filename.txt

    sed -n '2~2p' H1_$filename.out > H1_fom$filename.txt
    sed -n '2~2p' H2_$filename.out > H2_fom$filename.txt
    sed -n '2~2p' f_H1_$filename.out > f_H1_fom$filename.txt
    sed -n '2~2p' f_H2_$filename.out > f_H2_fom$filename.txt
fi

rm -rf snapshot_*

#!/usr/bin/env python
import sys
import os
import subprocess
import string
import shutil

print("Test MD with Davidson solver...")

nargs=len(sys.argv)

mpicmd = sys.argv[1]+" "+sys.argv[2]+" "+sys.argv[3]
for i in range(4,nargs-5):
  mpicmd = mpicmd + " "+sys.argv[i]
print("MPI run command: {}".format(mpicmd))

exe = sys.argv[nargs-5]
inp1 = sys.argv[nargs-4]
inp2 = sys.argv[nargs-3]
coords = sys.argv[nargs-2]
print("coordinates file: %s"%coords)

#create links to potentials files
dst = 'pseudo.Li_ONCVPSP_LDA'
src = sys.argv[-1] + '/' + dst

if not os.path.exists(dst):
  print("Create link to %s"%dst)
  os.symlink(src, dst)

#run quench
command = "{} {} -c {} -i {}".format(mpicmd,exe,inp1,coords)
print("Run command: {}".format(command))
output1 = subprocess.check_output(command,shell=True)
lines=output1.split(b'\n')

#analyse output of quench
for line in lines:
  num_matches = line.count(b'%%')
  if num_matches:
    print(line)

#run MD
for i in range(2):
  command = "ls -ld snapshot* | awk '{ print $9 }' | tail -n1"
  print(command)
  restart_file = subprocess.check_output(command,shell=True)
  restart_file=str(restart_file[:-1],'utf-8')
  print(restart_file)

  os.rename(restart_file, 'snapshotDavidson')

  #run MGmol
  command = "{} {} -c {}".format(mpicmd,exe,inp2)
  output2 = subprocess.check_output(command,shell=True)

  #remove used restart files
  shutil.rmtree('snapshotDavidson')

  #analyse mgmol standard output
  lines=output2.split(b'\n')

  print("Check energy conservation...")
  tol = 1.e-4
  energy = 0.
  count = 0
  for line in lines:
    if line.count(b'%%'):
      print(line)
    if line.count(b'Total') and line.count(b'Energy'):
      print(line)
      count=count+1
      words=line.split()
    
      energy=eval(words[2])
      if count==1:
        first_energy=energy

      if count>1 and abs(energy-first_energy)>tol:
        print("ERROR Energy = {} != {}".format(energy,first_energy))
        sys.exit(1)

#remove last restart files
command = "ls -ld snapshot* | awk '{ print $9 }' | tail -n1"
restart_file = subprocess.check_output(command,shell=True)
restart_file=str(restart_file[:-1],'utf-8')
shutil.rmtree(restart_file)

sys.exit(0)

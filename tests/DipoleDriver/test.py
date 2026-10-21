#!/usr/bin/env python
import sys
import os
import subprocess
import string

print("Test computation of dipole moment...")

nargs=len(sys.argv)

mpicmd = sys.argv[1]+" "+sys.argv[2]+" "+sys.argv[3]
for i in range(4,nargs-6):
  mpicmd = mpicmd + " "+sys.argv[i]
print("MPI run command: {}".format(mpicmd)) 

mgmol      = sys.argv[nargs-6]
compDipole = sys.argv[nargs-5]
inp1 = sys.argv[nargs-4]
inp2 = sys.argv[nargs-3]
coords = sys.argv[nargs-2]
print("coordinates file: %s"%coords)

#create links to potentials files
cwd = os.getcwd()

dst = 'pseudo.O_ONCV_PBE_SG15'
src = sys.argv[nargs-1] + '/' + dst
if not os.path.exists(cwd+'/'+dst):
  print("Create link to %s"%dst)
  os.symlink(src, dst)

dst = 'pseudo.H_ONCV_PBE_SG15'
src = sys.argv[nargs-1] + '/' + dst
if not os.path.exists(cwd+'/'+dst):
  print("Create link to %s"%dst)
  os.symlink(src, dst)

#run mgmol
command = "{} {} -c {} -i {}".format(mpicmd,mgmol,inp1,coords)
print("Run command: {}".format(command))

output = subprocess.check_output(command,stderr=subprocess.STDOUT,shell=True)

#analyse mgmol standard output
#make sure force is below tolerance
lines=output.split(b'\n')

convergence=0
for line in lines:
  if line.count(b'achieved') and line.count(b'convergence'):
    convergence=1
    break

if convergence==0:
  print("MGmol: Solver did not converge")
  sys.exit(1)

#run driver
command = "{} {} -c {}".format(mpicmd,compDipole,inp2)
print("Run command: {}".format(command))

output = subprocess.check_output(command,stderr=subprocess.STDOUT,shell=True)

#analyse driver standard output
#make sure force is below tolerance
lines=output.split(b'\n')

for line in lines:
  if line.count(b'Dipole') and line.count(b'Debye'):
    print(line)
    words=line.split()
    dy = eval(words[4][:-1])
    print("Dipole (Debye): {}".format(dy))

tol = 1.e-3
ref_dipole=1.878
if abs(dy-ref_dipole)>tol:
  print("Expected dipole: {}".format(ref_dipole))
  sys.exit(1)

print("Test PASSED")
sys.exit(0)

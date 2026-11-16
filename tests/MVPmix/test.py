#!/usr/bin/env python
import sys
import os
import subprocess
import string

print("Test MVP solver with mixing coefficient...")

nargs=len(sys.argv)

mpicmd = sys.argv[1]+" "+sys.argv[2]+" "+sys.argv[3]
for i in range(4,nargs-4):
  mpicmd = mpicmd + " "+sys.argv[i]
print("MPI run command: {}".format(mpicmd)) 

exe = sys.argv[nargs-4]
inp = sys.argv[nargs-3]
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
command = "{} {} -c {} -i {}".format(mpicmd,exe,inp,coords)
print("Run command: {}".format(command))

output = subprocess.check_output(command,stderr=subprocess.STDOUT,shell=True)

#analyse mgmol standard output
#make sure force is below tolerance
lines=output.split(b'\n')

flag = 0
eigenvalues=[]
energies=[]
ecount=0
for line in lines:
  if line.count(b'FERMI'):
    flag = 0
  if flag==1:
    words=line.split()
    for w in words:
      eigenvalues.append(eval(w))
  if line.count(b'Eigenvalues'):
    flag = 1
    eigenvalues=[]
  if line.count(b'%%'):
    words=line.split()
    e=words[5][0:-1]
    print(e)
    ecount=ecount+1
    energies.append(eval(e))
print(energies)

convergence=0
for line in lines:
  if line.count(b'DFTsolver:') and line.count(b'convergence'):
    convergence=1
    break

if convergence==0:
  print("MVP Solver did not converge")
  sys.exit(1)

print(eigenvalues)
tol = 1.e-4
eigenvalue0 = -0.916
if abs(eigenvalues[0]-eigenvalue0)>tol:
  print("Expected eigenvalue 0 to be {}".format(eigenvalue0))
  sys.exit(1)
eigenvalue8 = 0.219
if abs(eigenvalues[8]-eigenvalue8)>tol:
  print("Expected eigenvalue 8 to be {}".format(eigenvalue8))
  sys.exit(1)

niterations = ecount
print("MVP solver ran for {} iterations".format(niterations))
if niterations>180:
  print("MVP test FAILED for taking too many iterations")
  sys.exit(1)

print("Check energy...")
last_energy = energies[-1]
print("Energy = {}".format(last_energy))
if last_energy>-17.16269:
  print("Last energy = {}".format(last_energy))
  sys.exit(1)

print("Test PASSED")
sys.exit(0)

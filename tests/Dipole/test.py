#!/usr/bin/env python
import sys
import os
import subprocess
import string

print("Test computation of dipole moment...")

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
command = "{} {} -c {} -i {}".format(mpicmd,exe,inp1,coords)
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
  print("1st run: Solver did not converge")
  sys.exit(1)

for line in lines:
  if line.count(b'Dipole') and line.count(b'Debye'):
    print(line)
    words=line.split()
    dy = eval(words[4][:-1])
    print("Dipole (Debye): {}".format(dy))
  if line.count(b'Dipole') and line.count(b'a.u.'):
    print(line)
    words=line.split()
    dyau = eval(words[4][:-1])
    print("Dipole (a.u.): {}".format(dyau))

tol = 1.e-3
ref_dipole=1.878
if abs(dy-ref_dipole)>tol:
  print("Expected dipole: {}".format(ref_dipole))
  sys.exit(1)

#run mgmol again with efield
command = "{} {} -c {}".format(mpicmd,exe,inp2)
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
  print("2nd run: DFT Solver did not converge")
  sys.exit(1)

for line in lines:
  if line.count(b'Dipole') and line.count(b'a.u.'):
    print(line)
    words=line.split()
    dyaup = eval(words[4][:-1])
    print("Dipole (a.u.): {}".format(dyaup))
  if line.count(b'Potentials:') and line.count(b'ex'):
    print(line)
    words=line.split()
    efield = eval(words[6][:-1])
    print("efield = {}".format(efield))

polarizibility = (dyaup-dyau)/efield
print("Polarizibility: {}".format(polarizibility))

tol = 1.e-1
ref_polarizibility=10.1
if abs(polarizibility-ref_polarizibility)>tol:
  print("Expected polarizibility: {}".format(ref_polarizibility))
  sys.exit(1)

print("Test PASSED")
sys.exit(0)

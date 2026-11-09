#!/usr/bin/env python
import sys
import os
import subprocess
import string

print("Test Order 6...")

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
dst = 'pseudo.Si_ONCV_PBE_SG15'
src = sys.argv[nargs-1] + '/' + dst

if not os.path.exists(dst):
  print("Create link to %s"%dst)
  os.symlink(src, dst)

#run mgmol
command = "{} {} -c {} -i {}".format(mpicmd,exe,inp,coords)
print("Run command: {}".format(command))
output = subprocess.check_output(command,shell=True)

os.remove(dst)

#analyse mgmol standard output
#make sure forces are below tolerance
lines=output.split(b'\n')

tol = 5.e-3
ref = [0., 0., 5.e-2]
found_forces = False
count = 0
for line in lines:
  if line.count(b'%%'):
    print(line)
  if line.count(b'##'):
    words=line.split()
    if len(words)==8:
      print(line)
      found_forces = True
      count = count + 1
      for i in range(5,8):
        f = eval(words[i])
        if (abs(f)-ref[i-5])>tol:
          print(" force {} = {}".format(i-5, f))
          sys.exit(1)
  if count == 5:
    break

if count < 2:
  print("2 forces not found")
  sys.exit(1)

sys.exit(0)

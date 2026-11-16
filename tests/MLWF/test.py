#!/usr/bin/env python
import sys
import os
import subprocess
import string

print("Test MLWF...")

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
src = sys.argv[-1] + '/' + dst

if not os.path.exists(dst):
  print("Create link to %s"%dst)
  os.symlink(src, dst)

#run MGmol
command = "{} {} -c {} -i {}".format(mpicmd,exe,inp,coords)
output = subprocess.check_output(command,shell=True)

#analyse mgmol standard output
lines=output.split(b'\n')

print("Check MLWF convergence and spread values...")

found = False
max_sweeps = 40
for line in lines:
  num_matches = line.count(b'MLWFTransform')
  if num_matches:
    found = True

  if found:
    num_matches = line.count(b'jade')
    if num_matches:
      print(line)
      words=line.split()
      nsweeps = eval(words[0])
      if nsweeps > max_sweeps:
        print("Test failed: nsweeps = {}".format(nsweeps));
        sys.exit(1)

    num_matches = line.count(b'&&')
    if num_matches:
      print(line)
      words=line.split()
      spread = eval(words[5])
      if spread>3.2:
        print("Test failed: spread = {}".format(spread));
        sys.exit(1)
      if spread<2.5:
        print("Test failed: spread = {}".format(spread));
        sys.exit(1)

sys.exit(0)

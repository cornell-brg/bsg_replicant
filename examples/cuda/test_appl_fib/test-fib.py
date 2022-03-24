import os
import subprocess

def gen_sh(n, gsize, X, Y):
  with open("run.sh", 'w') as outfile:
    cmd = "(make clean; make FIB_IN={0} FIB_GSIZE={1} TILE_GROUP_DIM_X={2} TILE_GROUP_DIM_Y={3} exec.log > fib-{0}-{1}.dp 2>&1;)".format(
      n,
      gsize,
      X,
      Y)
    print(cmd)
    outfile.write(cmd)

if __name__ == "__main__":
  path = str(os.path.abspath(os.getcwd()))
  print(path)
  for n in range(16,22):
    for g in range(3,n-8):
      gen_sh(n, g, 16, 8)
      run = subprocess.run(["sh", path + "/run.sh"], env=os.environ)

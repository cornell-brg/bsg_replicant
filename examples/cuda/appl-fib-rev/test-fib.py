import os
import subprocess

with open("tests.mk", "w") as outfile:
    def gen_sh(n, gsize, X, Y, impl):
        cmd = "TESTS += $(call test-name,{0},{1},{2},{3},{4})\n".format(
            n,
            gsize,
            X,
            Y,
            impl)
        outfile.write(cmd)

    path = str(os.path.abspath(os.getcwd()))
    for n in range(16,22):
        for g in range(3,n-8):
            gen_sh(n, g, 16, 8, "APPLRTS")
            gen_sh(n, g, 16, 8, "CELLO")
            gen_sh(n, g, 16, 8, "SERIAL")


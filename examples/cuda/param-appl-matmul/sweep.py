N = [32, 256]
IMPL = ['SERIAL', 'APPLRTS', 'CELLO']
TGD = [(1,1),(2,2),(4,4),(8,8),(16,8)]
with open('tests.mk', 'w') as ofile:
    for impl in IMPL:
        for n in N:
            for (tgx,tgy) in TGD:
                ofile.write('TESTS += $(call test-name,{},{},{},{})\n'.format(
                    n,
                    tgx,
                    tgy,
                    impl
                ))


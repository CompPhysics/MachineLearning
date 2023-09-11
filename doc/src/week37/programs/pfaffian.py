from pfapack import pfaffian as pf
import numpy.matlib

A = numpy.matlib.rand(100, 100)
A = A - A.T
pfa1 = pf.pfaffian(A)
pfa2 = pf.pfaffian(A, method="H")
pfa3 = pf.pfaffian_schur(A)

print(pfa1, pfa2, pfa3)

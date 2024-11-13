# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import engine as ng
import test_suite as tst
import matplotlib.pyplot as plt

## Diagnostics de performance
import numpy as np
def diffAbs(uFinal,uRef):
	return 1/uFinal.shape[0]*np.sum((uFinal - uRef)**2)**.5


def conservation(uFinal,uRef,scheme):
	integ0 = 2*np.sum(uFinal*scheme.dx)
	integ1 = 2*np.sum(uRef*scheme.dx)
	return integ1 - integ0

# Conservativité pour intégration RK4

# test 1

# +
test = tst.Test1()
scheme = ng.MUSCL(test)
scheme.form = "KT"
scheme.compute(scheme.tFinal)
plt.plot(scheme.x, scheme.uFinal)
plt.plot(scheme.x, scheme.uF, marker = "o", markersize=5, linestyle = "None")

print("Erreur RMS: ", diffAbs(scheme.uFinal,scheme.uF))
print("Erreur de conservation : ", conservation(scheme.uF,scheme.u0(scheme.x),scheme))
# -

# test 2

# +
test = tst.Test2()
scheme = ng.MUSCL(test)
scheme.form = "KT"
scheme.compute(scheme.tFinal)
plt.plot(scheme.x, scheme.uFinal)
plt.plot(scheme.x, scheme.uF, marker = "o", markersize=5, linestyle = "None")

print("Erreur RMS: ", diffAbs(scheme.uFinal,scheme.uF))
print("Erreur de conservation : ", conservation(scheme.uF,scheme.u0(scheme.x),scheme))
# -

# test 3

# +
test = tst.Test3()
scheme = ng.MUSCL(test)
scheme.form = "KT"
scheme.compute(scheme.tFinal)
plt.plot(scheme.x, scheme.uFinal)
plt.plot(scheme.x, scheme.uF, marker = "o", markersize=5, linestyle = "None")

print("Erreur RMS: ", diffAbs(scheme.uFinal,scheme.uF))
print("Erreur de conservation : ", conservation(scheme.uF,scheme.u0(scheme.x),scheme))
# -

# test 4

# +
test = tst.Test4()
scheme = ng.MUSCL(test)
scheme.form = "KT"
scheme.compute(scheme.tFinal)
plt.plot(scheme.x, scheme.uFinal)
plt.plot(scheme.x, scheme.uF, marker = "o", markersize=5, linestyle = "None")

print("Erreur RMS: ", diffAbs(scheme.uFinal,scheme.uF))
print("Erreur de conservation : ", conservation(scheme.uF,scheme.u0(scheme.x),scheme))
# -

# test 5

# +
test = tst.Test5()
scheme = ng.MUSCL(test)
scheme.form = "KT"
scheme.compute(scheme.tFinal)
plt.plot(scheme.x, scheme.uFinal)
plt.plot(scheme.x, scheme.uF, marker = "o", markersize=5, linestyle = "None")

print("Erreur RMS: ", diffAbs(scheme.uFinal,scheme.uF))
print("Erreur de conservation : ", conservation(scheme.uF,scheme.u0(scheme.x),scheme))

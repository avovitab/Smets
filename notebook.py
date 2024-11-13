#!/usr/bin/env python
# coding: utf-8

# In[1]:


import engine as ng
import test_suite as tst
import matplotlib.pyplot as plt


# In[2]:


## Diagnostics de performance
import numpy as np
def diffAbs(uFinal,uRef):
	return 1/uFinal.shape[0]*np.sum((uFinal - uRef)**2)**.5


# In[ ]:


def conservation(uFinal,uRef,scheme):
	integ0 = 2*np.sum(uFinal*scheme.dx)
	integ1 = 2*np.sum(uRef*scheme.dx)
	return integ1 - integ0


# Conservativité pour intégration RK4

# test 1

# In[4]:


test = tst.Test1()
scheme = ng.MUSCL(test)
scheme.form = "KT"
scheme.compute(scheme.tFinal)
plt.plot(scheme.x, scheme.uFinal)
plt.plot(scheme.x, scheme.uF, marker = "o", markersize=5, linestyle = "None")

print("Erreur RMS: ", diffAbs(scheme.uFinal,scheme.uF))
print("Erreur de conservation : ", conservation(scheme.uF,scheme.u0(scheme.x),scheme))


# test 2

# In[5]:


test = tst.Test2()
scheme = ng.MUSCL(test)
scheme.form = "KT"
scheme.compute(scheme.tFinal)
plt.plot(scheme.x, scheme.uFinal)
plt.plot(scheme.x, scheme.uF, marker = "o", markersize=5, linestyle = "None")

print("Erreur RMS: ", diffAbs(scheme.uFinal,scheme.uF))
print("Erreur de conservation : ", conservation(scheme.uF,scheme.u0(scheme.x),scheme))


# test 3

# In[6]:


test = tst.Test3()
scheme = ng.MUSCL(test)
scheme.form = "KT"
scheme.compute(scheme.tFinal)
plt.plot(scheme.x, scheme.uFinal)
plt.plot(scheme.x, scheme.uF, marker = "o", markersize=5, linestyle = "None")

print("Erreur RMS: ", diffAbs(scheme.uFinal,scheme.uF))
print("Erreur de conservation : ", conservation(scheme.uF,scheme.u0(scheme.x),scheme))


# test 4

# In[7]:


test = tst.Test4()
scheme = ng.MUSCL(test)
scheme.form = "KT"
scheme.compute(scheme.tFinal)
plt.plot(scheme.x, scheme.uFinal)
plt.plot(scheme.x, scheme.uF, marker = "o", markersize=5, linestyle = "None")

print("Erreur RMS: ", diffAbs(scheme.uFinal,scheme.uF))
print("Erreur de conservation : ", conservation(scheme.uF,scheme.u0(scheme.x),scheme))


# test 5

# In[8]:


test = tst.Test5()
scheme = ng.MUSCL(test)
scheme.form = "KT"
scheme.compute(scheme.tFinal)
plt.plot(scheme.x, scheme.uFinal)
plt.plot(scheme.x, scheme.uF, marker = "o", markersize=5, linestyle = "None")

print("Erreur RMS: ", diffAbs(scheme.uFinal,scheme.uF))
print("Erreur de conservation : ", conservation(scheme.uF,scheme.u0(scheme.x),scheme))


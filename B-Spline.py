#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 22:43:35 2018

@author: ryuma
"""

import numpy as np
from scipy import interpolate

import matplotlib.pyplot as plt


ctr =np.array( [(0,0),(1,0.1),(2,0.1),(3,0 ),(4,1),(5, 2)])
x=ctr[:,0]
y=ctr[:,1]

#x=np.append(x,x[0])
#y=np.append(y,y[0])

tck,u = interpolate.splprep([x,y],k=3,s=0)
u=np.linspace(0,1,num=50,endpoint=True)
out = interpolate.splev(u,tck)

plt.figure()
plt.plot(x, y, 'ro', out[0], out[1], 'b')
plt.legend(['Points', 'Interpolated B-spline', 'True'],loc='best')
plt.axis([min(x)-1, max(x)+1, min(y)-1, max(y)+1])
plt.title('B-Spline interpolation')
plt.show()
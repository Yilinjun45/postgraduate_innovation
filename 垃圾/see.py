import sys, os
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import importlib.util
import lumapi
from scipy.interpolate import interpolate
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
fdtd = lumapi.FDTD(r"D:\桌面\大创\大创优化\垃圾\multilayer_absorber.fsp")
R = -fdtd.transmission('R')
T = fdtd.transmission('T')
A=1-R-T
f=fdtd.getdata('R','f').squeeze()
plt.plot(f,A)
plt.show()


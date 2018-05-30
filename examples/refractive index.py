# -*- coding: utf-8 -*-
"""
Created on Tue May 29 08:43:47 2018

@author: XuBo
"""

from __future__ import division, print_function

import sys

sys.path.append(r'C:\Users\xub\Desktop\Python project\Packages')
#sys.path.append(r'E:\xbl_Berry\Desktop\Python project\Packages')

import numpy as np
import matplotlib.pyplot as plt

from lib.crystals.crystals import crystal
from lib.crystals.data import lbo3

deltas = np.arange(0,45,0.1)

a1 = []
a2 = []

for delta in deltas:

    lbo = crystal(lbo3, 633, 25, 90, 0+delta)
    a1.append((lbo.nhl[0]-lbo.nhl[1]))
    
    lbo = crystal(lbo3, 633, 25, 90, 90-delta)
    a2.append((lbo.nhl[0]-lbo.nhl[1]))
    
fig, ax = plt.subplots()

ax.plot(deltas, a1)
ax.plot(deltas, a2)


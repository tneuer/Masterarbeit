#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2020-10-26 18:01:12
    # Description :
####################################################################################
"""

import os

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 100)
x1 = x[x>0]
x2 = x[x<1]
y1 = np.log(x1)
y2 = np.log(1-x2)
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(x2, y2, label=r"Original:         $\log(1 - D(G(z; \phi); \theta))$")
ax.plot(x1, y1, label=r"Transformed: $\log(D(G(z; \phi); \theta))$")
ax.set_xlabel(r"Discriminator prediction $D(G(z; \phi)$")
ax.set_ylabel(r"Generator loss $\mathcal{L}(D, G)$")
ax.legend()
plt.savefig("../../Thesis/figures/GAN/genLoss.png")

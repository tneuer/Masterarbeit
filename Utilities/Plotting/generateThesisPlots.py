#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-10-21 09:59:35
    # Description :
####################################################################################
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


p = np.linspace(0.000001, 1, 1000)
plt.plot(p, np.log(1-p), label="log(1-p)")
plt.plot(p, np.log(p), label="log(p")
plt.legend()
plt.xlabel("D(x) = D(G(z))")
plt.ylabel("Generator loss")
# plt.savefig("/home/tneuer/Backup/Uni/Masterarbeit/figures/GAN/genLoss.png")
plt.show()

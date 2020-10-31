#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2020-10-29 15:43:52
    # Description :
####################################################################################
"""

import os
import sys
sys.path.insert(1, "../Utilities")
sys.path.insert(1, "../Preprocessing")
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import scipy.stats as stats
import matplotlib.pyplot as plt

savefolder = "../../Thesis/figures/"
#############################################################################################################
############ Wasserstein vs KL & JS
#############################################################################################################
mu1, sigma1 = 0, 2
mu2, sigma2 = 10, 4
x1 = np.linspace(mu1 - 4*sigma1, mu1 + 4*sigma1, 100)
x2 = np.linspace(mu2 - 4*sigma2, mu2 + 4*sigma2, 100)

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(x1, stats.norm.pdf(x1, mu1, sigma1), lw=2, color="k")
ax.fill_between(x1, stats.norm.pdf(x1, mu1, sigma1), alpha=0.6)
ax.plot(x2, stats.norm.pdf(x2, mu2, sigma2), lw=2, color="k")
ax.fill_between(x2, stats.norm.pdf(x2, mu2, sigma2), alpha=0.6)
ax.annotate("", xy=(2.5, 0.095), xytext=(8, 0.095), arrowprops=dict(arrowstyle="<->"))
ax.annotate("Wasserstein", xy=(2.5, 0.1))
ax.annotate("", xy=(3.7, 0), xytext=(3.7, 0.03), arrowprops=dict(arrowstyle="<->"))
ax.annotate("KL / JS", xy=(4.4, 0.026))
ax.axis("off")
plt.savefig(savefolder+"Loss/wgan_js.png")
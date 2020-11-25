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
import scipy.stats as stats
import matplotlib.pyplot as plt

from scipy.stats import entropy
from numpy.linalg import norm



#############################################################################################################
# Wasserstein vs KL & JS
#############################################################################################################
# mu1, sigma1 = 0, 2
# mu2, sigma2 = 10, 4
# x1 = np.linspace(mu1 - 4*sigma1, mu1 + 4*sigma1, 100)
# x2 = np.linspace(mu2 - 4*sigma2, mu2 + 4*sigma2, 100)

# fig, ax = plt.subplots(nrows=1, ncols=1)
# ax.plot(x1, stats.norm.pdf(x1, mu1, sigma1), lw=2, color="k")
# ax.fill_between(x1, stats.norm.pdf(x1, mu1, sigma1), alpha=0.6)
# ax.plot(x2, stats.norm.pdf(x2, mu2, sigma2), lw=2, color="k")
# ax.fill_between(x2, stats.norm.pdf(x2, mu2, sigma2), alpha=0.6)
# ax.annotate("", xy=(2.5, 0.095), xytext=(8, 0.095), arrowprops=dict(arrowstyle="<->"))
# ax.annotate("Wasserstein", xy=(2.5, 0.1))
# ax.annotate("", xy=(3.7, 0), xytext=(3.7, 0.03), arrowprops=dict(arrowstyle="<->"))
# ax.annotate("KL / JS", xy=(4.4, 0.026))
# ax.axis("off")
# plt.savefig("../../Thesis/figures/Loss/wgan_js.png")


#####################################################################################################
# Generator loss modification plot
#####################################################################################################
# x = np.linspace(0, 1, 100)
# x1 = x[x>0]
# x2 = x[x<1]
# y1 = np.log(x1)
# y2 = np.log(1-x2)
# fig, ax = plt.subplots(nrows=1, ncols=1)
# ax.plot(x2, y2, label=r"Original:         $\log(1 - D(G(z; \phi); \theta))$")
# ax.plot(x1, y1, label=r"Transformed: $\log(D(G(z; \phi); \theta))$")
# ax.set_xlabel(r"Discriminator prediction $D(G(z; \phi)$")
# ax.set_ylabel(r"Generator loss $\mathcal{L}(D, G)$")
# ax.legend()
# plt.savefig("../../Thesis/figures/GAN/genLoss.png")


#####################################################################################################
# Forward and backward KL loss
#####################################################################################################

# fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 12))

# x = np.linspace(-6, 6, 100)
# y_true = stats.norm.pdf(x, -2, 0.8) + stats.norm.pdf(x, 2, 0.8)
# for ax in np.ravel(axs):
#     ax.axis("off")
#     ax.plot(x, y_true, lw=2, color="blue")
#     ax.fill_between(x, y_true, alpha=0.6)
#     ax.annotate(r"$P_r(x)$", xy=(0.5, 0.4), color="blue")


# y1 = stats.norm.pdf(x, -1.9, 1)*1.4
# axs[0, 0].plot(x, y1, lw=2, color="green")
# axs[0, 0].fill_between(x, y1, alpha=0.6, color="palegreen")
# axs[0, 0].annotate(r"$P_g(x)$", xy=(-4.3, 0.2), color="green")
# axs[0, 0].annotate("", xy=(2, 0.0), xytext=(3.5, 0.3), arrowprops=dict(arrowstyle="->", color="darkred"))
# axs[0, 0].annotate("", xy=(3.5, 0.0), xytext=(3.6, 0.3), arrowprops=dict(arrowstyle="->", color="darkred"))
# axs[0, 0].annotate("", xy=(5, 0.0), xytext=(3.7, 0.3), arrowprops=dict(arrowstyle="->", color="darkred"))
# axs[0, 0].annotate(r"Explodes for $P_r > 0$"+"\n"+r"and $P_g \approx 0$", xy=(3.1, 0.32), color="darkred")

# axs[1, 1].plot(x, y1, lw=2, color="green")
# axs[1, 1].fill_between(x, y1, alpha=0.6, color="palegreen")
# axs[1, 1].annotate(r"$P_g(x)$", xy=(-4.3, 0.2), color="green")

# y2 = stats.norm.pdf(x, -0.1, 2.5)
# axs[0, 1].plot(x, y2, lw=2, color="green")
# axs[0, 1].fill_between(x, y2, alpha=0.6, color="palegreen")
# axs[0, 1].annotate(r"$P_g(x)$", xy=(0, 0.17), color="green")
# axs[0, 1].annotate("", xy=(-6, 0.0), xytext=(3.5, 0.3), arrowprops=dict(arrowstyle="->", color="darkred"))
# axs[0, 1].annotate("", xy=(5.4, 0.0), xytext=(3.6, 0.3), arrowprops=dict(arrowstyle="->", color="darkred"))
# axs[0, 1].annotate("", xy=(6, 0.0), xytext=(3.7, 0.3), arrowprops=dict(arrowstyle="->", color="darkred"))
# axs[0, 1].annotate(r"Explodes for $P_r \approx 0$"+"\n"+r"and $P_g > 0$", xy=(3.1, 0.32), color="darkred")
# axs[1, 0].plot(x, y2, lw=2, color="green")
# axs[1, 0].fill_between(x, y2, alpha=0.6, color="palegreen")
# axs[1, 0].annotate(r"$P_g(x)$", xy=(0, 0.17), color="green")


# line2 = plt.Line2D((.515,.515),(.1,.9), color="k", linewidth=0.5, linestyle="--")
# line1 = plt.Line2D((.52,.52),(.1,.9), color="k", linewidth=0.5, linestyle="--")
# fig.add_artist(line1)
# fig.add_artist(line2)
# axs[0, 0].set_title(r"Forward KL$[P_r||P_g]$ = $\int P_r \cdot\dfrac{P_r}{P_g}$dx")
# axs[0, 1].set_title(r"Reversed KL$[P_g||P_r]$ = $\int P_g \cdot\dfrac{P_g}{P_r}$dx")
# plt.savefig("../../Thesis/figures/Loss/KLandRevKL.png")


#####################################################################################################
# Jensen-Shannon divergence
#####################################################################################################

# def JSD(P, Q):
#     _P = P / norm(P, ord=1)
#     _Q = Q / norm(Q, ord=1)
#     _M = 0.5 * (_P + _Q)

#     _M[_M<0.0001] = 0.0000001
#     _P[_P<0.0001] = 0.0000001
#     KL_PM = entropy(_P, _M)
#     KL_QM = entropy(_Q, _M)
#     return 0.5 * (KL_PM + KL_QM)

# x = np.linspace(-5, 30, 200)
# y_true = stats.norm.pdf(x, 0, 1)

# mus = np.linspace(0, 25, 200)
# jsd = np.array([JSD(y_true, stats.norm.pdf(x, mu, 1)) for mu in mus])

# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
# fig.subplots_adjust(left=0.05, bottom=0.07, right=0.99, top=0.99, wspace=None, hspace=None)
# ax.plot(mus, jsd)
# ax.set_xlabel(r"Distance in $\mu$")
# ax.set_ylabel(r"Jensen-Shannon divergence JSD($P_r, P_g$)")
# ax.annotate(r"$\log(2)$", xy=(2, 0.7), color="red")

# ax_inner = fig.add_subplot(336)
# ax_inner.plot(x, y_true, lw=1, color="blue")
# ax_inner.fill_between(x, y_true, alpha=0.6)
# ax.plot([0, 0], [0, np.log(2)], color="blue", linestyle="--", linewidth=1)

# ax_inner.plot(x, stats.norm.pdf(x, 2, 1), lw=1, color="orange")
# ax_inner.fill_between(x, stats.norm.pdf(x, 2, 1), alpha=0.6)
# ax.plot([2, 2], [0, np.log(2)], color="orange", linestyle="--", linewidth=1)

# ax_inner.plot(x, stats.norm.pdf(x, 15, 1), lw=1, color="green")
# ax_inner.fill_between(x, stats.norm.pdf(x, 15, 1), alpha=0.6)
# ax.plot([15, 15], [0, np.log(2)], color="green", linestyle="--", linewidth=1)

# ax_inner.plot(x, stats.norm.pdf(x, 20, 1), lw=1, color="red")
# ax_inner.fill_between(x, stats.norm.pdf(x, 20, 1), alpha=0.6)
# ax.plot([20, 20], [0, np.log(2)], color="red", linestyle="--", linewidth=1)

# ax_inner.plot(x, stats.norm.pdf(x, 25, 1), lw=1, color="purple")
# ax_inner.fill_between(x, stats.norm.pdf(x, 25, 1), alpha=0.6)
# ax.plot([25, 25], [0, np.log(2)], color="purple", linestyle="--", linewidth=1)

# ax.plot([0, 25], [np.log(2)+0.001, np.log(2)+0.001], color="red", linestyle="--", linewidth=0.5)
# plt.savefig("../../Thesis/figures/Loss/JSdiv.png")





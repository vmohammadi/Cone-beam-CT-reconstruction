# -*- coding: utf-8 -*-
"""
Program for practicing the basics of CT reconstruction

"""

!pip install NumPy SciPy pylops

import matplotlib.pyplot as plt
import numpy as np
from numba import jit
import pylops

plt.close("all")
np.random.seed(10)

#%% Functions

@jit(nopython=True)
def radoncurve(x, r, theta):
    return (
        (r - ny // 2) / (np.sin(theta) + 1e-15)
        + np.tan(np.pi / 2.0 - theta) * x
        + ny // 2
    )
#%% Read Shepp Logan Phantom 

!wget https://github.com/matteo-ronchetti/torch-radon/raw/master/examples/phantom.npy
x = np.load("phantom.npy")

#%% Radon transform
x = x / x.max()
nx, ny = x.shape

ntheta = 151
theta = np.linspace(0.0, np.pi, ntheta, endpoint=False)

RLop = pylops.signalprocessing.Radon2D(
    np.arange(ny),
    np.arange(nx),
    theta,
    kind=radoncurve,
    centeredh=True,
    interp=False,
    engine="numba",
    dtype="float64",
)

y = RLop.H * x

#%% Perform back-projection

xrec = RLop * y

fig, axs = plt.subplots(1, 3, figsize=(10, 4))
axs[0].imshow(x.T, vmin=0, vmax=1, cmap="gray")
axs[0].set_title("Model")
axs[0].axis("tight")
axs[1].imshow(y.T, cmap="gray")
axs[1].set_title("Data")
axs[1].axis("tight")
axs[2].imshow(xrec.T, cmap="gray")
axs[2].set_title("Adjoint model in PyLops")
axs[2].axis("tight")
fig.tight_layout()

#%% Reconstruction

Dop = [
    pylops.FirstDerivative(
        (nx, ny), axis=0, edge=True, kind="backward", dtype=np.float64
    ),
    pylops.FirstDerivative(
        (nx, ny), axis=1, edge=True, kind="backward", dtype=np.float64
    ),
]
D2op = pylops.Laplacian(dims=(nx, ny), edge=True, dtype=np.float64)

# L2 technique
xinv_sm = pylops.optimization.leastsquares.regularized_inversion(
    RLop.H, y.ravel(), [D2op], epsRs=[1e1], **dict(iter_lim=20)
)[0]
xinv_sm = np.real(xinv_sm.reshape(nx, ny))

# TV technique
mu = 1.5
lamda = [1.0, 1.0]
niter = 3
niterinner = 4

xinv = pylops.optimization.sparsity.splitbregman(
    RLop.H,
    y.ravel(),
    Dop,
    niter_outer=niter,
    niter_inner=niterinner,
    mu=mu,
    epsRL1s=lamda,
    tol=1e-4,
    tau=1.0,
    show=False,
    **dict(iter_lim=20, damp=1e-2)
)[0]
xinv = np.real(xinv.reshape(nx, ny))

fig, axs = plt.subplots(1, 3, figsize=(10, 4))
axs[0].imshow(x.T, vmin=0, vmax=1, cmap="gray")
axs[0].set_title("Model")
axs[0].axis("tight")
axs[1].imshow(xinv_sm.T, vmin=0, vmax=1, cmap="gray")
axs[1].set_title("L2 Inversion")
axs[1].axis("tight")
axs[2].imshow(xinv.T, vmin=0, vmax=1, cmap="gray")
axs[2].set_title("TV-Reg Inversion")
axs[2].axis("tight")
fig.tight_layout()
#!/usr/bin/env python
"""Permutation test: is the arc-length concavity distinguishable from shuffled-label null?"""
import numpy as np, h5py
from scipy.interpolate import CubicSpline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

H5 = "/ocean/projects/cis250201p/jjain2/data/latents_ltfv6_navtest.h5"
N_COMPONENTS, N_BINS, N_GRID, BIN_MIN = 64, 30, 2000, 3.0
LAYER = "backbone_transformers_2"
LABEL = "dist_nearest_front_halfplane"
N_PERM = 200

def centroids(pcs, t, n_bins):
    order = np.argsort(t)
    chunks = np.array_split(order, n_bins)
    ct = np.array([t[c].mean() for c in chunks if len(c) > 0])
    cp = np.array([pcs[c].mean(axis=0) for c in chunks if len(c) > 0])
    keep = np.concatenate([[True], np.diff(ct) > 1e-9])
    return ct[keep], cp[keep]

def conc(pcs, y):
    ct, cp = centroids(pcs, np.log(y), N_BINS)
    sp = CubicSpline(ct, cp, axis=0, bc_type="natural")
    gt = np.linspace(ct.min(), ct.max(), N_GRID); gp = sp(gt)
    cum = np.concatenate([[0.], np.cumsum(np.linalg.norm(np.diff(gp,axis=0),axis=1))])
    cal = np.interp(ct, gt, cum); cd = np.exp(ct)
    dd = (cd-cd.min())/(cd.max()-cd.min())
    Ln = (cal-cal.min())/(cal.max()-cal.min())
    integ = np.trapezoid if hasattr(np,"trapezoid") else np.trapz
    return float(integ(Ln-dd, dd))

with h5py.File(H5) as h:
    yr = h[LABEL][:]; X = h[LAYER][:]
v = np.isfinite(yr) & (yr >= BIN_MIN)
y = yr[v].astype(np.float64); X = X[v].astype(np.float64)
pcs = StandardScaler().fit_transform(
    PCA(n_components=min(N_COMPONENTS, X.shape[0]-1, X.shape[1]))
    .fit_transform(StandardScaler().fit_transform(X)))

real = conc(pcs, y)
rng = np.random.default_rng(0)
null = np.array([conc(pcs, rng.permutation(y)) for _ in range(N_PERM)])
p = (np.sum(null >= real) + 1) / (N_PERM + 1)
z = (real - null.mean()) / null.std()

print(f"layer={LAYER}  label={LABEL}  n={len(y)}  n_perm={N_PERM}")
print(f"real concavity       = {real:+.4f}")
print(f"null mean +- std     = {null.mean():+.4f} +- {null.std():.4f}")
print(f"null 95th percentile = {np.percentile(null,95):+.4f}")
print(f"z-score vs null      = {z:+.2f}")
print(f"permutation p-value  = {p:.4f}")
print("  p<0.05 & z>2  => concavity is distance-specific (real signal)")
print("  p~0.5 & z~0   => concavity is a method artifact (shuffle reproduces it)")

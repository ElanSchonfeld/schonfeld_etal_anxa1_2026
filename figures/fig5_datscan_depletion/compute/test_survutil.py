#!/usr/bin/env python
"""Unit tests for concordance_index."""
import numpy as np
from survutil import concordance_index as C

t = [1, 2, 3, 4]; e = [1, 1, 1, 1]; r = [4, 3, 2, 1]
assert abs(C(t, e, r) - 1.0) < 1e-12, C(t, e, r)

assert abs(C(t, e, [1, 2, 3, 4]) - 0.0) < 1e-12

assert abs(C(t, e, [7, 7, 7, 7]) - 0.5) < 1e-12

rng = np.random.default_rng(0)
for _ in range(200):
    n = rng.integers(5, 40)
    tt = rng.exponential(5, n); ee = rng.binomial(1, 0.6, n); rr = rng.normal(size=n)
    a, b = C(tt, ee, rr), C(tt, ee, -rr)
    if np.isnan(a): continue
    assert abs(a + b - 1.0) < 1e-9, (a, b)

assert abs(C([1, 2, 3], [1, 0, 1], [3, 1, 2]) - 1.0) < 1e-12
assert abs(C([1, 2, 3], [1, 0, 1], [3, 4, 2]) - 0.5) < 1e-12

assert np.isnan(C([1, 2, 3], [0, 0, 0], [3, 1, 2]))

print("all concordance_index tests passed")

from survutil import censor_km, uno_c, td_auc, brier_ipcw, breslow_baseline, surv_at

for _ in range(50):
    n = rng.integers(6, 40)
    tt = rng.exponential(5, n); ee = np.ones(n, int); rr = rng.normal(size=n)
    a, b = C(tt, ee, rr), uno_c(tt, ee, rr)
    if np.isnan(a): continue
    assert abs(a - b) < 1e-9, (a, b)

g = censor_km([1, 2, 3], [1, 1, 1])[1]
assert np.allclose(g, 1.0), g
g = censor_km([1, 2, 3], [0, 0, 0])[1]
assert np.allclose(g, [2 / 3, 1 / 3, 0.0]), g

t = np.array([1., 2., 5., 6.]); e = np.array([1, 1, 0, 0])
assert abs(td_auc(t, e, np.array([9., 8., 2., 1.]), 3.0) - 1.0) < 1e-9
assert abs(td_auc(t, e, np.array([1., 2., 8., 9.]), 3.0) - 0.0) < 1e-9

assert abs(brier_ipcw(t, e, np.array([0., 0., 1., 1.]), 3.0)) < 1e-9
assert abs(brier_ipcw(t, e, np.array([1., 1., 0., 0.]), 3.0) - 1.0) < 1e-9

tt = np.array([1., 2., 3., 4.]); ee = np.array([1, 1, 1, 1]); lp = np.zeros(4)
grid, H0 = breslow_baseline(tt, ee, lp)
assert abs(H0[2] - (1 / 4 + 1 / 3 + 1 / 2)) < 1e-12, H0
assert abs(surv_at(grid, H0, lp, 3.0)[0] - np.exp(-(1 / 4 + 1 / 3 + 1 / 2))) < 1e-12

s = surv_at(grid, H0, np.array([-1., 0., 1., 2.]), 3.0)
assert np.all(np.diff(s) < 0), s

print("all IPCW metric tests passed")

from survutil import partial_loglik, cox_fit
import pandas as pd

t = np.array([1., 2., 3., 4.]); e = np.array([1, 1, 1, 1])
assert abs(partial_loglik(t, e, np.zeros(4)) - (-np.log(4) - np.log(3) - np.log(2) - np.log(1))) < 1e-12
assert abs(partial_loglik(t, e, np.zeros(4)) - partial_loglik(t, e, np.full(4, 3.7))) < 1e-9

rng2 = np.random.default_rng(7)
n = 300
x = rng2.normal(size=n); tt = rng2.exponential(np.exp(-0.5 * x)); ee = rng2.binomial(1, 0.7, n)
df = pd.DataFrame(dict(t=tt, e=ee, x=x))
m = cox_fit(df, ['x'], 't', 'e')
b = float(m.params[0])
assert abs(partial_loglik(tt, ee, b * x) - float(m.llf)) < 1e-6, (partial_loglik(tt, ee, b * x), m.llf)
for off in (-0.3, -0.1, 0.1, 0.3):
    assert partial_loglik(tt, ee, (b + off) * x) < partial_loglik(tt, ee, b * x)

print("all partial_loglik tests passed")

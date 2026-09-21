#!/usr/bin/env python
"""Small survival helpers so the analysis needs no package outside m2h_env (no lifelines/sksurv)."""
import numpy as np
import statsmodels.api as sm
from statsmodels.duration.hazard_regression import PHReg


def concordance_index(time, event, risk):
    time = np.asarray(time, float); event = np.asarray(event, int); risk = np.asarray(risk, float)
    order = np.argsort(time)
    time, event, risk = time[order], event[order], risk[order]
    num = den = 0.0
    for i in range(len(time)):
        if event[i] != 1: continue
        later = time > time[i]
        if not later.any(): continue
        r = risk[later]
        den += r.size
        num += (risk[i] > r).sum() + 0.5 * (risk[i] == r).sum()
    return num / den if den else np.nan


def cox_fit(df, cols, tcol, ecol):
    """Cox PH (Efron ties) on the given columns."""
    X = df[cols].to_numpy(float)
    return PHReg(df[tcol].to_numpy(float), X, status=df[ecol].to_numpy(int), ties='efron').fit()


def cox_lp(res, df, cols):
    """Linear predictor (log relative hazard) for new rows under a fitted Cox model."""
    return df[cols].to_numpy(float) @ np.asarray(res.params, float)


def lr_test(res_full, res_red, ddf):
    """Likelihood-ratio test of nested Cox models."""
    from scipy.stats import chi2
    stat = 2.0 * (res_full.llf - res_red.llf)
    return float(stat), float(chi2.sf(stat, ddf))


def zfit(train, cols):
    """Mean/SD from TRAIN only (no leakage), returned as a transform closure."""
    mu = train[cols].mean(); sd = train[cols].std(ddof=0).replace(0, 1.0)
    return lambda d: (d[cols] - mu) / sd


def censor_km(time, event):
    """Kaplan-Meier of the CENSORING distribution G(t) = P(C > t), as a step function."""
    time = np.asarray(time, float); event = np.asarray(event, int)
    grid = np.unique(time)
    n, g, out = len(time), 1.0, []
    for t in grid:
        at_risk = (time >= t).sum()
        cens = ((time == t) & (event == 0)).sum()
        if at_risk > 0 and cens > 0: g *= (1 - cens / at_risk)
        out.append(g)
    return grid, np.asarray(out)


def _gval(grid, G, t, floor=1e-8):
    """G evaluated at t-, i.e. the last grid value strictly before t (0 -> 1.0)."""
    t = np.atleast_1d(np.asarray(t, float))
    idx = np.searchsorted(grid, t, side='left') - 1
    v = np.where(idx < 0, 1.0, G[np.clip(idx, 0, len(G) - 1)])
    return np.maximum(v, floor)


def uno_c(time, event, risk, tau=None):
    time = np.asarray(time, float); event = np.asarray(event, int); risk = np.asarray(risk, float)
    grid, G = censor_km(time, event)
    if tau is None: tau = time.max()
    num = den = 0.0
    for i in range(len(time)):
        if event[i] != 1 or time[i] > tau: continue
        w = 1.0 / _gval(grid, G, time[i])[0] ** 2
        later = time > time[i]
        if not later.any(): continue
        r = risk[later]
        den += w * r.size
        num += w * ((risk[i] > r).sum() + 0.5 * (risk[i] == r).sum())
    return num / den if den else np.nan


def td_auc(time, event, risk, t0):
    """Uno's time-dependent AUC at horizon t0: cases are events by t0, controls survive past t0."""
    time = np.asarray(time, float); event = np.asarray(event, int); risk = np.asarray(risk, float)
    grid, G = censor_km(time, event)
    case = (time <= t0) & (event == 1)
    ctrl = time > t0
    if case.sum() == 0 or ctrl.sum() == 0: return np.nan
    w = 1.0 / _gval(grid, G, time[case])
    rc = risk[ctrl]
    num = sum(wi * ((ri > rc).sum() + 0.5 * (ri == rc).sum()) for wi, ri in zip(w, risk[case]))
    return float(num / (w.sum() * rc.size))


def brier_ipcw(time, event, surv_t0, t0):
    """Graf's IPCW Brier score at t0. surv_t0 is predicted P(T > t0) per subject."""
    time = np.asarray(time, float); event = np.asarray(event, int); s = np.asarray(surv_t0, float)
    grid, G = censor_km(time, event)
    died = (time <= t0) & (event == 1)
    alive = time > t0
    b = np.zeros(len(time))
    b[died] = (s[died] ** 2) / _gval(grid, G, time[died])
    b[alive] = ((1.0 - s[alive]) ** 2) / _gval(grid, G, np.full(alive.sum(), t0))
    return float(b.mean())


def breslow_baseline(time, event, lp):
    """Breslow cumulative baseline hazard from a fitted linear predictor."""
    time = np.asarray(time, float); event = np.asarray(event, int); lp = np.asarray(lp, float)
    ex = np.exp(lp - lp.max())
    shift = lp.max()
    grid = np.unique(time[event == 1])
    H, out = 0.0, []
    for t in grid:
        d = ((time == t) & (event == 1)).sum()
        rsum = ex[time >= t].sum()
        if rsum > 0: H += d / (rsum * np.exp(shift))
        out.append(H)
    return grid, np.asarray(out)


def surv_at(grid, H0, lp, t0):
    """Predicted S(t0 | x) = exp(-H0(t0) * exp(lp)) for each subject."""
    idx = np.searchsorted(grid, t0, side='right') - 1
    h = H0[idx] if idx >= 0 else 0.0
    return np.exp(-h * np.exp(np.asarray(lp, float)))


def partial_loglik(time, event, lp):
    time = np.asarray(time, float); event = np.asarray(event, int); lp = np.asarray(lp, float)
    mx = lp.max(); ex = np.exp(lp - mx)
    total = 0.0
    for t in np.unique(time[event == 1]):
        died = (time == t) & (event == 1)
        rsum = ex[time >= t].sum()
        if rsum <= 0: continue
        total += (lp[died] - mx).sum() - died.sum() * np.log(rsum)
    return float(total)

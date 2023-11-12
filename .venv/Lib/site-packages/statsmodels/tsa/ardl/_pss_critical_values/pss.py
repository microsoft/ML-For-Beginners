#!/usr/bin/env python
# coding: utf-8

from itertools import product
import os

import numpy as np

PATH = os.environ.get("PSS_PATH", "..")


def pss_block(
    seed, k, case, i1, block_id, m=2_000_000, t=1_000, save=True, path="./"
):
    file_name = f"pss-k-{k}-case-{case}-i1-{i1}-block-{block_id}.npz"
    file_name = os.path.join(path, file_name)
    if save and os.path.exists(file_name):
        return
    rs = np.random.default_rng(seed)
    const = np.ones(t - 1)
    tau = np.arange(1, t).astype(float)
    f = np.empty(m)
    for j in range(m):
        u = rs.standard_normal((k + 1, t))
        y = np.cumsum(u[0])
        if i1:
            x = np.cumsum(u[1:], axis=1).T
        else:
            x = u[1:].T
        lhs = np.diff(y)
        rhv = [y[:-1], x[:-1]]
        if case == 2:
            rhv.append(const)
        elif case == 4:
            rhv.append(tau)
        if case >= 3:
            rhv.append(const)
        if case == 5:
            rhv.append(tau)
        rest = k + 1
        if case in (2, 4):
            rest += 1
        rhs = np.column_stack(rhv)
        b = np.linalg.lstsq(rhs, lhs, rcond=None)[0]
        u = lhs - rhs @ b
        s2 = u.T @ u / (u.shape[0] - rhs.shape[1])
        xpx = rhs.T @ rhs
        vcv = np.linalg.inv(xpx) * s2
        r = np.eye(rest, rhs.shape[1])
        rvcvr = r @ vcv @ r.T
        rb = r @ b
        f[j] = rb.T @ np.linalg.inv(rvcvr) @ rb / rest
    percentiles = [0.05]
    percentiles += [i / 10 for i in range(1, 10)]
    percentiles += [1 + i / 2 for i in range(18)]
    percentiles += [i for i in range(10, 51)]
    percentiles += [100 - v for v in percentiles]
    percentiles = sorted(set(percentiles))
    percentiles = np.asarray(percentiles)
    q = np.percentile(f, percentiles)
    if save:
        np.savez(file_name, q=q, percentiles=percentiles)
    return q


seed = [
    3957597042,
    2709280948,
    499296859,
    1555610991,
    2390531900,
    2160388094,
    4098495866,
    47221919,
]
ss = np.random.SeedSequence(seed)
k = list(range(1, 11))
case = list(range(1, 6))
i1 = [True, False]
block_id = list(range(32))
params = list(product(k, case, i1, block_id))
seeds = ss.generate_state(8 * len(params)).reshape((-1, 8)).tolist()
configs = []
for _s, (_k, _case, _i1, _block_id) in zip(seeds, params):
    configs.append(
        {
            "seed": _s,
            "k": _k,
            "case": _case,
            "i1": _i1,
            "block_id": _block_id,
            "path": PATH,
        }
    )

if __name__ == "__main__":
    from joblib import Parallel, delayed

    Parallel(n_jobs=10)(delayed(pss_block)(**c) for c in configs)

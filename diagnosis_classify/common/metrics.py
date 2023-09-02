"""
Created by yizhi.chen.
"""

def dice(x, y):
    i, u = [t.sum() for t in [x * y, x + y]]
    dc = (2 * i + 1) / (u + 1)
    dc = dc.mean()
    return dc


def recall(x, y):
    rc = ((x * y).sum()+1e-8) / (y.sum() + 1e-8)
    return rc


def precision(x, y):
    pc = ((x * y).sum()+1e-8) / (x.sum() + 1e-8)
    return pc


def fbeta(x, y, beta=1):
    rc = recall(x, y)
    pc = precision(x, y)
    fs = (1 + beta ** 2) * pc * rc / (beta ** 2 * pc + rc + 1e-8)
    return fs


def fa(x, y):
    return (x * (1-y)).mean()

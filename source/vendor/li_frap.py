"""Numerical kernels from Li Pengbo / FRAP_by_time.py. Default constants retained; no equation changes."""
import numpy as np
K_ON=K_OFF=.02
PHI0=0.0

def partition_mu_eq(u, c0, p, alpha, uth):
    """同样依赖 u 的荧光物质平衡浓度: 富相内高、贫相内低"""
    sigma = 1.0 / (1.0 + np.exp(-alpha * (u - uth)))
    return c0 * (1.0 + (p - 1.0) * sigma)

def frap_step(mu, mu_eq, K2, D_mu, tau, dt):
    """半隐式 FRAP 步进: 扩散隐式 + 反应(趋向局部平衡)显式"""
    mu_hat = np.fft.fft2(mu)
    reaction = np.fft.fft2((mu - mu_eq) / tau)
    mu_hat = (mu_hat - dt * reaction) / (1.0 + dt * D_mu * K2)
    return np.real(np.fft.ifft2(mu_hat))

def frap_half_time(ch, u, params):
    """在冻结的凝聚场 u 上做中心 3×3 FRAP，返回 (τ_half, mu0, reached)。

    τ_half: 中心 μ 恢复至漂白前浓度一半 (0.5·mu0) 所需时间；
    reached: 是否在 T_max 内达到该值（否则 τ_half 为 NaN）。
    """
    N, c = ch.N, ch.N // 2
    dt = params["dt"]
    uth = params["uth"] if params["uth"] is not None else ch.m0
    mu_eq = partition_mu_eq(u, params["c0"], params["p"], params["alpha"], uth)

    # 预平衡: 让 μ 松弛到自身动力学的稳态，mu0 = 漂白前中心荧光浓度
    mu = mu_eq.copy()
    for _ in range(int(round(params["T_pre"] / dt))):
        mu = frap_step(mu, mu_eq, ch.K2, params["D_mu"], params["tau"], dt)
    mu0 = float(mu[c, c])

    # 漂白中心 3×3
    mu = mu.copy()
    mu[c - 1:c + 2, c - 1:c + 2] = 0.0

    target = 0.5 * mu0
    if mu0 <= 0:
        return np.nan, mu0, False

    n_max = int(round(params["T_max"] / dt))
    prev_v = 0.0  # t=0（刚漂白完）中心 μ = 0
    for k in range(1, n_max + 1):
        mu = frap_step(mu, mu_eq, ch.K2, params["D_mu"], params["tau"], dt)
        v = float(mu[c, c])
        if v >= target:
            # 线性插值求精确越过 0.5·mu0 的时刻
            t0, t1 = (k - 1) * dt, k * dt
            tau_half = t0 + (target - prev_v) / (v - prev_v) * (t1 - t0)
            return tau_half, mu0, True
        prev_v = v
    return np.nan, mu0, False
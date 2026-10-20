"""Numerical kernels from Li Pengbo / light_off/condensate_light.py. Default constants retained; no equation changes."""
import numpy as np
K_ON=K_OFF=.02
PHI0=0.0

class ProteinStateSwitch:
    """蛋白在失活态 A 与活化态 B 之间的切换 —— 单向一级反应。

    开灯:  A → B ,  dφ/dt = +k_on·(1 − φ)
    关灯:  B → A ,  dφ/dt = −k_off·φ          （φ = 活化态 B 所占比例）

    每个方向各自是单向一级反应（不含逆反应项），因此每段都是指数松弛，可写出
    解析解并逐步精确积分（无条件稳定、不限制 dt）：

        开灯:  φ ← 1 − (1 − φ)·exp(−k_on·dt)
        关灯:  φ ← φ·exp(−k_off·dt)

    切换不再是瞬间跳变，时间尺度分别为 1/k_on 与 1/k_off。
    """

    def __init__(self, k_on=K_ON, k_off=K_OFF, phi0=PHI0):
        self.k_on = float(k_on)
        self.k_off = float(k_off)
        self.phi = float(np.clip(phi0, 0.0, 1.0))

    def advance(self, on, dt):
        """推进一步，返回更新后的活化比例 φ。on=True 表示开灯。"""
        if on:
            self.phi = 1.0 - (1.0 - self.phi) * np.exp(-self.k_on * dt)
        else:
            self.phi = self.phi * np.exp(-self.k_off * dt)
        return self.phi

def light_on_at(step, on_steps, off_steps):
    """step 从 1 开始计；返回该步对应的光态（True=开灯）。默认从开灯开始。"""
    cyc = on_steps + off_steps
    r = (step - 1) % cyc + 1   # 1..cyc
    return r <= on_steps

def free_diffuse(u, K2, D_off, dt):
    """自由扩散 ∂u/∂t = D_off·∇²u（谱方法全隐式）。
    K2[0,0]=0 → DC 分量不变，总质量严格守恒；无条件稳定。"""
    u_hat = np.fft.fft2(u)
    u_hat = u_hat / (1.0 + D_off * K2 * dt)
    return np.real(np.fft.ifft2(u_hat))

def light_step(ch, u, phi, D_off, dt):
    """按当前活化比例 φ 混合两种动力学（平均场近似）：

        φ = 1（全活化） → 纯 Cahn-Hilliard（相分离驱动力全开）
        φ = 0（全失活） → 纯自由扩散（驱动力关闭，液滴溶解）
        0 < φ < 1       → 两者按 φ 线性插值（部分活化）

    两步都严格守恒总质量，故插值后依然守恒。
    """
    if phi >= 1.0:
        return ch.step(u)
    u_df = free_diffuse(u, ch.K2, D_off, dt)
    if phi <= 0.0:
        return u_df
    return phi * ch.step(u) + (1.0 - phi) * u_df
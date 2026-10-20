"""
Cahn-Hilliard 相分离（旋节线分解）模拟 —— 第四版

方程:  ∂u/∂t = ∇²[ l(D·u − k·u² − m·u³) − κ∇²u ]
       μ = ∂f/∂u − κ∇²u,  f(u) = l(D·u²/2 − k·u³/3 − m·u⁴/4)

相对第三版 (condensate3.py) 的改动:
  1. 原势 (potential='original') 中写死的系数 0.04 改为可调参数 m，
     f(u) 相应由 l(D·u²/2 − k·u³/3 − 0.01·u⁴) 改为 l(D·u²/2 − k·u³/3 − m·u⁴/4)；
     默认 m=0.04，与 v3 完全等价。
  2. quartic 势中表示守恒均值的参数 m 改名为 m0，避免与新的系数 m 冲突。
  3. GIF 时间采样由"纯对数"改为"对数+线性混合"：
     早期（旋节线放大/显影阶段）用对数采样保持帧密，捕捉指数增长；
     晚期（缓慢粗化/Ostwald 熟化）改用线性采样，使晚期采样频率更密，
     便于观察液滴合并与粗化细节（分界比例可用 gif_switch 调整）。

其余内容（半隐式谱方法、液滴统计、自由能监控等）与 condensate3.py 相同。
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

try:
    from scipy import ndimage
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    print("提示: 未安装 scipy，无法做液滴统计，只输出浓度场动画。")


# ---------- 初始化 ----------

def initialize_grid(N, seed='random', rng=None, mean=1.0):
    """u 是序参量，基态取 mean，允许出现负值。
    mean < 1 时富相占比变小，会形成"连续相中漂浮的孤立圆液滴"形貌"""
    rng = rng or np.random.default_rng()
    if seed == 'spot':
        # 中心一个方形种子 + 微小随机扰动（扰动让液滴后续能成核）
        u = np.full((N, N), mean) + 0.01 * rng.standard_normal((N, N))
        r = N // 20
        u[N//2-r:N//2+r, N//2-r:N//2+r] = mean + 0.2
    elif seed == 'random':
        u = np.full((N, N), mean) + 0.1 * rng.standard_normal((N, N))
    else:
        raise ValueError(f"未知 seed: {seed}")
    return u


# ---------- 半隐式谱方法步进 ----------

class CahnHilliard:
    """
    potential='quartic' (推荐): 对称双阱 f(u) = l·((u-m0)²-w²)²/4，
        μ = l·[(u-m0)³ − w²(u−m0)] − κ∇²u，两极小在 m0±w (m0=守恒均值)。
        有界的标准 φ⁴ 势，能产生真正的液滴与 Ostwald 粗化。
    potential='original': 保留 l(D·u − k·u² − m·u³) 的旧势。
        注意: 该势 f⁽⁴⁾<0 无下界、只有一个极小 (u=0)，u≈1 处于亚稳区，
        数学上不会自发相分离；仅用于与旧版对比。
        m 为 u⁴ 项的系数（v3 中写死为 0.04），可自由调整。
    """

    def __init__(self, N=128, L=100.0, D=2.2, k=1.0, l=0.05, kappa=0.035,
                 dt=0.5, potential='quartic', w=1.5, m=0.04, m0=1.0):
        self.N, self.L = N, L
        self.l, self.kappa, self.dt = l, kappa, dt
        self.potential, self.w = potential, w
        self.m = m      # original 势中 u⁴ 项的系数（旧版写死 0.04）
        self.m0 = m0    # quartic 势中守恒均值（两极小在 m0±w）
        self.h = L / N

        if potential == 'original':
            self.D_eff = D              # 线性部分系数 (implicit 用)
            self.k, self.D = k, D
            # μ 的非线性部分: −l·(k·u² + m·u³)
            self.mu_nl = lambda u: -l * (self.k * u**2 + self.m * u**3)
        elif potential == 'quartic':
            # (u−m0)³ − w²(u−m0) = u³ − 3m0·u² + (3m0²−w²)u + (w²m0 − m0³)
            # 隐式分母 1 + dt·K²·(l·D_eff + κK²) 的最小值在 K²* = −l·D_eff/(2κ) 取到,
            # 值为 1 − dt·l²D_eff²/(4κ); 只要它 >0 分母就恒正 —— 这才是正确判据。
            # D_eff<0 (双阱/失稳区) 完全允许, 半隐式谱方法处理旋节线分解的标准做法;
            # 旧卫语句 w²<3m0² 只覆盖 D_eff>0, 会把 ±1 约定 (m0=0) 的拟合运行挡掉。
            self.D_eff = 3 * m0**2 - w**2
            if dt * (l * self.D_eff) ** 2 / (4 * kappa) >= 1:
                raise ValueError("隐式分母会变号: 需 dt·l²·D_eff²/(4κ) < 1 (调小 dt 或加大 kappa)")
            self.mu_nl = lambda u: l * (u**3 - 3 * m0 * u**2 + (w**2 * m0 - m0**3))
        else:
            raise ValueError(f"未知 potential: {potential}")

        # 周期边界波数
        self.K = 2.0 * np.pi * np.fft.fftfreq(N, d=self.h)
        self.KX, self.KY = np.meshgrid(self.K, self.K)
        self.K2 = self.KX**2 + self.KY**2
        # K2[0,0]=0：k=0 模式分子为零、隐式分母为 1，总质量严格守恒

        # 隐式分母: 1 + dt·K²·(l·D_eff + κK²)
        self.implicit = 1.0 + dt * self.K2 * (l * self.D_eff + kappa * self.K2)

    def step(self, u):
        """一步半隐式更新：μ 的非线性部分显式，线性部分 (l·D_eff·u − κ∇²u) 隐式
        ∇²μ 的谱 = −K²·(F[l·D_eff]·û + F[μ_nl])，线性项移到左边隐式求解"""
        K2, dt = self.K2, self.dt
        u_hat = np.fft.fft2(u)
        u_hat = (u_hat - dt * K2 * np.fft.fft2(self.mu_nl(u))) / self.implicit
        return np.real(np.fft.ifft2(u_hat))

    def free_energy(self, u):
        """总自由能，用于监控演化方向"""
        if self.potential == 'original':
            f = self.l * (self.D * u**2 / 2 - self.k * u**3 / 3 - self.m * u**4 / 4)
        else:
            f = self.l * (((u - self.m0)**2 - self.w**2)**2 / 4)
        grad2 = (np.roll(u, 1, 0) - u)**2 + (np.roll(u, -1, 0) - u)**2 \
              + (np.roll(u, 1, 1) - u)**2 + (np.roll(u, -1, 1) - u)**2
        return f.sum() * self.h**2 + 0.5 * self.kappa * grad2.sum() * self.h**2


# ---------- 液滴统计 ----------

def droplet_stats(u, threshold):
    """阈值分割后统计液滴：返回 (平均等效半径, 液滴数)"""
    if not HAVE_SCIPY:
        return np.nan, np.nan
    mask = u > threshold
    if not mask.any():
        return 0.0, 0.0
    labels, n = ndimage.label(mask)
    areas = ndimage.sum_labels(np.ones_like(u), labels, index=np.arange(1, n + 1))
    radii = np.sqrt(areas / np.pi)  # 等效圆半径
    return float(radii.mean()), int(n)


# ---------- 主模拟 ----------

def simulate(N=128, D=2.2, k=1.0, l=0.05, kappa=0.035, dt=0.5,
             steps=5000, measure_every=50, seed='random',
             potential='quartic', w=1.5, m=0.04, mean=1.0,
             out_csv='droplet_rt.csv', show_animation=True,
             save_gif=None, gif_frames=300, gif_switch=0.2):
    """
    save_gif: 传入文件名（如 'droplets.gif'）则把整个演化导出为 GIF；
              None 则只弹交互窗口。gif_frames 控制 GIF 帧数（默认 200，采样较密，
              需要更密可继续调大）。
    mean:     初始均值。w=1.5 时富相占比 ≈ (mean+0.5)/3，
              mean=1 为双连续迷宫，mean≤0.4 为孤立圆液滴。
    m:        potential='original' 时 u⁴ 项的系数（v3 写死为 0.04）。
    gif_switch: 动画采样由对数切到线性的时间分界（占 steps 的比例，默认 0.2）。
              调小 → 更早进入线性段，晚期帧更密；调大 → 更多早期对数帧。
    """
    ch = CahnHilliard(N=N, L=100.0, D=D, k=k, l=l, kappa=kappa, dt=dt,
                      potential=potential, w=w, m=m)
    rng = np.random.default_rng(42)
    u = initialize_grid(N, seed=seed, rng=rng, mean=mean)
    u0_mean = u.mean()

    # 阈值: 取 spinodal 分解的贫相/富相分界（粗略用初值均值）
    threshold = u0_mean

    times, radii, ndrops, energies, masses = [], [], [], [], []
    print(f"{'t':>8} {'<R>':>8} {'液滴数':>7} {'自由能':>12} {'质量漂移':>10}")

    for t in range(1, steps + 1):
        u = ch.step(u)
        if not np.isfinite(u).all():
            raise RuntimeError(f"t={t} 出现 NaN/inf，请减小 dt")
        if t % measure_every == 0:
            r, n = droplet_stats(u, threshold)
            times.append(t * dt)
            radii.append(r)
            ndrops.append(n)
            energies.append(ch.free_energy(u))
            masses.append(u.mean() - u0_mean)
            print(f"{t*dt:8.1f} {r:8.3f} {n:7d} {energies[-1]:12.4f} {masses[-1]:10.2e}")

    # 输出 R(t) 数据
    if HAVE_SCIPY:
        np.savetxt(out_csv,
                   np.column_stack([times, radii, ndrops, energies, masses]),
                   delimiter=',', header='t,R_mean,n_droplets,free_energy,mass_drift',
                   comments='')
        print(f"\nR(t) 等数据已保存到 {out_csv}")

    if show_animation or save_gif:
        # 动画/GIF 从初始场重新开始演化（统计循环已经把 u 推到末态）
        u = initialize_grid(N, seed=seed, rng=np.random.default_rng(42), mean=mean)
        # 画布铺满整个图像（无边距），GIF 里浓度场占满每一帧
        fig = plt.figure(figsize=(6, 6), frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        im = ax.imshow(u, cmap='viridis', interpolation='bilinear',
                       vmin=u.min(), vmax=u.max(), animated=True)
        # 色标固定在双阱两极小值上：早期小幅噪声呈中间色(青绿)，
        # 液滴随旋节线放大逐渐"显影"，而不是一上来就饱和
        fixed_clim = potential == 'quartic'
        if fixed_clim:
            im.set_clim(ch.m0 - ch.w, ch.m0 + ch.w)

        # 对数-线性混合时间采样：
        #   早期（旋节线放大/显影阶段）用对数采样——指数增长在前几帧内
        #   就会饱和，线性采样看不到"从噪声中浮现"的过程，故早期帧密；
        #   晚期（缓慢粗化/Ostwald 熟化）改用线性采样——纯对数下晚期帧
        #   越来越疏，看不出液滴合并细节；线性采样让晚期采样频率更密。
        #   切换点由 gif_switch 控制（默认前 20% 时间走对数段）。
        if gif_frames <= 2 or steps <= 2:
            schedule = np.unique(np.linspace(1, steps, gif_frames).astype(int))
        else:
            t_split = min(max(2, int(steps * gif_switch)), steps - 1)  # 对数→线性切换点
            n_log = max(2, int(gif_frames * 0.5))       # 对数段帧数
            n_log = min(n_log, gif_frames - 1)
            n_lin = gif_frames - n_log                   # 线性段帧数
            early = np.geomspace(1, t_split, n_log).astype(int)
            late = np.linspace(t_split + 1, steps, n_lin).astype(int)
            schedule = np.unique(np.concatenate([early, late]))
        current = {'step': 0}

        def update(frame):
            nonlocal u
            target = schedule[frame]
            for _ in range(target - current['step']):
                u = ch.step(u)
            current['step'] = target
            im.set_array(u)
            if not fixed_clim:
                im.set_clim(u.min(), u.max())
            return [im]

        frames = len(schedule)
        _anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
        if save_gif:
            print(f"正在导出 GIF: {save_gif} ({frames} 帧, 对数+线性混合采样)...")
            _anim.save(save_gif, writer='pillow', fps=15, dpi=80)
            print(f"已保存 {save_gif}")
        if show_animation:
            plt.show()

    return u


if __name__ == '__main__':
    # 默认: 对称双阱势 (w=1.5, 两极小在 -0.5 与 2.5)，守恒均值 m0=1.0
    # 半隐式谱方法允许较大 dt；想看 Ostwald 粗化就把 steps 调大
    simulate(N=128, D=2.2, k=1.0, l=0.05, kappa=0.035, dt=0.5,
             steps=20000, measure_every=20, seed='random', mean=0.5,
             potential='quartic', w=1.5, m=0.04)

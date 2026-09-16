"""
词题1(源域:12kHz 驱动端| 滚动体 0007)的快速可复现实验代码
- 加载真实数据文件:B0071.mat(放在根目录)
- 管线:窗口化 → 稳健标准化 → 希尔伯特包絡 → 包絡谱 → 机理频帯指标与时域统计 → 可视化保存
- 结果输出到 result 文件夹
"""

import os
import numpy as np
import scipy.io as sio
import scipy.signal as sps
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 ['Microsoft YaHei'] 微软雅黑 等
plt.rcParams['axes.unicode_minus'] = False   # 解决负号 '-' 显示为方块的问题
# ======================
# 基本参数与工具函数
# ======================

try:
    os.chdir("./源域数据集/12kHz_DE_data/B/0007")
except Exception:
    pass

# 创建结果文件夹
out_dir = Path("result1")
out_dir.mkdir(exist_ok=True,parents=True)

#采样频率(题目中驱动端 12kHz)
FS = 12000.0

# 轴承几何参数(源域试验台数据给定：SKF6205 驱动端)
N_roll = 9            # 滚动体数
d_roll_inch = 0.3126  # 滚动体直径 （英寸）
D_pitch_inch = 1.537  # 轴承节径（英寸）
theta_deg = 0.0       # 接触角（度）
inch2m = 0.0254
d_roll = d_roll_inch*inch2m
D_pitch = D_pitch_inch * inch2m
theta = np.deg2rad(theta_deg)

# -----------------------
# MAD 稳健标准化
# -----------------------
def mad_std(x):
    '''计算中位数与 MAD 对应的稳健标准差'''
    med = np.median(x)
    mad = np.median(np.abs(x-med))
    return med, mad * 1.4826

def robust_standardize(x):
    """稳健标准化（对弱冲击更友好）"""
    med, sigma = mad_std(x)
    sigma = sigma if sigma > 1e-12 else 1.0
    return (x-med) / sigma

# -----------------------
# 估计转频（若RPM未提供）：在原始频谱低频段寻找主峰
# -----------------------
def estimate_fr(signal, fs,fmax=100.0):
    f, Pxx = sps.welch(signal, fs=fs, nperseg=8*1024,
                       noverlap=4*1024,nfft=8*1024,detrend='constant')
    mask = (f > 0.5) & (f < fmax)
    if np.sum(mask) <5:
        return  None
    f_low, P_low = f[mask],Pxx[mask]
    return f_low[np.argmax(P_low)]

# -----------------------
# 计算滚动体特征频率（BSF)
# -----------------------
def calc_bsf(fr,D,d,theta):
    # BSF = (D / (2d)) * fr * (1 - (d/D * cos(theta))^2)
    ratio = d/D *np.cos(theta)
    return (D / (2.0 * d)) * fr * (1.0 - ratio**2)

# -----------------------
# 包络与包络谱
# -----------------------
def envelope_and_spetrum(x,fs, nfft=2**15, window='hann'):
    """希尔伯特包络 + Welch 包络谱"""
    analytic = sps.hilbert(x)
    env = np.abs(analytic)
    env = sps.detrend(env,type='constant')
    f, Penv = sps.welch(env, fs=fs, window = window,nperseg = 4096, noverlap = 2048, nfft=nfft,detrend=False)
    return env, f, Penv

# -----------------------
# 频带积分能量与旁带不均衡
# -----------------------
def band_energy_ratio(f,S,fc,bw):
    mask = (f >= (fc - bw)) & (f <= (fc +bw))
    if not np.any(mask):
        return 0.0
    E_band = np.trapz(S[mask], f[mask])
    E_total = np.trapz(S, f)
    return (E_band / E_total) if E_total > 0 else 0.0

def sideband_asymmetry(f, S, fc, fr, bw):
    left_mask = (f >= (fc - fr - bw)) & (f <= (fc - fr +bw))
    right_mask = (f >= (fc + fr -bw)) & (f <= (fc + fr +bw))
    El = np.trapz(S[left_mask], f[left_mask]) if np.any(left_mask) else 0.0
    Er = np.trapz(S[right_mask], f[right_mask]) if np.any(right_mask) else 0.0
    demon = (El + Er) if (El + Er) > 0 else 1.0
    return (Er - El) / demon

# -----------------------
# 时域统计特征
# -----------------------
def time_states(x):
    x = x.astype(float)
    N = len(x)
    mean = np.mean(x)
    std = np.std(x, ddof = 1) if N > 1 else 0.0
    z = (x - mean) / (std +1e-12)
    skew = np.mean(z**3)
    kurt = np.mean(z**4)
    rms = np.sqrt(np.mean(x**2))
    max_abs = np.max(np.abs(x))
    crest = max_abs / (rms + 1e-12)
    impulse = max_abs / (np.mean(np.abs(x)) + 1e-12)
    return dict(mean = mean, std=std, skew=skew, kurt=kurt, crest=crest, impulse=impulse,rms=rms,max_abs=max_abs)

# -----------------------
# 绘图风格（尽量美观，浅色系）
# -----------------------
plt.rcParams.update({
    "figure.figsize":(10, 4),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.facecolor": "#FAFAFA",
    "savefig.dpi": 200
})

# -----------------------
# 1) 加载真实数据：B007_1.mat (根目录)
# -----------------------
mat_path = Path("E:/model/源域数据集/12kHz_DE_data/B/0007/B007_1.mat")
if not mat_path.exists():
    raise  FileNotFoundError("未找到文件")

mat = sio.loadmat(str(mat_path), squeeze_me=True,struct_as_record=False)

# 解析驱动端通道：优先匹配*_DE_time 或 DE_time
def extract_de_signal(matdict):
    for k in list(matdict.keys()):
        if k.endswith("_DE_time") or k == "DE_time":
            arr = np.asarray(matdict[k].ravel())
            if arr.size > 1000 and np.issubdtype(arr.dtype,np.number):
                return arr, k
    # 次优：包含"DE"的数值一维数组
    for k,v in matdict.items():
        if isinstance(v, np.ndarray) and v.ndim == 1 and v.size > 1000 and np.issubdtype(v.dtype,np.number):
            if "DE" in k.upper():
                return v, k
    # 兜底：最长的一维数值数组
    cand = None
    for k,v in matdict.items():
        if isinstance(v, np.ndarray) and v.ndim == 1 and v.size > 1000 and np.issubdtype(v.dtype,np.number):
            if (cand is None) or (v.size > cand[1].size):
                cand = (k,v)
    if cand is not None:
        return cand[1],cand[0]
    raise RuntimeError("无法从 .mat 中解析驱动端通道，请手动检查变量名")

# 解析转速（若提供）
def extract_rpm(matdict):
    for k in matdict.keys():
        if "RPM" in k.upper():
            v = np.asarray(matdict[k]).ravel()
            rpm = float(np.mean(v))
            if rpm > 0:
                return rpm
    return None

sig_de, key_used = extract_de_signal(mat)
rpm = extract_rpm(mat)
fr = rpm / 60.0 if rpm is not  None else estimate_fr(sig_de,FS,fmax=100.0)
if fr is None or fr <= 0:
    fr = 10.0 # 兜底

# 计算 BSF (滚动体特征频率)
bsf = calc_bsf(fr,D_pitch,d_roll,theta)

# ===============
# 2) 窗口化与特征抽取
# ===============
# 窗口长度：若bsf合理则取 16 个bsf周期，否则取2秒
Tw = 2.0 if bsf <=1.0 else float(np.clip(16/bsf,0.5,2.0))
Nw = int(Tw *FS)
Ns = int(0.5*Nw)  # 50% 重叠
N = len(sig_de)

# 预先绘制原始波形概览（前5秒）
t = np.arange(N) / FS
fig, ax = plt.subplots(figsize=(12,3.8))
ax.plot(t, sig_de, lw=0.8)
ax.set_title(f"驱动端原始波形（变量{key_used}）")
ax.set_xlabel("时间 / s")
ax.set_ylabel("加速度幅值")
ax.set_xlim(0,min(t[-1],5.0))
fig.tight_layout()
fig.savefig(out_dir / "wave_overview.png")
plt.close(fig)

# 滑窗切片
starts = np.arange(0, N - Nw +1, Ns)
window_level_rows = []

for idx, st in enumerate(starts):
    seg = sig_de[st:st+Nw].astype(float)
    # 稳健标准化
    seg_std = robust_standardize(seg)
    # 包络与包络谱
    env, f_env,S_env = envelope_and_spetrum(seg_std,FS,nfft=2**15)

    # 频带宽度：以包络谱分辨率为基准的倍数
    df = f_env[1] - f_env[0]
    bw = max(3 * df, 0.02 * max(bsf,1.0))  # 不小于 3 个频点

    # 机理频带能量比（BSF 及谐波）
    R1 = band_energy_ratio(f_env,S_env,bsf,bw)
    R2 = band_energy_ratio(f_env,S_env,2*bsf,bw)
    R3 = band_energy_ratio(f_env,S_env,3*bsf,bw)

    # 旁带不均衡（以转频为间隔）
    B1 = sideband_asymmetry(f_env,S_env,bsf,fr,bw)

    # 时域统计（基于稳健标准化后的片段）
    stats = time_states(seg_std)

    row = {
        "start_index":int(st),
        "start_time_s":st / FS,
        "fr_est_hz":fr,
        "bsf_est_hz":bsf,
        "band_bw_hz":bw,
        "R_BSF":R1,
        "R_2BSF":R2,
        "R_3BSF":R3,
        "B_asym_BSF":B1,
        **{f"time_{k}":v for k, v in stats.items()}
    }
    window_level_rows.append(row)

    # 可视化（进保存前4个窗口的详细图）
    if idx < 4:
        # 窗口波形与包络
        tt = np.arange(len(seg_std)) / FS
        fig, ax = plt.subplots(1,1,figsize=(12,3.8))
        ax.plot(tt, seg_std, lw=0.8, label="标准化波形")
        ax.plot(tt, env, lw=0.8, alpha=0.8, label="包络")
        ax.set_title(f"窗口{idx+1} 波形与包络")
        ax.set_xlabel("时间 / s")
        ax.set_ylabel("幅值（标准化）")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(out_dir / f"win{idx+1:02d}_wave_env.png")
        plt.close(fig)

        # 包络谱与机理频带标注
        fig, ax = plt.subplots(1,1,figsize=(12,4.0))
        ax.plot(f_env, S_env, lw=1.0)
        ax.set_xlim(0, min(1000,f_env[-1]))
        ax.set_xlabel("频率 / Hz")
        ax.set_ylabel("包络功率谱密度")
        ax.set_title(f"窗口{idx + 1} 包络谱（含 BSF 谐波与旁带）")
        for k in range(1,4):
            fk = k* bsf
            if fk < f_env[-1]:
                ax.axvline(fk, ls="--",lw=1.0,alpha=0.6,color="#444444")
                ax.text(fk,ax.get_ylim()[1]*0.85,f"{k}XBSF",rotation=90,va="top",ha="center",fontsize=9,color="#444444")
                # 旁带 正负fr
                if fk + fr < f_env[-1]:
                    ax.axvspan(fk+fr-bw, fk+fr+bw,color="#FFB6C1",alpha=0.35, lw=0)
                if fk - fr >0:
                    ax.axvspan(max(0,fk-fr-bw),fk-fr+bw,color="#ADD8E6",alpha=0.35, lw=0)
                # 机理频带
                ax.axvspan(max(0, fk - bw), fk + bw, color="#90EE90", alpha=0.25, lw=0)
            fig.tight_layout()
            fig.savefig(out_dir / f"win{idx + 1:02d}_wave_spectrum.png")
            plt.close(fig)

# 窗口级特征表
df_feat = pd.DataFrame(window_level_rows)
df_feat.to_csv(out_dir / "features_window_level.csv", index=False,encoding="utf-8-sig")

# 文件级稳健聚合（中位数与分位）
feature_q = ["R_BSF","R_2BSF","R_3BSF","B_asym_BSF"]
feature_med = ["time_mean","time_std","time_skew","time_kurt","time_crest","time_impulse","time_rms","time_max_abs"]

rows = []
for feat in feature_q:
    s = df_feat[feat].dropna()
    rows.append({"feature":feat,"stat":"median","value":float(np.median(s))})
    rows.append({"feature":feat,"stat":"q25","value":float(np.quantile(s,0.25))})
    rows.append({"feature":feat,"stat":"q75","value":float(np.quantile(s,0.75))})
for feat in feature_med:
    s = df_feat[feat].dropna()
    rows.append({"feature": feat, "stat": "median", "value": float(np.median(s))})

df_agg = pd.DataFrame(rows)
df_agg.to_csv(out_dir / "features_file_level_summary.csv", index=False,encoding="utf-8-sig")

# 汇总可视化：R_BSF 与 B_asym_BSF 的时序轨迹
fig, ax = plt.subplots(1,1,figsize=(12,3.6))
ax.plot(df_feat["start_time_s"], df_feat["R_BSF"],lw=1.0,label="BSF 带能量比")
ax.plot(df_feat["start_time_s"], df_feat["B_asym_BSF"],lw=1.0,label="BSF 旁带不均衡")
ax.set_xlabel("时间 / s（窗口起点）")
ax.set_ylabel("指标值）")
ax.set_title("窗口级机理指标时许轨迹")
ax.legend(loc="best")
fig.tight_layout()
fig.savefig(out_dir / "trajectory_R_BSF_B_asym.png")
plt.close(fig)

print("处理完毕：已在 result1/输出窗口级与文件级特征表，并保存若干可视化图片。")
print("关键文件：")
print(" - result1/features_window_level.csv")
print(" - result1/features_title_level_summary.csv")
print(" - result1/wave_overview.png")
print(" - result1/win01_wave_env.png, win01_spectrum.png 等（前四个窗口）")
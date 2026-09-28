"""
全面分析脚本（基于 desi-metadata-hdu1.mrt）
功能：
- 读取并清洗数据（处理 '--' / masked values）
- 计算 Eddington ratio（近似）
- 描述性统计（logL5100, MBH, lambda_Edd）
- 相关性与线性回归（BEL luminosities vs continuum）
- PCA + KMeans 聚类（发现子类）
- 绘图并打印关键结果
"""
import warnings
warnings.filterwarnings("ignore")

from astropy.table import Table
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, linregress
import os
print("当前工作目录：", os.getcwd())

path = os.path.join(os.path.dirname(__file__), "sdss-metadata-hdu2.mrt")


# -----------------------
# 1. 读取数据（请修改下面的路径！）
# -----------------------
# 这里的路径需要改成你电脑中 desi-metadata-hdu1.mrt 文件的实际位置
# 例如：如果文件和此代码在同一个文件夹，直接写 "desi-metadata-hdu1.mrt"
path = "../sdss/sdss-metadata-hdu2.mrt"  # <-- 重点修改这里
# main.py

print("读取：", path)
t = Table.read(path, format="ascii.cds")

# 转 pandas，处理缺失值
df = t.to_pandas()
df.replace({"--": np.nan}, inplace=True)

# 转换为数值类型（处理可能的非数值数据）
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    except Exception:
        pass

print(f"数据总行数 = {len(df)}；列名：{list(df.columns)}\n")

# -----------------------
# 2. 选择/构造关键列
# -----------------------
# 连续谱和谱线亮度列
cols = {}
for key in ["logL5100", "logL2500", "logLMgII", "logLHb", "logLHa", "zspec", "RAdeg", "DEdeg"]:
    cols[key] = key if key in df.columns else None

# 合并黑洞质量列（优先 Hb，再 Ha）
mbh_cols = []
if "logMBH-Hb" in df.columns:
    mbh_cols.append("logMBH-Hb")
if "logMBH-Ha" in df.columns:
    mbh_cols.append("logMBH-Ha")

if mbh_cols:
    df["logMBH"] = df[mbh_cols].bfill(axis=1).iloc[:, 0]  # 用第一个可用列填充
else:
    df["logMBH"] = np.nan

# -----------------------
# 3. 计算 Eddington ratio (近似)
# -----------------------
const_C = np.log10(9) - 38.100370550  # 单位换算常数
df["log_lambda_Edd"] = np.nan
# 筛选有亮度和黑洞质量数据的行
mask_for_lambda = df["logL5100"].notna() & df["logMBH"].notna()
df.loc[mask_for_lambda, "log_lambda_Edd"] = df.loc[mask_for_lambda, "logL5100"] - df.loc[mask_for_lambda, "logMBH"] + const_C
df["lambda_Edd"] = 10 ** df["log_lambda_Edd"]  # 转换为原始值

# -----------------------
# 4. 描述性统计
# -----------------------
def quick_stats(series, name):
    s = series.dropna()
    print(f"=== {name} ===")
    if len(s) == 0:
        print("无可用数据\n")
        return
    print("count:", len(s))
    print("median:", np.median(s))
    print("mean:", np.mean(s))
    print("std:", np.std(s))
    print("min, max:", s.min(), s.max(), "\n")

# 计算各关键指标的统计值
quick_stats(df["logL5100"], "logL5100")
quick_stats(df["logLMgII"], "logLMgII")
quick_stats(df["logLHb"], "logLHb")
quick_stats(df["logLHa"], "logLHa")
quick_stats(df["logMBH"], "logMBH")
quick_stats(df["log_lambda_Edd"], "log(lambda_Edd)")

print("物理提示：log_lambda_Edd 越小表明吸积率相对越低，若多数样本中位数接近 -2，支持文献中低爱丁顿比率相关性。\n")

# -----------------------
# 5. 相关性分析：谱线亮度 vs 连续谱亮度
# -----------------------
def correlate_and_plot(x, y, xlabel, ylabel, title, fname=None):
    ok = x.notna() & y.notna()  # 筛选有效数据
    if ok.sum() < 5:
        print(f"跳过 {title}（数据太少）\n")
        return
    xx = x[ok].values
    yy = y[ok].values
    # 计算相关性和线性回归
    r, p = pearsonr(xx, yy)
    lr = linregress(xx, yy)
    slope, intercept = lr.slope, lr.intercept

    print(f"{title}: N={len(xx)}, Pearson r={r:.3f} (p={p:.2e}), slope={slope:.3f}, intercept={intercept:.3f}")

    # 绘图
    plt.figure(figsize=(7,5))
    plt.scatter(xx, yy, s=20, alpha=0.6)
    xs = np.linspace(np.min(xx), np.max(xx), 100)
    plt.plot(xs, intercept + slope*xs, linestyle='--', label=f"y={slope:.2f}x+{intercept:.2f}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=150)
    plt.show()

# 连续谱与各谱线的相关性分析
correlate_and_plot(df["logL5100"], df["logLMgII"], "logL5100", "logLMgII", "logLMgII vs logL5100", "LMgII_vs_L5100.png")
correlate_and_plot(df["logL5100"], df["logLHb"], "logL5100", "logLHb", "logLHb vs logL5100", "LHb_vs_L5100.png")
correlate_and_plot(df["logL5100"], df["logLHa"], "logL5100", "logLHa", "logLHa vs logL5100", "LHa_vs_L5100.png")

# 若有 logL2500，补充分析
if "logL2500" in df.columns:
    correlate_and_plot(df["logL2500"], df["logLMgII"], "logL2500", "logLMgII", "logLMgII vs logL2500", "LMgII_vs_L2500.png")

# -----------------------
# 6. 天球分布 & 红移分布
# -----------------------
# 天球密度图（赤经 vs 赤纬）
plt.figure(figsize=(8,6))
ok = df["RAdeg"].notna() & df["DEdeg"].notna()
plt.hexbin(df.loc[ok, "RAdeg"], df.loc[ok, "DEdeg"], gridsize=80)
plt.xlabel("RA (deg)")
plt.ylabel("DEC (deg)")
plt.title("Sky density (RA vs DEC)")
plt.colorbar(label="counts")
plt.tight_layout()
plt.savefig("RA_DEC_hexbin.png")
plt.show()

# 红移分布图
plt.figure(figsize=(7,5))
plt.hist(df["zspec"].dropna(), bins=25)
plt.xlabel("zspec")
plt.ylabel("N")
plt.title("Redshift distribution")
plt.tight_layout()
plt.savefig("z_hist.png")
plt.show()

# -----------------------
# 7. PCA + KMeans 聚类（寻找子类）
# -----------------------
# 选择聚类用的特征列
feat_names = ["logL5100", "logLHb", "zspec", "log_lambda_Edd"]
feat_df = df[feat_names].copy()
feat_df = feat_df.dropna()  # 仅用完整数据
print("用于聚类的样本数：", len(feat_df))

if len(feat_df) >= 10:
    X = feat_df.values
    # 数据标准化
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # PCA降维到2维（方便可视化）
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(Xs)

    # KMeans聚类（3类）
    k = 3
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(Xp)

    # 绘制聚类结果
    plt.figure(figsize=(7,5))
    for lab in range(k):
        sel = labels == lab
        plt.scatter(Xp[sel,0], Xp[sel,1], s=30, alpha=0.7, label=f"cluster {lab}")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.title("PCA space colored by KMeans clusters")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pca_kmeans.png")
    plt.show()

    # 输出聚类统计结果
    feat_df["cluster"] = labels
    cluster_summary = feat_df.groupby("cluster").agg(["count", "mean", "median", "std"])
    print("\nCluster summary (mean & median):\n", cluster_summary.loc[:, (slice(None), ["mean","median"])])
    print("\n物理解释提示：观察各 cluster 在 log_lambda_Edd, logMBH 上的差异 -> 可能指示低吸积率子集 vs 高吸积率子集\n")

else:
    print("样本太少，跳过 PCA+KMeans\n")

# -----------------------
# 8. 结论提示 & 保存结果
# -----------------------
print("\n=== 初步结论提示（自动生成，需人工校验） ===")
print("1) 若 logLMgII / logLHb 与 logL5100 强相关（r ~ 0.8-1.0 且 slope ~ 1），支持 BEL 由中心连续谱主导（吸积率变化驱动）。")
print("2) 若 log_lambda_Edd 的中位数 ~ -2，则样本支持文献中 CL-AGNs 在低爱丁顿比率下更活跃/更变异的结论。")
print("3) PCA+KMeans 可能识别出若干子类（低 lambda_Edd、较高 MBH、或有显著线-连续不一致的天体），建议对每个 cluster 做光变曲线/多波段检查以确认物理机制（内在吸积率波动 vs 外部触发）。")
print("4) 注意样本选择/观测偏倚（例如低 z 下宿主星系污染、光纤口径差异）会影响统计，需要结合选择函数修正。")

# 保存聚类结果
feat_df.to_csv("cluster_features_with_labels.csv", index=False)
print("\n已保存 cluster_features_with_labels.csv 及若干图像文件（pca_kmeans.png, RA_DEC_hexbin.png, z_hist.png 等）。")

# -----------------------
# 新增：更深入的分析与图像
# -----------------------
# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 1. 爱丁顿比率 vs 红移（探索吸积率随宇宙时间的变化）
plt.figure(figsize=(8, 6))
ok = df["log_lambda_Edd"].notna() & df["zspec"].notna()
plt.scatter(df.loc[ok, "zspec"], df.loc[ok, "log_lambda_Edd"],
            s=30, alpha=0.6, c='purple')
# 添加中位数趋势线（按红移分箱）
z_bins = np.linspace(df.loc[ok, "zspec"].min(), df.loc[ok, "zspec"].max(), 10)
z_medians = []
lambda_medians = []
for i in range(len(z_bins) - 1):
    mask_bin = (df.loc[ok, "zspec"] >= z_bins[i]) & (df.loc[ok, "zspec"] < z_bins[i + 1])
    if mask_bin.sum() > 5:  # 每个箱至少5个样本
        z_medians.append(np.median(df.loc[ok, "zspec"][mask_bin]))
        lambda_medians.append(np.median(df.loc[ok, "log_lambda_Edd"][mask_bin]))
plt.plot(z_medians, lambda_medians, 'r--', linewidth=2, label='红移分箱中位数')
plt.xlabel("红移 z（越大离我们越远）")
plt.ylabel("log(爱丁顿比率)（越小吸积越慢）")
plt.title("吸积率随宇宙时间的变化（红移越大，时间越早）")
plt.grid(alpha=0.2)
plt.legend()
plt.tight_layout()
plt.savefig("lambda_vs_z.png", dpi=150)
plt.show()

# 2. 黑洞质量 vs 爱丁顿比率（按亮/暗态分组，探索质量与吸积率的关系）
# 假设论文中"亮态"指logL5100较高，这里用中位数划分
if "logL5100" in df.columns and len(df["logL5100"].dropna()) > 0:
    logL_median = df["logL5100"].median()
    mask_bright = (df["logL5100"] >= logL_median) & df["logMBH"].notna() & df["log_lambda_Edd"].notna()
    mask_dim = (df["logL5100"] < logL_median) & df["logMBH"].notna() & df["log_lambda_Edd"].notna()

    plt.figure(figsize=(8, 6))
    plt.scatter(df.loc[mask_bright, "logMBH"], df.loc[mask_bright, "log_lambda_Edd"],
                s=30, alpha=0.6, c='orange', label='亮态（高亮度）')
    plt.scatter(df.loc[mask_dim, "logMBH"], df.loc[mask_dim, "log_lambda_Edd"],
                s=30, alpha=0.6, c='blue', label='暗态（低亮度）')
    plt.xlabel("黑洞质量（log太阳质量）")
    plt.ylabel("log(爱丁顿比率)")
    plt.title("黑洞质量与吸积率的关系（按亮度分组）")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig("mbh_vs_lambda.png", dpi=150)
    plt.show()

# 3. 谱线亮度变化幅度 vs 连续谱亮度变化幅度（探索"不一致"的天体）
# 计算变化幅度（假设用亮态/暗态的比值，这里简化为log差）
if all(col in df.columns for col in ["logL5100", "logLMgII"]):
    df["delta_L5100"] = df["logL5100"] - df["logL5100"].median()  # 相对中位数的变化
    df["delta_LMgII"] = df["logLMgII"] - df["logLMgII"].median()
    ok_delta = df["delta_L5100"].notna() & df["delta_LMgII"].notna()

    plt.figure(figsize=(8, 6))
    plt.scatter(df.loc[ok_delta, "delta_L5100"], df.loc[ok_delta, "delta_LMgII"],
                s=30, alpha=0.6, c='green')
    # 画参考线y=x（表示变化一致）和偏离2倍标准差的异常区
    plt.plot([-2, 2], [-2, 2], 'k--', label='变化一致（y=x）')
    std = np.std(df.loc[ok_delta, "delta_L5100"] - df.loc[ok_delta, "delta_LMgII"])
    plt.fill_between([-2, 2], [-2 - 2 * std, 2 - 2 * std], [-2 + 2 * std, 2 + 2 * std],
                     color='gray', alpha=0.2, label=f'正常波动（±2σ）')
    plt.xlabel("5100Å连续谱亮度变化（相对中位数）")
    plt.ylabel("MgII谱线亮度变化（相对中位数）")
    plt.title("谱线与连续谱变化的一致性（偏离点可能是特殊天体）")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig("delta_line_vs_continuum.png", dpi=150)
    plt.show()

print(
    "\n已新增4幅深入分析图像：lambda_vs_z.png, mbh_vs_lambda.png, delta_line_vs_continuum.png, cluster_lambda_hist.png")
# -----------------------
# 新增：更深入的分析与图像（修复字体警告）
# -----------------------
# 注释掉中文显示设置（避免找不到字体的警告）
# plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
# plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 1. 爱丁顿比率 vs 红移（探索吸积率随宇宙时间的变化）
plt.figure(figsize=(8, 6))
ok = df["log_lambda_Edd"].notna() & df["zspec"].notna()
plt.scatter(df.loc[ok, "zspec"], df.loc[ok, "log_lambda_Edd"],
            s=30, alpha=0.6, c='purple')
# 添加中位数趋势线（按红移分箱）
z_bins = np.linspace(df.loc[ok, "zspec"].min(), df.loc[ok, "zspec"].max(), 10)
z_medians = []
lambda_medians = []
for i in range(len(z_bins) - 1):
    mask_bin = (df.loc[ok, "zspec"] >= z_bins[i]) & (df.loc[ok, "zspec"] < z_bins[i + 1])
    if mask_bin.sum() > 5:  # 每个箱至少5个样本
        z_medians.append(np.median(df.loc[ok, "zspec"][mask_bin]))
        lambda_medians.append(np.median(df.loc[ok, "log_lambda_Edd"][mask_bin]))
plt.plot(z_medians, lambda_medians, 'r--', linewidth=2, label='Redshift bin median')  # 英文标签
plt.xlabel("Redshift z (larger = farther)")  # 英文标签
plt.ylabel("log(Eddington ratio) (smaller = slower accretion)")  # 英文标签
plt.title("Accretion rate vs cosmic time")  # 英文标签
plt.grid(alpha=0.2)
plt.legend()
plt.tight_layout()
plt.savefig("lambda_vs_z.png", dpi=150)
plt.show()

# 2. 黑洞质量 vs 爱丁顿比率（按亮/暗态分组）
if "logL5100" in df.columns and len(df["logL5100"].dropna()) > 0:
    logL_median = df["logL5100"].median()
    mask_bright = (df["logL5100"] >= logL_median) & df["logMBH"].notna() & df["log_lambda_Edd"].notna()
    mask_dim = (df["logL5100"] < logL_median) & df["logMBH"].notna() & df["log_lambda_Edd"].notna()

    plt.figure(figsize=(8, 6))
    plt.scatter(df.loc[mask_bright, "logMBH"], df.loc[mask_bright, "log_lambda_Edd"],
                s=30, alpha=0.6, c='orange', label='Bright state')
    plt.scatter(df.loc[mask_dim, "logMBH"], df.loc[mask_dim, "log_lambda_Edd"],
                s=30, alpha=0.6, c='blue', label='Dim state')
    plt.xlabel("Black hole mass (log solar mass)")
    plt.ylabel("log(Eddington ratio)")
    plt.title("Black hole mass vs accretion rate")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig("mbh_vs_lambda.png", dpi=150)
    plt.show()

    import matplotlib.pyplot as plt
    import numpy as np

    # 假设 df 中有 "logMBH", "log_lambda_Edd", "redshift" 三列
    mask = df["logMBH"].notna() & df["log_lambda_Edd"].notna() & df["zspec"].notna()

    plt.figure(figsize=(8, 6))

    # 散点图，用红移决定颜色
    sc = plt.scatter(
        df.loc[mask, "logMBH"],
        df.loc[mask, "log_lambda_Edd"],
        c=df.loc[mask, "zspec"],  # 颜色由 redshift 决定
        cmap='viridis',  # 选择 colormap，可改为 'plasma', 'coolwarm' 等
        s=30,
        alpha=0.8
    )

    plt.xlabel("Black hole mass (log solar mass)")
    plt.ylabel("log(Eddington ratio)")
    plt.title("Black hole mass vs accretion rate (color by redshift)")
    plt.grid(alpha=0.2)
    plt.colorbar(sc, label="Redshift")  # 显示颜色条
    plt.tight_layout()
    plt.savefig("mbh_vs_lambda_redshift.png", dpi=150)
    plt.show()

# 3. 谱线亮度变化幅度 vs 连续谱亮度变化幅度
if all(col in df.columns for col in ["logL5100", "logLMgII"]):
    df["delta_L5100"] = df["logL5100"] - df["logL5100"].median()
    df["delta_LMgII"] = df["logLMgII"] - df["logLMgII"].median()
    ok_delta = df["delta_L5100"].notna() & df["delta_LMgII"].notna()

    plt.figure(figsize=(8, 6))
    plt.scatter(df.loc[ok_delta, "delta_L5100"], df.loc[ok_delta, "delta_LMgII"],
                s=30, alpha=0.6, c='green')
    plt.plot([-2, 2], [-2, 2], 'k--', label='Consistent change (y=x)')
    std = np.std(df.loc[ok_delta, "delta_L5100"] - df.loc[ok_delta, "delta_LMgII"])
    plt.fill_between([-2, 2], [-2 - 2 * std, 2 - 2 * std], [-2 + 2 * std, 2 + 2 * std],
                     color='gray', alpha=0.2, label=f'Normal fluctuation (±2σ)')
    plt.xlabel("5100Å continuum change (relative to median)")
    plt.ylabel("MgII line change (relative to median)")
    plt.title("Line vs continuum change consistency")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig("delta_line_vs_continuum.png", dpi=150)
    plt.show()

# 4. 聚类结果与爱丁顿比率的关系
if len(feat_df) >= 10 and "cluster" in feat_df.columns:
    plt.figure(figsize=(8, 6))
    for lab in feat_df["cluster"].unique():
        sel = feat_df["cluster"] == lab
        plt.hist(feat_df.loc[sel, "log_lambda_Edd"], bins=15,
                 alpha=0.6, label=f'Cluster {lab} (n={len(sel)})',
                 histtype='stepfilled')
    plt.xlabel("log(Eddington ratio)")
    plt.ylabel("Count")
    plt.title("Accretion rate distribution by cluster")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig("cluster_lambda_hist.png", dpi=150)
    plt.show()

print(
    "\n已新增4幅深入分析图像：lambda_vs_z.png, mbh_vs_lambda.png, delta_line_vs_continuum.png, cluster_lambda_hist.png")

if len(feat_df) >= 10 and "cluster" in feat_df.columns:
    plt.figure(figsize=(8,6))
    for lab in feat_df["cluster"].unique():
        sel = feat_df["cluster"] == lab
        plt.hist(feat_df.loc[sel, "logL5100"], bins=15,
                 alpha=0.6, label=f'Cluster {lab} (n={sel.sum()})',
                 histtype='stepfilled')
    plt.xlabel("Continuum luminosity (log L5100)")
    plt.ylabel("Count")
    plt.title("Continuum luminosity distribution by cluster")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig("cluster_continuum_hist.png", dpi=150)
    plt.show()

import os
import datetime
import pandas as pd

def get_file_info(path):
    """
    读取文件信息：名称、大小（MB）、修改时间
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"❌ 文件不存在：{os.path.abspath(path)}")

    size_mb = os.path.getsize(path) / (1024 * 1024)
    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(path))

    return {
        "文件名": os.path.basename(path),
        "文件大小(MB)": round(size_mb, 3),
        "修改时间": mtime.strftime("%Y-%m-%d %H:%M:%S")
    }
"""
SDSS版本：复刻目标图样式 → 横纵坐标为logL5100 vs logLHb
"""
import warnings
warnings.filterwarnings("ignore")
from astropy.table import Table
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# -----------------------
# 读取SDSS数据（替换为你的SDSS文件路径）
# -----------------------
data_path = "../sdss/sdss-metadata-hdu2.mrt"  # 替换为你的SDSS数据路径
t = Table.read(data_path, format="ascii.cds")
df = t.to_pandas()

# 提取核心列：logL5100和logLHb
df["logL5100"] = pd.to_numeric(df["logL5100"], errors='coerce')
df["logLHb"] = pd.to_numeric(df["logLHb"], errors='coerce')
df = df.dropna(subset=["logL5100", "logLHb"])

# 标准化坐标（与目标图一致：std units）
x = (df["logL5100"] - df["logL5100"].mean()) / df["logL5100"].std()
y = (df["logLHb"] - df["logLHb"].mean()) / df["logLHb"].std()

# -----------------------
# 计算拟合（匹配目标图标注格式）
# -----------------------
slope, intercept, r_val, _, _ = linregress(x, y)
fit_label = f"Fit: y={slope:.2f}x+{intercept:.2f} (r={r_val:.2f})"

# -----------------------
# 1:1复刻目标图样式
# -----------------------
plt.figure(figsize=(8, 6))

# 散点：SDSS用深蓝色（区分DESI的绿色）
plt.scatter(x, y, color="#1976D2", s=20, alpha=0.7)

# y=x参考线：黑色虚线（和目标图一致）
plt.plot([-3, 2], [-3, 2], "k--", linewidth=1.5, label="Perfect sync (y=x)")

# 拟合线：红色实线（和目标图一致）
plt.plot(x, slope*x + intercept, "r-", linewidth=2, label=fit_label)

# ±2σ波动区：灰色半透明（和目标图一致）
residuals = y - (slope*x + intercept)
std_res = np.std(residuals)
plt.fill_between(
    [-3, 2],
    (slope*np.array([-3, 2]) + intercept) - 2*std_res,
    (slope*np.array([-3, 2]) + intercept) + 2*std_res,
    color="gray", alpha=0.2, label="±2σ fluctuation"
)

# 坐标/标题/图例：完全匹配目标图格式
plt.xlabel("Normalized logL5100 (std units)", fontsize=12)
plt.ylabel("Normalized logLHb (std units)", fontsize=12)
plt.title("SDSS - logL5100 vs logLHb (Enhanced Correlation)", fontsize=14)
plt.xlim(-3, 2)  # 和目标图X轴范围一致
plt.ylim(-6, 2)  # 和目标图Y轴范围一致
plt.grid(linestyle="--", alpha=0.3)
plt.legend(loc="upper left", fontsize=10)

plt.tight_layout()
plt.savefig("SDSS_logL5100_vs_logLHb_matched.png", dpi=150)
plt.show()

# 获取两个文件的信息
desi_info = get_file_info(desi_path)
sdss_info = get_file_info(sdss_path)

# 生成表格
df = pd.DataFrame([desi_info, sdss_info])

print(df)

# 输出到 Excel
output_path = "/Users/mac/Desktop/PythonProject/DESI_SDSS_数据集信息表.xlsx"
df.to_excel(output_path, index=False)
print("✅ 已生成文件：", output_path)



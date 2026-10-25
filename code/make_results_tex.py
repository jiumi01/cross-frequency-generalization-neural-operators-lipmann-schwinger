# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Generate paper tables and inline macros directly from the evaluation JSONs.

Nothing is transcribed by hand: every number in the manuscripts comes from
results/eval_*.json via this script, so the PDFs cannot drift from the data.
Emits both the English and the Chinese variant.
"""
import os
import re
import sys
import glob
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

# Lazily filled by load_spectrum(); "unset" distinguishes "not read yet" from
# "read and absent", which is what makes the `--` fallbacks honest.
_SPECTRUM = "unset"

# The five arms of Table~\ref{tab:main}; their seed count is what that table's
# caption must quote (see seed_macros).
MAIN_ARMS = ("main_fno_a005", "main_fno_a010", "main_fno_a025",
             "main_fno_a050", "main_fno_a100")

CAP = { "en": dict(
  main=("Cross-frequency generalization of the FNO: mean relative $L_2$ error, "
        "over \\NumSeedsMain\\ seeds and 200 paired media per frequency. Training "
        "uses $f\\in[1,4]$; the header splits the columns at that edge. Rows "
        "marked $\\psi_{\\mathrm{inc}}$ are the trivial baseline "
        "$u_\\theta\\equiv0$."),
  summary=("Generalization cliff per contrast arm: mean relative $L_2$ error over "
           "\\NumSeedsMain\\ seeds, with the seed standard deviation beside each "
           "mean. $\\varepsilon_{\\mathrm{in}}$, "
           "$\\varepsilon_{\\infty}$: in-band and plateau mean relative $L_2$; "
           "$r$: $\\varepsilon(5)/\\varepsilon(4)$."),
  propnorm=("Discrete operator norm of the Lippmann--Schwinger propagator, "
            "$\\max_{\\mathbf{n}}|\\kappa^{2}\\widehat{G}_{\\kappa}(\\mathbf{n})|$, "
            "on the $64\\times64$ grid. The continuum is flat in $f$ at "
            "$\\PropNominal$; grid norm ranges from $\\PropLo$ at $f=\\PropLoF$ "
            "to $\\PropHighest$; the forward map is not uniformly conditioned, "
            "the grid maximum is at $f=4.25$."),
  arch=("Architecture comparison at matched budget: families swapped into the "
        "$v_{\\max}=0.25$ and $1.00$ arms, FNO rows of other contrasts kept for "
        "one scale. $\\varepsilon_{\\mathrm{in}}$: in-band mean over $[1,4]$; "
        "$\\varepsilon_{f}$: mean relative $L_{2}$ at test frequency $f$; "
        "params (M): million parameters."),
  band=("Training-band-width ablation at $v_{\\max}=0.25$, read across two "
        "families. The FNO steps at the band edge and the U-Net does not, but in "
        "both the gain sits just beyond the new edge and the in-band floor is "
        "unmoved by it."),
  mode=("Retained-mode budget $M$ at $v_{\\max}=0.25$, with training band "
        "$[1,4]$; $M=20$ is the reference arm. $f^{\\star}$ is the first test "
        "frequency whose mean error exceeds $6\\varepsilon(4)$."),
  born=("Truncated Born (Neumann) series on the same paired media. Entries above "
        "unity mean the series is worse than predicting zero."),
  realgeo=("Third level: the Marmousi windows against the synthetic control. "
           "Columns are $\\varepsilon_{\\mathrm{in}}$, $\\varepsilon_{\\infty}$ "
           "and $r$ of Table~\\ref{tab:summary}; both groups are means over the "
           "same \\RealWinN{} windows rather than the \\MainNMedia{} paired "
           "media of Table~\\ref{tab:main}, and differ only in geometry."),
  paired=("Paired tests on the per-sample errors. $n$: entries averaged; "
          "$\\Delta=\\bar e_{A}-\\bar e_{B}$: mean paired difference, negative "
          "when $A$ is the more accurate, 95\\% interval resampling media; $p$: "
          "sign-flip; $d_z$: standardised paired effect size, $\\bar\\Delta/"
          "\\mathrm{sd}(\\Delta)$; $*$: interval excluding zero."),
  si_paired=("Paired tests on the per-sample errors. $n$: entries averaged; "
             "$\\Delta=\\bar e_{A}-\\bar e_{B}$: mean paired difference, "
             "negative when $A$ is the more accurate, 95\\% interval "
             "resampling media; $p$: sign-flip; $d_z$: standardised paired "
             "effect size, $\\bar\\Delta/\\mathrm{sd}(\\Delta)$; $*$: interval "
             "excluding zero."),
  arms=("Complete inventory of the trained arms behind the seeding protocol. "
        "$\\varepsilon(f)$: seed-mean relative $L_2$; the last column averages "
        "$f\\ge6$. For the widened-band arms these lie inside their own "
        "training band; for the $[1,2]$ arm, far outside."),
  c1=("Discriminating experiment for the conditioning confound: arms share "
      "medium and protocol, differing only in regularisation. The baseline "
      "fixes $\\gamma=0.05$, whose attained norm spans $\\PropSpread\\times$ "
      "over the test grid; the compensated arm holds it constant. 3-seed "
      "means."),
  disc=("Discriminator arms: all rows share the baseline's media, protocol "
        "and seeds; each changes one ingredient only (input encoding of $f$, "
        "loss, or initialisation). Onset is the interval in which the error "
        "first leaves $3\\times$ its own in-band level. 3-seed means."),
  anchor=("Analytic anchor on every other arm's protocol: paired media, "
          "$n=200$, seeds shared with the trained rows. $a,b$ are fitted on a "
          "prefix of the training cache only -- no test medium, no "
          "out-of-band frequency. The learned residual is identically zero."),
  deep=("Analytic head depth on the common protocol: paired media, $n=200$, "
        "shared seeds. Each row is fitted on one in-band prefix; $M$ counts "
        "Born corrections, $M_{\\mathrm{sel}}$ is picked in band. The last "
        "column is the trained plateau over the head's."),
  gridthree=("(a) Reference convergence: one medium drawn at $256^2$ and "
             "downsampled to the two coarser grids. (b) The trained FNO, on its "
             "native $64^2$ input, scored against three targets."),
  ci=("Headline scalars with $95\\%$ intervals from \\CIBoot\\ bootstrap "
      "replicates resampling media; onset: first node reaching half the "
      "plateau."),
  locators=("All three boundary locators, recomputed on every rung of the four "
            "control ladders. \\emph{n/a}: the locator declines to answer, as "
            "on the widest band, where every test node lies inside the trained "
            "band."),
  res=("Resolution check: three locators and two scalars at $64^{2}$ and "
       "$128^{2}$. (a) native protocol; (b) one media draw discretised on both "
       "grids; (c) Marmousi windows; (d) their control. Rows of (c), (d) give "
       "$f^{\\star}$, jump and errors only."),
  res_note=("$\\varepsilon_{\\infty}$ averages $f>5$ in (a), (b), as in "
            "Table~\\ref{tab:main}, and $f\\ge5$ in (c), (d), as in "
            "Supplementary Table~15."),
  banddown=("Below-band FNO behaviour per contrast arm: mean relative "
            "$L_2$ error over \\NumSeeds\\ seeds at the left band edge and "
            "below, against the in-band floor (Table~\\ref{tab:summary}) and "
            "the trivial baseline at $f=0.5$; the last column is the ratio."),
  data=("Training-set-size ladder at the reference budget ($M=20$, "
        "$v_{\\max}=0.25$, $f\\in[1,4]$), \\NDataRunsPerRung\\ seeds per rung. "
        "The last column is that rung's largest single step, with the interval "
        "it spans; the ladder stops because this arm's generated cache is "
        "exhausted there."),
  data_lab=dict(h=("$n_{\\mathrm{train}}$", "$\\varepsilon_{\\mathrm{in}}$",
                   "$\\varepsilon_{4}$", "$\\varepsilon_{4.25}$",
                   "$\\varepsilon_{\\infty}$", "largest step $f$")),
  paired_lab=dict(
    unet_base_vs_fno_a025="U-Net (base) $-$ FNO, $v_{\\max}=0.25$, plateau",
    unet_pmatch_vs_fno_a025="U-Net (matched) $-$ FNO, $v_{\\max}=0.25$, plateau",
    unet_base_vs_fno_a100="U-Net (base) $-$ FNO, $v_{\\max}=1.00$, plateau",
    unet_pmatch_vs_fno_a100="U-Net (matched) $-$ FNO, $v_{\\max}=1.00$, plateau",
    capw64_vs_fno_a025="FNO $w=64$ $-$ FNO $w=40$, plateau",
    capw64_vs_fno_a025_in="FNO $w=64$ $-$ FNO $w=40$, in band",
    mode_m02_vs_m20_in="$M=2$ $-$ $M=20$, in band",
    jump_fno_a025="FNO $f=5$ $-$ $f=4$, $v_{\\max}=0.25$",
    jump_fno_a100="FNO $f=5$ $-$ $f=4$, $v_{\\max}=1.00$",
    jump_unet_base_a025="U-Net (base) $f=5$ $-$ $f=4$",
    jump_unet_pmatch_a025="U-Net (matched) $f=5$ $-$ $f=4$",
    jump_don_a025="DeepONet $f=5$ $-$ $f=4$",
  ),
  realgeo_h=("real geology", "synthetic"),
  h_lab=("train band", "model", "order", "params (M)"),
  cost=("Per-sample cost on the $64\\times64$ problem, one GPU, batch 32. The "
        "solver row carries its own timings. Speed-up is quoted at both ends of "
        "the test range, against an unpreconditioned Bi-CGSTAB solve, and is "
        "therefore an upper bound rather than a constant."),
  cost_lab=dict(
    solver="Bi-CGSTAB (reference)",
    fno="$\\mathrm{FNO}$ $w=40$, $M=20$",
    unet_base="U-Net (base)",
    unet_match="U-Net (matches FNO)",
    don="$\\mathrm{DeepONet}$ ($w=32$)",
    h=("model", "params (M)", "train (s/ep.)", "infer. (ms)")),
 ),
 "cn": dict(
  main=("FNO 的跨频率泛化：\\NumSeedsMain\\ 个种子、每个频率 200 个配对介质的"
        "平均相对 $L_2$ 误差。训练仅使用 $f\\in[1,4]$；表头在该边处拆分各列。"
        "标为 $\\psi_{\\mathrm{inc}}$ 的行是平凡基线 $u_\\theta\\equiv0$。"),
  summary=("各对比度臂的泛化悬崖汇总：为 \\NumSeedsMain\\ 个随机种子的均值，"
           "紧邻的一列为该窗口上的种子标准差。$\\varepsilon_{\\mathrm{in}}$：训练频带内、"
           "留出介质上的平均相对 $L_2$ 误差；$\\varepsilon_{\\infty}$：$f\\ge6$ 的"
           "平均值；$r$：$\\varepsilon(5)/\\varepsilon(4)$。"),
  propnorm=("Lippmann--Schwinger 传播子在 $64\\times64$ 网格上的离散算子范数 "
            "$\\max_{\\mathbf{n}}|\\kappa^{2}\\widehat{G}_{\\kappa}(\\mathbf{n})|$。"
            "连续符号对 $f$ 为 $\\PropNominal$；网格上取到的范数介于 "
            "$\\PropLo$（$f=\\PropLoF$）与 $\\PropHighest$ 之间，"
            "故前向算子测试网格的条件数并不一致；网格最大值出现在 $f=4.25$。"),
  arch=("相同训练预算下的架构对比：三族模型替换进 $v_{\\max}=0.25$ 与 $1.00$ 两臂。"
        "$\\varepsilon_{\\mathrm{in}}$ 为 $[1,4]$ 带内均值；$\\varepsilon_{f}$ 为测试频率 "
        "$f$ 处平均相对 $L_2$ 误差；params (M) 为百万计参数量。"),
  band=("$v_{\\max}=0.25$ 的训练带宽消融，跨两个架构族读取。FNO 在带缘呈台阶，"
        "而 U-Net 无台阶；但两者的收益都集中在新的带缘之外，"
        "带内下界也都不随带宽移动。"),
  mode=("$v_{\\max}=0.25$ 时保留模态预算 $M$ 的影响（训练频带 $[1,4]$，$M=20$ 为"
        "基准）。模态预算构成第二条独立的泛化边界；$f^{\\star}$ 为平均误差首次"
        "超过 $6\\varepsilon(4)$ 的测试频率。"),
  born=("同一批配对介质上的截断 Born（Neumann）级数。该级数在弱对比度区间是强基线，"
        "当耦合量 $v_{\\max}/(2\\gamma)$ 超过 1 后发散；大于 1 的数值意味着它比"
        "直接预测零更差。"),
  realgeo=("第三级：Marmousi 窗口与合成对照。列为表 \\ref{tab:summary} 的 "
           "$\\varepsilon_{\\mathrm{in}}$、$\\varepsilon_{\\infty}$ 与 $r$；"
           "两组均为同一批 \\RealWinN{} 个窗口上的均值，而非表 "
           "\\ref{tab:main} 的 \\MainNMedia{} 个配对介质，且两组只差几何。"),
  paired=("配对检验。$A$、$B$ 为首列两配置；$n$ 为条目数（介质 $\\times$ 种子 "
          "$\\times$ 频点）；$\\Delta$ 为平均配对差值，95\\% 区间对\emph{介质}重抽样；"
          "$p$ 为双侧符号翻转置换检验（保留一位有效数字）；星号表示区间不含 0。"),
  arms=("支撑种子协议的全部训练臂清单。$\\varepsilon(f)$：测试频率 $f$ 处的"
        "种子均值相对 $L_2$；末列对 $f\\ge6$ 取平均。对加宽带臂，这些频率位于"
        "其自身训练带内——那正是这些臂的目的；对 $[1,2]$ 臂则远在带外。"),
  c1=("条件数混杂的判别实验：两臂介质与协议相同、仅正则化不同。"
      "基线固定 $\\gamma=0.05$，取到范数在测试网格上跨 $\\PropSpread$ 倍；"
      "补偿臂令其恒定。3 种子均值。"),
  disc=("判别实验各臂：所有臂与基线共享介质、协议与种子，每次只改动一个环节"
        "（$f$ 的输入编码、损失或初始化）。onset 指误差首次超过自身带内水平 "
        "3 倍所在的网格区间。3 种子均值。"),
  anchor=("解析锚定，按与其余各臂完全相同的协议评分（$n=200$，介质与种子配对）。"
          "系数 $a,b$ 仅由训练缓存前缀拟合：不含测试介质，不含带外频率。"
          "学习残差恒为零。"),
  deep=("解析头深度：$n=200$，配对介质与种子。每行由同一带内前缀拟合；"
        "$M$ 为 Born 项数。末列为训练算子平台与本头平台之比。"),
  gridthree=("(a) 参考解收敛性：同一介质在 $256^2$ 上生成并降采样。(b) FNO "
             "固定 $64^2$ 输入，对三个目标分辨率评分。"),
  ci=("主标量及其 $95\\%$ 区间（\\CIBoot\\ 次按介质重采样）；"
      "onset 为曲线首次达到平台一半的频点。"),
  locators=("三个边界定位器在四条控制实验梯子每一级上的取值。"
            "\\emph{n/a} 表示该定位器拒绝作答；最宽频带上它无定义，"
            "因为全部测试节点都在训练频带内。"),
  res=("分辨率检查：三个定位器与两个标量在 $64^{2}$/$128^{2}$ 上重算。"
       "(a) 原生协议；(b) 同一批介质离散于两种网格；(c) Marmousi 窗口；"
       "(d) 合成对照。(c)(d) 行只给 $f^{\\star}$、跃变与误差。"),
  # The CN manuscript carries the level-3 table in its own body (\input{gen_tab_realgeo}),
  # so it must be cited by \ref there; only the EN version leaves it to the
  # supplement, where the pointer is by number because the label would dangle.
  res_note=("(a)(b) 的 $\\varepsilon_{\\infty}$ 取 $f>5$ 均值（与表~\\ref{tab:main} 一致），"
            "(c)(d) 取 $f\\ge5$ 均值（与表~\\ref{tab:realgeo} 一致）。"),
  banddown=("FNO 各臂的带下行为：\\NumSeeds\\ 个种子在左带缘 $f=1.0$ 及其"
            "下方的平均相对 $L_2$ 误差，对照带内下限（表 \\ref{tab:summary}）"
            "与 $f=0.5$ 处平凡基线；末列为两个 $f=0.5$ 条目之比。"),
  data=("训练样本量阶梯（基准预算 $M=20$、$v_{\\max}=0.25$、$f\\in[1,4]$），"
        "每档 \\NDataRunsPerRung\\ 个种子。末列为该档的最大单步及其跨越的频点区间；"
        "阶梯止于该臂生成缓存的全部样本。"),
  data_lab=dict(h=("$n_{\\mathrm{train}}$", "$\\varepsilon_{\\mathrm{in}}$",
                   "$\\varepsilon_{4}$", "$\\varepsilon_{4.25}$",
                   "$\\varepsilon_{\\infty}$", "最大单步 $f$")),
  paired_lab=dict(
    unet_base_vs_fno_a025="U-Net（基础）$-$ FNO，$v_{\\max}=0.25$，平台",
    unet_pmatch_vs_fno_a025="U-Net（匹配）$-$ FNO，$v_{\\max}=0.25$，平台",
    unet_base_vs_fno_a100="U-Net（基础）$-$ FNO，$v_{\\max}=1.00$，平台",
    unet_pmatch_vs_fno_a100="U-Net（匹配）$-$ FNO，$v_{\\max}=1.00$，平台",
    capw64_vs_fno_a025="FNO $w=64$ $-$ FNO $w=40$，平台",
    capw64_vs_fno_a025_in="FNO $w=64$ $-$ FNO $w=40$，带内",
    mode_m02_vs_m20_in="$M=2$ $-$ $M=20$，带内",
    jump_fno_a025="FNO $f=5$ $-$ $f=4$，$v_{\\max}=0.25$",
    jump_fno_a100="FNO $f=5$ $-$ $f=4$，$v_{\\max}=1.00$",
    jump_unet_base_a025="U-Net（基础）$f=5$ $-$ $f=4$",
    jump_unet_pmatch_a025="U-Net（匹配）$f=5$ $-$ $f=4$",
    jump_don_a025="DeepONet $f=5$ $-$ $f=4$",
  ),
  realgeo_h=("真实地质", "合成对照"),
  h_lab=("训练频带", "模型", "阶数", "参数量 (M)"),
  cost=("$64\\times64$ 问题上的单样本开销（同一 GPU，批大小 32）。"
        "求解器行给出其自身的耗时，训练列为每轮秒数；加速比在测试频段两端"
        "分别给出，且相对无预条件的 Bi-CGSTAB 求解，故为优势上界。"),
  cost_lab=dict(
    solver="Bi-CGSTAB（参考）",
    fno="$\\mathrm{FNO}$ $w=40$，$M=20$",
    unet_base="U-Net（基础）",
    unet_match="U-Net（匹配 FNO）",
    don="$\\mathrm{DeepONet}$（$w=32$）",
    h=("模型", "参数量 (M)", "训练 (s/epoch)", "推理 (ms)")),
 ),
}


def arm_key(v):
    return "a%03d" % int(round(v * 100))


def freq_label(f):
    """Frequency as it appears in a table header.

    ``"%.1f"`` was used here originally and it rounded the two densified
    band-edge points: 4.25 printed as "4.2" and 4.75 as "4.8", so the table
    disagreed with the text and with \\TestFreqList about what the test grid
    actually is -- while the whole argument for a *step* rests on that 0.25
    spacing.  Keep one decimal for whole frequencies (4.0) and two otherwise.
    """
    if abs(f - round(f)) < 1e-9:
        return "%.1f" % f
    s = ("%.2f" % f).rstrip("0")
    return s + "0" if s.endswith(".") else s


def load(res_dir=None):
    out = {}
    for p in sorted(glob.glob(os.path.join(res_dir or C.RES, "eval_*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        out[d["name"]] = d
    return out


def runs_for(ev, prefix):
    """Eval records matching `prefix` that were produced on the CURRENT grid.

    An eval file outlives the protocol it was produced under.  If F_TEST gains
    or loses a node, every stale record has one column fewer (or more) than the
    table it is about to be written into, and the resulting LaTeX either fails
    to compile or -- worse -- silently associates errors with the wrong
    frequencies.  Filter on the recorded grid instead of trusting the filename.
    """
    cur = []
    for n in sorted(ev):
        if not n.startswith(prefix):
            continue
        if list(ev[n].get("f_test") or []) != list(C.F_TEST):
            continue
        cur.append(n)
    return cur


def mean_over_seeds(ev, names, vmax):
    cur = []
    for n in names:
        rows = ev[n]["sweep"]["%.2f" % vmax]
        cur.append([r["mean"] for r in rows])
    F = [r["f"] for r in ev[names[0]]["sweep"]["%.2f" % vmax]]
    A = np.array(cur)
    return (np.array(F), A.mean(0),
            (A.std(0, ddof=0) if A.shape[0] > 1 else np.zeros(A.shape[1])))


def inband(ev, names, vmax):
    vals = []
    for n in names:
        for r in ev[n]["in_band"]:
            if abs(r["vmax"] - vmax) < 1e-6:
                vals.append(r["mean"])
    return float(np.mean(vals)), (float(np.std(vals)) if len(vals) > 1 else 0.0)


def fit_alpha(F, E, f0):
    m = F > f0
    if m.sum() < 3:
        return np.nan, np.nan
    x, y = np.log(F[m] / f0), np.log(E[m])
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    r2 = 1 - ((y - yhat) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-30)
    return float(coef[0]), float(r2)


def fmt(x, n=3):
    # NB: the fallback is deliberately bare `--`, not `{--}`.  Wrapping it in
    # braces made the emitted macro read \newcommand{\X}{{--}}, which both
    # prints a stray en-dash in the PDF and slipped past the audit's
    # `}{--}` pattern -- i.e. an unresolved number could reach the manuscript
    # silently.  Keep every fallback in exactly one form.
    return ("%%.%df" % n) % x if np.isfinite(x) else "--"


def fmt_p(p):
    # one significant figure, typographically correct scientific notation.
    # %.0e yields the bare "1e-04"; the table renders it as $1\times10^{-4}$.
    m, e = ("%.0e" % p).split("e")
    return r"%s\times10^{%d}" % (m, int(e))


def sd_bound(ev, prefixes, vmax_list, use_inband=False):
    """Largest per-cell seed standard deviation over the named runs.

    A 15-column table cannot carry a second number in every cell, so instead of
    duplicating the table the caption states the measured worst case.  The value
    is the maximum over exactly the cells the table prints (plus the in-band
    column when the table has one), so it is checkable rather than asserted, and
    it bounds what a reader can infer about a single entry's stability.
    """
    mx = 0.0
    for pref in prefixes:
        names = runs_for(ev, pref)
        if not names:
            continue
        for vmax in vmax_list:
            _, _, S = mean_over_seeds(ev, names, vmax)
            if np.size(S) and np.any(np.isfinite(S)):
                mx = max(mx, float(np.nanmax(S)))
            if use_inband:
                mx = max(mx, inband(ev, names, vmax)[1])
    return mx


def na_or(x, n=2):
    """A measured quantity as a number, or the explicit token `n/a`.

    Not `--`.  Two reasons, both observed: the delivered file must not carry
    placeholder-shaped tokens even when unreferenced, and `$--$` inside math
    mode prints as two minus signs, which a reader parses as a range or an
    interval rather than as "undefined here".  This happens when a locator is
    defined by a threshold that a given arm never crosses, which is a
    meaningful state and should be named, not blanked.
    """
    return fmt(x, n) if np.isfinite(x) else "n/a"


def load_spectrum():
    """spectrum.json, or None.  Cached: the main table and the macros both use it.

    Single loader on purpose.  The trivial baseline is a property of the data
    distribution rather than of any training run, so if the table and the
    spectral figure read it through different code paths they can disagree
    without either of them being wrong on its own terms.
    """
    global _SPECTRUM
    if _SPECTRUM == "unset":
        try:
            _SPECTRUM = json.load(open(os.path.join(C.RES, "spectrum.json"),
                                       encoding="utf-8"))
        except Exception:
            _SPECTRUM = None
    return _SPECTRUM


def triv_row(vmax):
    """Trivial baseline ||u||_2/||psi||_2 per test frequency for one arm, else None.

    The architectures return psi_inc + u_theta, so u_theta == 0 is a predictor
    they contain; its relative error is exactly this ratio.  Every out-of-band
    plateau in the paper has to be read against it.

    Source order is `triv_paired.json` then `spectrum.json`, and the order is
    not cosmetic.  `spectrum.json` records the baseline as a mean over 600
    media; the model rows it is compared against are means over the 200
    evaluation media.  The two draws share their media and angles (checked
    bitwise in `baseline_paired.prefix_check`) but not their sample size, so
    the 600-media denominator contains the 200 evaluation media plus 400
    others.  Restricting it to the evaluation media moves each cell by 2.7% on
    average and by up to 8.3% (results/triv_paired.json, `rel_diff_vs_600`) --
    the same size as the plateau shift the refinement study reports -- so the
    numerator and the denominator have to come from the same sample or the
    ratio inherits that difference as if it were a finding.  `triv_paired` is
    the paired one and is preferred whenever it exists.
    """
    p = os.path.join(C.RES, "triv_paired.json")
    if os.path.exists(p):
        d = json.load(open(p, encoding="utf-8"))
        key = None
        for k in d["data"]:
            if abs(float(k) - float(vmax)) < 1e-9:
                key = k
        if key is None:
            return None
        arm = d["data"][key]
        return np.array([np.mean(arm["f=%g" % f]) for f in C.F_TEST], float)
    s = load_spectrum()
    if not s:
        return None
    arm = s["data"].get(str(vmax))
    if not arm:
        return None
    out = []
    for f in C.F_TEST:
        r = arm.get("f=%g" % f)
        if r is None or "triv" not in r:
            return None
        out.append(r["triv"])
    return np.array(out, float)


def table_main(ev, arms, cap):
    F, body = None, []
    for vmax in arms:
        names = runs_for(ev, "main_fno_%s_" % arm_key(vmax))
        if not names:
            body.append("%.2f & " % vmax + " & ".join(["{--}"] * len(C.F_TEST)) + " \\\\")
            continue
        Fv, M, S = mean_over_seeds(ev, names, vmax)
        F = Fv
        body.append("%.2f & " % vmax + " & ".join(fmt(m) for m in M) + " \\\\")
        tb = triv_row(vmax)
        if tb is not None:
            # Interleaved under its own arm rather than pooled at the bottom:
            # the comparison that matters is a plateau against the baseline of
            # the SAME arm, and pooling would invite reading across rows.
            body.append("\\quad$\\psi_{\\mathrm{inc}}$ & "
                        + " & ".join(fmt(t) for t in tb) + " \\\\")
    freqs = F if F is not None else C.F_TEST
    hdr = " & ".join(freq_label(f) for f in freqs)
    # The in-band / out-of-band split is the fact this table exists to show,
    # so it is carried by the table's STRUCTURE -- a grouped two-tier header
    # and a rule at the band edge -- rather than by a sentence in the caption
    # the reader has to hold in mind across sixteen columns.
    n_in = sum(1 for f in freqs if f <= C.F_TRAIN[1] + 1e-9)
    n_out = len(freqs) - n_in
    head = ("\\begin{table*}[t]\n\\centering\n\\caption{%s}\n\\label{tab:main}\n"
            "\\footnotesize\n\\setlength{\\tabcolsep}{3.0pt}\n"
            "\\begin{tabular}{l" + "c" * n_in + "!{\\vline}" + "c" * n_out
            + "}\n\\toprule\n"
            " & \\multicolumn{%d}{c}{in band} & \\multicolumn{%d}{c}{out of "
            "band} \\\\\n" % (n_in, n_out)
            + "\\cmidrule(lr){2-%d}\\cmidrule(lr){%d-%d}\n"
            % (1 + n_in, 2 + n_in, 1 + len(freqs))
            + "$v_{\\max}$ & " + hdr + " \\\\\n\\midrule\n" + "\n".join(body)
            + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    return head % cap["main"]


def table_summary(ev, arms, cap):
    rows = []
    for vmax in arms:
        names = runs_for(ev, "main_fno_%s_" % arm_key(vmax))
        if not names:
            continue
        F, M, S = mean_over_seeds(ev, names, vmax)
        ib, ibs = inband(ev, names, vmax)
        plateau = float(np.mean(M[F >= 6]))
        m4, m5 = np.isclose(F, 4.0), np.isclose(F, 5.0)
        cliff = float(M[m5][0] / M[m4][0]) if m4.any() and m5.any() else np.nan
        a, r2 = fit_alpha(F, M, 4.0)
        # Seed spread of the plateau, from the same three seeds as the mean.
        psd = float(np.mean(S[F >= 6])) if (F >= 6).any() else np.nan
        rows.append((vmax, ib, ibs, plateau, cliff, a, r2, psd))
    t = ("\\begin{table}[t]\n\\centering\n\\caption{%s}\n\\label{tab:summary}\n"
         "\\small\n\\setlength{\\tabcolsep}{4.6pt}\n"
         "\\begin{tabular}{lccccc}\n\\toprule\n"
         " & \\multicolumn{2}{c}{$\\varepsilon_{\\mathrm{in}}$} & "
         "\\multicolumn{2}{c}{$\\varepsilon_{\\infty}$} & \\\\\n"
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
         "$v_{\\max}$ & mean & $\\mathrm{SD}$ & mean & $\\mathrm{SD}$ & $r$ "
         "\\\\\n\\midrule\n"
         % cap["summary"])
    # Two separate SD columns would be indistinguishable if both were headed
    # `SD`; the grouped header names the quantity each SD belongs to.  The
    # caption used to say "in parentheses", which this table has never done.
    # The fitted exponent is NOT printed.  It is reported in the prose only as a
    # range over fit windows (Section on the step), where it is used to argue
    # that no single exponent is identifiable; printing one point estimate here
    # would contradict that argument in the same manuscript.
    for (vmax, ib, ibs, plateau, cliff, a, r2, psd) in rows:
        t += ("%.2f & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ \\\\\n"
              % (vmax, fmt(ib), fmt(ibs), fmt(plateau), fmt(psd), fmt(cliff, 2)))
    return t + "\\bottomrule\n\\end{tabular}\n\\end{table}\n", rows


def table_realgeo(arms, cap):
    """Third rung: real geological structure against the synthetic control."""
    p = os.path.join(C.RES, "realgeo.json")
    if not os.path.exists(p):
        return ""
    d = json.load(open(p, encoding="utf-8"))
    s = d.get("summary")
    if not s:
        return ""
    t = ("\\begin{table}[t]\n\\centering\n\\caption{%s}\n\\label{tab:realgeo}\n"
         "\\small\n\\setlength{\\tabcolsep}{4.2pt}\n"
         "\\begin{tabular}{lcccccc}\n\\toprule\n"
         " & \\multicolumn{3}{c}{%s} & \\multicolumn{3}{c}{%s} \\\\\n"
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
         "$v_{\\max}$ & $\\varepsilon_{\\mathrm{in}}$ & $\\varepsilon_{\\infty}$ & $r$"
         " & $\\varepsilon_{\\mathrm{in}}$ & $\\varepsilon_{\\infty}$ & $r$ \\\\\n\\midrule\n"
         % (cap["realgeo"], cap["realgeo_h"][0], cap["realgeo_h"][1]))
    for vmax in arms:
        k = "%.2f" % vmax
        if k not in s["family"]["real"]:
            continue
        r = s["family"]["real"][k]
        y = s["family"]["synthetic"][k]
        t += ("%.2f & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ \\\\\n"
              % (vmax, fmt(r["in_band"]), fmt(r["plateau"]), fmt(r["r5"], 2),
                 fmt(y["in_band"]), fmt(y["plateau"]), fmt(y["r5"], 2)))
    return t + "\\bottomrule\n\\end{tabular}\n\\end{table}\n"


def table_arch(ev, arms, cap):
    # Labels come from the single definition in common.ARCH_GROUPS so that the
    # table rows and the Fig. 4 legend cannot drift apart.
    #
    # eps(4) and eps(5) are printed because they are what decides whether a
    # family steps or decays.  Giving only eps_in / eps_6 / eps_12 hid the
    # band-edge behaviour exactly where the text claimed all three families
    # behaved alike, so a reader had to take the claim on trust.
    groups = [(p + "_", lab) for p, lab in C.ARCH_GROUPS]
    t = ("\\begin{table}[t]\n\\centering\n\\caption{%s}\n\\label{tab:arch}\n"
         "\\small\n\\setlength{\\tabcolsep}{3.2pt}\n"
         "\\begin{tabular}{clcccccc}\n\\toprule\n"
         "$v_{\\max}$ & %s & %s & $\\varepsilon_{\\mathrm{in}}$ & "
         "$\\varepsilon_{4}$ & $\\varepsilon_{5}$ & "
         "$\\varepsilon_{6}$ & $\\varepsilon_{12}$ \\\\\n\\midrule\n"
         % (cap["arch"], cap["h_lab"][1], cap["h_lab"][3]))
    first = True
    for vmax in arms:
        rows = []
        for pref, lab in groups:
            names = runs_for(ev, pref % arm_key(vmax))
            if not names:
                continue
            F, M, S = mean_over_seeds(ev, names, vmax)
            ib, _ = inband(ev, names, vmax)
            at = lambda fv: (M[np.isclose(F, fv)][0]
                             if np.isclose(F, fv).any() else np.nan)
            rows.append("$%.2f$ & %s & %.2f & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ \\\\\n"
                        % (vmax, lab, ev[names[0]]["params"] / 1e6, fmt(ib),
                           fmt(at(4.0)), fmt(at(5.0)), fmt(at(6.0)), fmt(at(12.0))))
        if not rows:
            continue
        if not first:
            t += "\\midrule\n"
        first = False
        t += "".join(rows)
    return t + "\\bottomrule\n\\end{tabular}\n\\end{table}\n"


def table_band(ev, arms, cap):
    # (run prefix, family label, band label).  The two U-Net rows are the
    # control the band-width claim needed: widening the band was verified on the
    # FNO alone while the architecture table showed the U-Net has no step at
    # all, so a referee could read "only the band moves the cliff" as a claim
    # about one family.  Both families are in the same table now.
    rows = [("main_fno_a025_", "FNO", "[1,4]"), ("band_b2_", "FNO", "[1,2]"),
            ("band_b3_", "FNO", "[1,3]"), ("band_b6_", "FNO", "[1,6]"),
            ("band_b12_", "FNO", "[1,12]"),
            ("base_unet_a025_", "U-Net", "[1,4]"),
            ("uband_b6_", "U-Net", "[1,6]")]
    t = ("\\begin{table}[t]\n\\centering\n\\caption{%s}\n\\label{tab:band}\n"
         "\\small\n\\setlength{\\tabcolsep}{3.6pt}\n"
         "\\begin{tabular}{llccccc}\n\\toprule\n"
         "%s & %s & $\\varepsilon_{\\mathrm{in}}$ & $\\varepsilon_{5}$ & "
         "$\\varepsilon_{6}$ & $\\varepsilon_{8}$ & $\\varepsilon_{12}$ \\\\\n"
         "\\midrule\n" % (cap["band"], cap["h_lab"][1], cap["h_lab"][0]))
    for pref, fam, lab in rows:
        names = runs_for(ev, pref)
        if not names:
            continue
        F, M, S = mean_over_seeds(ev, names, 0.25)
        ib, _ = inband(ev, names, 0.25)
        at = lambda fv: (M[np.isclose(F, fv)][0] if np.isclose(F, fv).any()
                         else np.nan)
        t += ("%s & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ \\\\\n"
              % (fam, lab, fmt(ib), fmt(at(5.0)), fmt(at(6.0)), fmt(at(8.0)),
                 fmt(at(12.0))))
    return t + "\\bottomrule\n\\end{tabular}\n\\end{table}\n"


def f_star(F, M_, e4=None, thr=6.0):
    """First test frequency whose mean error exceeds thr x eps(4).

    The location of the boundary is only meaningful with its rule attached, so
    the rule lives here once and is reused by the mode table and by the level-3
    windows; hard-coding the threshold in two places is how the manuscript and
    the tables drift apart.  Returns NaN when no test frequency crosses, rather
    than silently returning the largest one.
    """
    if e4 is None:
        e4 = M_[np.isclose(F, 4.0)][0] if np.isclose(F, 4.0).any() else np.nan
    hits = [f for f, e in zip(F, M_) if np.isfinite(e) and e > thr * e4]
    return float(hits[0]) if hits else float("nan")


def table_mode(ev, cap):
    t = ("\\begin{table}[t]\n\\centering\n\\caption{%s}\n\\label{tab:mode}\n"
         "\\small\n\\setlength{\\tabcolsep}{4pt}\n"
         "\\begin{tabular}{cccccc}\n\\toprule\n"
         "$M$ & $\\varepsilon_{\\mathrm{in}}$ & $\\varepsilon_{5}$ & "
         "$\\varepsilon_{8}$ & $\\varepsilon_{12}$ & $f^{\\star}$ \\\\\n"
         "\\midrule\n" % cap["mode"])
    for m in [1, 2, 3, 4, 8, 12, 20, 24, 32]:
        names = (runs_for(ev, "main_fno_a025_") if m == 20
                 else runs_for(ev, "mode_m%02d_" % m))
        if not names:
            continue
        F, M_, S = mean_over_seeds(ev, names, 0.25)
        ib, _ = inband(ev, names, 0.25)
        at = lambda fv: (M_[np.isclose(F, fv)][0] if np.isclose(F, fv).any()
                         else np.nan)
        t += ("%d & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ \\\\\n"
              % (m, fmt(ib), fmt(at(5.0)), fmt(at(8.0)), fmt(at(12.0)),
                 na_or(f_star(F, M_))))
    return t + "\\bottomrule\n\\end{tabular}\n\\end{table}\n"


# Where the per-epoch training times are frozen.  Same reason as
# CACHE_INVENTORY: runs/ (2.9 GB of checkpoints) is not shipped, and wall clock
# is not reproducible across machines anyway, so the measured medians are carried
# forward rather than recomputed from data that is not there.
TRAIN_TIME_INVENTORY = "train_time.json"


def _train_time_load():
    """Read the frozen per-epoch training times, tolerating a missing file."""
    p = os.path.join(C.RES, TRAIN_TIME_INVENTORY)
    if not os.path.exists(p):
        return {}
    try:
        return json.load(open(p, encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def _median_train_s_per_epoch(prefix, seeds=(0, 1, 2)):
    """Median per-epoch seconds over the seeds of one arm.

    Wall clock is contaminated here and the table says so: every run stops at the
    same 55-epoch budget, so the per-epoch time is comparable work, yet a single
    architecture's value spreads by up to 3x across seeds because the campaigns
    shared the device with the evaluation queue.  A median is reported rather
    than a single seed precisely because no seed is representative.

    The seeds are pinned to the original campaign on purpose.  W2 (2026-10-25)
    re-trained this arm with the device otherwise idle: 5.5 s/epoch against the
    12-17 s/epoch it took under contention.  Averaging the two conditions moved
    the median from 12.4 to 5.6 with no change to the model, and the row would
    then read as though the FNO had become three times cheaper to train.  Since
    the column is quoted for completeness and not for comparison, it is held to
    one measurement condition instead of tracking a mixed one.

    The measurements live in runs/, which the release omits (2.9 GB of
    checkpoints).  Wall clock is not reproducible across machines in the first
    place, so the medians measured while the run directories existed are frozen
    into results/train_time.json and read back where they are absent.  Without
    the record the column would collapse to `n/a` in a fresh clone while the
    PDF still printed the measured numbers, which is exactly the kind of
    silent divergence the rest of this toolchain refuses to allow.
    """
    vals = []
    for s in seeds:
        p = os.path.join(C.RUNS, "%s_s%d" % (prefix, s), "result.json")
        if not os.path.exists(p):
            continue
        try:
            d = json.load(open(p, encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if d.get("epochs_done"):
            vals.append(d["sec"] / d["epochs_done"])
    if vals:
        _t = float(np.median(vals))
        _inv = _train_time_load()
        if _inv.get(prefix) != _t:
            _inv[prefix] = _t
            try:
                os.makedirs(C.RES, exist_ok=True)
                with open(os.path.join(C.RES, TRAIN_TIME_INVENTORY), "w",
                          encoding="utf-8") as fh:
                    json.dump(_inv, fh, ensure_ascii=False, indent=1,
                              sort_keys=True)
            except OSError:
                pass
        return _t
    return float(_train_time_load().get(prefix, float("nan")))


def table_cost(cap):
    """Per-sample inference cost, parameter count and speed-up, four families.

    The parameter count alone cannot settle whether a gap is a capacity effect
    (the architecture table compares a matched U-Net), and the speed-up alone
    cannot either; the two are put side by side so a reader can see that the
    cheapest model is also the least accurate and the smallest.
    """
    p = os.path.join(C.RES, "cost.json")
    if not os.path.exists(p):
        return ""
    d = json.load(open(p, encoding="utf-8"))
    lo = min(d["solver"].values(), key=lambda r: r["per_sample_ms"])
    hi = max(d["solver"].values(), key=lambda r: r["per_sample_ms"])

    def fnum(x, n=2):
        return ("{:,}".format(int(x)).replace(",", "\\,") if n == 0
                else "%.*f" % (n, x))

    lab = cap["cost_lab"]
    rows = [(lab["solver"], None, None, lo["per_sample_ms"], 1.0, 1.0)]
    # (cost.json key, run prefix, label)
    fam = [("FNO w40 M20", "main_fno_a025", lab["fno"]),
           ("U-Net base32", "base_unet_a025", lab["unet_base"]),
           ("U-Net base76", "pmatch_unet_a025", lab["unet_match"]),
           ("DeepONet w32", "base_don_a025", lab["don"])]
    for key, pref, name in fam:
        if key not in d["model"]:
            continue
        m = d["model"][key]
        rows.append((name, m["params"], _median_train_s_per_epoch(pref),
                     m["per_sample_ms"], lo["per_sample_ms"] / m["per_sample_ms"],
                     hi["per_sample_ms"] / m["per_sample_ms"]))
    h = lab["h"]
    t = ("\\begin{table}[t]\n\\centering\n\\caption{%s}\n\\label{tab:cost}\n"
         "\\small\n\\setlength{\\tabcolsep}{3.2pt}\n"
         "\\begin{tabular}{lccccc}\n\\toprule\n"
         "%s & %s & %s & %s & $S_{1.5}$ & $S_{12}$ \\\\\n\\midrule\n"
         % ((cap["cost"],) + tuple(h)))
    for lab, par, trn, inf, s1, s2 in rows:
        # The solver row genuinely has no parameter count and no training phase,
        # which is stated with the token `n/a` rather than `--`: outside math
        # mode `--` prints as an en-dash and reads as a range.
        t += ("%s & %s & %s & %s & $%.1f$ & $%.1f$ \\\\\n"
              % (lab,
                 "n/a" if par is None else "%.2f" % (par / 1e6),
                 "n/a" if trn is None or not np.isfinite(trn) else "%.1f" % trn,
                 "%.2f" % inf, s1, s2))
    return t + "\\bottomrule\n\\end{tabular}\n\\end{table}\n"


def table_data(ev, cap):
    """Training-set-size ladder at the reference budget (M = 20).

    Three readings sit side by side on purpose, because the referee's decision
    rule turns on which of them moves with n_train:

      * the in-band floor falls monotonically and has not saturated;
      * the two out-of-band columns and the plateau barely move;
      * the largest single step of every row is the same band-edge interval.

    A table that reported only the first would read as "the machinery does not
    matter"; one that reported only the last would hide that the in-band floor
    is partly purchasable.  Both are in the same rows so neither can be quoted
    without the other.
    """
    rungs = _ndata_rows(ev)
    if not rungs:
        return ""
    lab = cap["data_lab"]
    t = ("\\begin{table}[t]\n\\centering\n\\caption{%s}\n\\label{tab:data}\n"
         "\\small\n\\setlength{\\tabcolsep}{3.6pt}\n"
         "\\begin{tabular}{rccccc}\n\\toprule\n%s \\\\\n\\midrule\n"
         % (cap["data"], " & ".join(lab["h"])))
    for r in rungs:
        t += ("%s & $%s$ & $%s$ & $%s$ & $%s$ & $%s\\to%s$ \\\\\n"
              % ("{:,}".format(r["n"]).replace(",", "\\,"),
                 fmt(r["inband"]), fmt(r["e4"]), fmt(r["e425"]),
                 fmt(r["plateau"]),
                 freq_label(r["step_lo"]), freq_label(r["step_hi"])))
    return t + "\\bottomrule\n\\end{tabular}\n\\end{table}\n"


def table_propnorm(cap):
    """Discrete propagator norm at the frequencies where it deviates.

    Only the informative rows are listed.  Every integer f >= 5 attains the same
    value, so printing them all would pad the table without adding information;
    the caption says so instead.  The two densified band-edge points are kept
    because one of them (4.75) is where the attained norm is also one of the
    smallest, which is exactly the coincidence a reader has to be able to check.
    """
    p = os.path.join(C.RES, "propnorm.json")
    if not os.path.exists(p):
        return ""
    d = json.load(open(p, encoding="utf-8"))
    nom = float(d["nominal"])
    at = dict(zip([round(float(f), 4) for f in d["f"]],
                  [float(v) for v in d["norm"]]))
    show = [1.5, 2.5, 3.5, 4.0, 4.25, 4.5, 4.75]
    t = ("\\begin{table}[t]\n\\centering\n\\caption{%s}\n"
         "\\label{tab:propnorm}\n\\small\n\\setlength{\\tabcolsep}{3.4pt}\n"
         "\\begin{tabular}{lcccccccc}\n\\toprule\n"
         "$f$ & " % cap["propnorm"])
    t += " & ".join(freq_label(f) for f in show)
    t += " & $\\ge5$ int. \\\\\n$\\max|\\kappa^{2}\\widehat{G}_{\\kappa}|$ & "
    t += " & ".join(fmt(at[f], 2) for f in show)
    t += " & $\\PropHighest$ \\\\\n"
    return t + "\\bottomrule\n\\end{tabular}\n\\end{table}\n"


def table_paired(cap):
    """Paired tests on the per-sample errors, one row per comparison.

    The benchmark is paired by construction, so quoting two means side by side
    throws away the design.  Every row is mean(e_A - e_B) over paired media,
    pooled over the seeds of each arm, with a 95% bootstrap CI and a two-sided
    sign-flip permutation p-value; both are produced by code/paired_table.py.

    The interval resamples the MEDIA.  The seeds are independent training runs
    but they are evaluated on identical media, so they are replicates inside a
    medium rather than extra independent units; ``n_units`` records how many
    media that is, and ``n`` how many (medium, cell) entries their mean
    contains.  An earlier revision resampled the entries and one in-band row
    therefore carried a star that the media-level test does not support.

    The last column is the standardised paired effect size (Cohen's d_z), the
    mean paired difference in units of its own standard deviation: a p-value
    says whether a difference is resolved, this says how large it is, and a
    comparison can carry a small p with a large effect or the reverse.
    """
    p = os.path.join(C.RES, "paired.json")
    if not os.path.exists(p):
        return ""
    d = json.load(open(p, encoding="utf-8"))
    lab = cap["paired_lab"]
    t = ("\\begin{table}[t]\n\\centering\n\\caption{%s}\n\\label{tab:paired}\n"
         "\\small\n\\setlength{\\tabcolsep}{3.6pt}\n"
         "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
         " & $n$ & $\\bar e_{A}$ & $\\bar e_{B}$ & $\\Delta$ (95\\%% CI) & $p$ "
         "& $d_z$ \\\\\n\\midrule\n" % cap["paired"])
    for c in d["cases"]:
        name = lab.get(c["label"], c["label"])
        star = "*" if c["lo"] > 0 or c["hi"] < 0 else ""
        t += ("%s & %d & %s & %s & $%+.3f$ $[%+.3f,%+.3f]$ & $%s$%s & $%+.2f$ "
              "\\\\\n"
              % (name, c.get("n", c["n_units"]), fmt(c["mean_a"]),
                 fmt(c["mean_b"]), c["diff"],
                 c["lo"], c["hi"], fmt_p(c["p"]), star, c.get("dz", 0.0)))
    return t + "\\bottomrule\n\\end{tabular}\n\\end{table}\n"


def table_si_paired(cap):
    """The same paired table for the supplementary material, English labels.

    The SI copy used to be hand-maintained inside SI_main.tex while this table
    was generated from results/paired.json.  Measured 2026-10-24: the two
    agreed by luck, and the seed extension of that day (W2) changes the
    pooled rows of six of the twelve cases, so the hand copy would have
    drifted silently.  One source, one rendering: the SI inputs this file.
    """
    p = os.path.join(C.RES, "paired.json")
    if not os.path.exists(p):
        return ""
    d = json.load(open(p, encoding="utf-8"))
    lab = cap["paired_lab"]
    t = ("\\begin{table}[t]\n\\centering\n\\caption{%s}\n\\label{tab:si_paired}\n"
         "\\small\n\\setlength{\\tabcolsep}{3.6pt}\n"
         "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
         " & $n$ & $\\bar e_{A}$ & $\\bar e_{B}$ & $\\Delta$ (95\\%% CI) & $p$ "
         "& $d_z$ \\\\\n\\midrule\n" % cap["si_paired"])
    for c in d["cases"]:
        name = lab.get(c["label"], c["label"])
        star = "*" if c["lo"] > 0 or c["hi"] < 0 else ""
        t += ("%s & %d & %s & %s & $%+.3f$ $[%+.3f,%+.3f]$ & $%s$%s & $%+.2f$ "
              "\\\\\n"
              % (name, c.get("n", c["n_units"]), fmt(c["mean_a"]),
                 fmt(c["mean_b"]), c["diff"],
                 c["lo"], c["hi"], fmt_p(c["p"]), star, c.get("dz", 0.0)))
    return t + "\\bottomrule\n\\end{tabular}\n\\end{table}\n"


def table_born(ev, arms, cap):
    ref = None
    for vmax in arms:
        n = runs_for(ev, "main_fno_%s_" % arm_key(vmax))
        if n:
            ref = n[0]
            break
    t = ("\\begin{table}[t]\n\\centering\n\\caption{%s}\n\\label{tab:born}\n"
         "\\small\n\\setlength{\\tabcolsep}{4pt}\n"
         "\\begin{tabular}{ccccc}\n\\toprule\n"
         "$v_{\\max}$ & %s & $\\varepsilon(1.5)$ & $\\varepsilon(4)$ & "
         "$\\varepsilon(12)$ \\\\\n\\midrule\n" % (cap["born"], cap["h_lab"][2]))
    if ref:
        for vmax in arms:
            for o in (1, 5):
                rows = ev[ref]["born_order%d" % o]["%.2f" % vmax]
                at = lambda fv: next((r["mean"] for r in rows
                                      if abs(r["f"] - fv) < 1e-9), np.nan)
                t += ("$%.2f$ & %d & %s & %s & %s \\\\\n"
                      % (vmax, o, fmt(at(1.5)), fmt(at(4.0)), fmt(at(12.0))))
    return t + "\\bottomrule\n\\end{tabular}\n\\end{table}\n"


# Where the cache scan is frozen.  It ships with results/ so that a clone
# without the 9.1 GB caches still reproduces the counts the manuscripts cite.
CACHE_INVENTORY = "cache_inventory.json"


def _inventory_load():
    """Read the frozen cache measurement, tolerating a missing or broken file."""
    p = os.path.join(C.RES, CACHE_INVENTORY)
    if not os.path.exists(p):
        return {}
    try:
        return json.load(open(p, encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def _inventory_store(key, val):
    """Record a measurement so a machine without the caches can still read it."""
    inv = _inventory_load()
    if inv.get(key) == val:
        return
    inv[key] = int(val)
    try:
        os.makedirs(C.RES, exist_ok=True)
        with open(os.path.join(C.RES, CACHE_INVENTORY), "w",
                  encoding="utf-8") as fh:
            json.dump(inv, fh, ensure_ascii=False, indent=1, sort_keys=True)
    except OSError:
        pass


def _frozen_value(key):
    """The recorded count for `key`, or 0 when it was never measured."""
    try:
        return int(_inventory_load().get(key, 0))
    except (TypeError, ValueError):
        return 0


def cache_samples(data_dir=None):
    """Number of samples in the generated training caches (header-only read).

    Accepts either a parent directory (``data/``, whose arms are subfolders) or
    a single arm directory (``data/train_a025/``).  Note the arm case: passing an
    arm directory to a ``*/*.npz`` glob matches nothing, and a caller that takes
    the "or fall back" branch would then silently publish a wrong number.

    The counts are a property of the caches, and the public release deliberately
    omits the caches themselves (9.1 GB, bit-reproducible from their seeds).  So a
    scan taken while the caches exist is frozen into results/cache_inventory.json
    and read back on a machine where they do not.  Without that record a fresh
    clone re-running the tex stage would silently emit '--' into macros the
    manuscripts cite, and the audit gate would reject the build over a defect
    that is only about where bytes are stored.  A live scan always wins when it
    is available, so a reader who regenerates the caches still gets a check of
    the frozen numbers rather than a shadow copy of them.
    """
    import zipfile
    data_dir = data_dir or C.DATA
    key = os.path.basename(os.path.normpath(data_dir)) or "data"
    tot = 0
    for pat in (os.path.join(data_dir, "*", "chunk_*.npz"),
                os.path.join(data_dir, "chunk_*.npz")):
        for p in glob.glob(pat):
            try:
                with zipfile.ZipFile(p) as z:
                    with z.open("v.npy") as fh:
                        np.lib.format.read_magic(fh)
                        shape, _, _ = np.lib.format.read_array_header_1_0(fh)
                tot += int(shape[0])
            except Exception:
                pass
    if tot > 0:
        # Keep the frozen record current with what was just measured.
        _inventory_store(key, tot)
        return tot
    # No caches on this machine, so fall back to the frozen measurement.
    return _frozen_value(key)


def dz_macros():
    """Two standardised paired effect sizes quoted in the prose.

    An external review asked that every paired comparison report an effect size
    beside its p-value, because "resolved" and "large" are different statements
    and, with the media as the paired unit, a small p can accompany a small
    effect.  The two rows here are the pair the main text contrasts: a plateau
    difference that is both resolved and large, and an in-band difference that
    is neither.
    """
    rows = {"PairDzUnet": "unet_base_vs_fno_a025",
            "PairDzWidth": "capw64_vs_fno_a025_in"}
    keys = list(rows) + ["PairPWidth"]
    p = os.path.join(C.RES, "paired.json")
    if not os.path.exists(p):
        return ["\\newcommand{\\%s}{--}" % k for k in keys]
    d = json.load(open(p, encoding="utf-8"))
    by = {c["label"]: c for c in d["cases"]}
    L = []
    for mac, label in rows.items():
        c = by.get(label)
        dz = None if c is None else c.get("dz")
        L.append("\\newcommand{\\%s}{%s}"
                 % (mac, "--" if dz is None or not np.isfinite(dz)
                    else "%+.2f" % dz))
    c = by.get(rows["PairDzWidth"])
    L.append("\\newcommand{\\PairPWidth}{%s}"
             % ("--" if c is None else fmt_p(c["p"])))
    return L


def seed_macros(ev):
    """Per-arm seed counts, so the prose cannot drift from the protocol.

    The manuscript used to say the ablations ran "one or two seeds" while the
    main arms ran three.  That was true at the time and became false after the
    replication pass, with no gate reading the sentence.  These macros are
    derived from the evaluation records themselves, so the claim in the text is
    a measurement rather than a recollection.

    ``NumSeeds`` is the minimum over all arms and ``NumSeedsMain`` the minimum
    over the five arms of Table~\\ref{tab:main}.  The latter is what that
    table's caption has to quote: after the W2 pass every arm in it carries
    six seeds, so the global minimum of three would understate the table's own
    protocol.  ``NArmsAtMax`` counts the arms that reached the maximum, which
    is how the text says *which* arms are replicated without naming them.
    """
    fam = {}
    for n in ev:
        m = re.match(r"^(.*)_s(\d+)$", n)
        if not m:
            continue
        fam.setdefault(m.group(1), set()).add(int(m.group(2)))
    if not fam:
        return ["\\newcommand{\\%s}{--}" % k for k in ("NumSeeds", "NumSeedsMax",
                                                       "NumSeedsMain",
                                                       "NArmsAtMax",
                                                       "NumArmsSeeded")]
    counts = [len(v) for v in fam.values()]
    mx = max(counts)
    main_counts = [len(fam[a]) for a in MAIN_ARMS if a in fam]
    return [
        "\\newcommand{\\NumSeeds}{%d}" % min(counts),
        "\\newcommand{\\NumSeedsMax}{%d}" % mx,
        "\\newcommand{\\NumSeedsMain}{%d}" % (min(main_counts)
                                              if main_counts else 0),
        "\\newcommand{\\NArmsAtMax}{%d}" % sum(1 for c in counts if c == mx),
        "\\newcommand{\\NumArmsSeeded}{%d}" % len(fam),
    ]


def solver_macros():
    """Solver-hygiene numbers measured by code/diag_solver_fig.py."""
    p = os.path.join(C.RES, "solver_fig.json")
    keys = ["NaiveBadPct", "NaiveSilentPct", "NaiveBatchLo", "NaiveBatchHi",
            "ZeroMaskBadPct", "FrozenBadPct", "DenseDiff", "SolverBatchN",
            "SolverBatchSize"]
    if not os.path.exists(p):
        return ["\\newcommand{\\%s}{--}" % k for k in keys]
    d = json.load(open(p, encoding="utf-8"))
    ag = d["freeze"]["aggregate"]
    # The aggregate hides a factor-of-two spread across the eight batches of the
    # campaign.  The prose used to say the second, independent batch "reproduces
    # the silent failure at a rate of X%, so the rate above is not a property of
    # one draw" -- which claims a stability the data do not show: the second
    # rate (21.6%, batch size 512) sits BELOW every one of the eight first-pass
    # batches (22.7%-46.7%).  The range is therefore generated here so the
    # sentence reports the spread instead of implying a constant.
    per = [c["naive"] / c["n"] for c in d["freeze"]["conds"]]
    return [
        "\\newcommand{\\NaiveBadPct}{%.1f}" % (100 * ag["naive"]["frac"]),
        "\\newcommand{\\NaiveSilentPct}{%.1f}" % (100 * ag["frac_silent"]),
        "\\newcommand{\\NaiveBatchLo}{%.1f}" % (100 * min(per)),
        "\\newcommand{\\NaiveBatchHi}{%.1f}" % (100 * max(per)),
        "\\newcommand{\\ZeroMaskBadPct}{%.1f}" % (100 * ag["zeromask"]["frac"]),
        "\\newcommand{\\FrozenBadPct}{%.1f}" % (100 * ag["frozen"]["frac"]),
        "\\newcommand{\\DenseDiff}{%s}" % sci(d["dense"]["max_diff"]),
        "\\newcommand{\\SolverBatchN}{%s}" % ("{:,}".format(ag["n_total"]).replace(",", "\\,")),
        # The per-batch size, as opposed to the aggregate above: the campaign
        # draws 8 heterogeneous batches and the figure talks about one of them.
        # Both numbers are quoted in captions, so both have to come from here.
        "\\newcommand{\\SolverBatchSize}{%s}" % ("{:,}".format(d["freeze"]["batch"]).replace(",", "\\,")),
    ]


def arm_curves(ev, vmax):
    """(F, plateau, err at f=4, err at f=5, cliff ratio) for one contrast arm."""
    names = runs_for(ev, "main_fno_%s_" % arm_key(vmax))
    if not names:
        return None
    F, M, _ = mean_over_seeds(ev, names, vmax)
    plateau = float(np.mean(M[F >= 6]))
    e4 = float(M[np.isclose(F, 4.0)][0]) if np.isclose(F, 4.0).any() else np.nan
    e5 = float(M[np.isclose(F, 5.0)][0]) if np.isclose(F, 5.0).any() else np.nan
    ok = np.isfinite(e4) and e4 > 0 and np.isfinite(e5)
    return F, plateau, e4, e5, (e5 / e4 if ok else np.nan), names


def precond_macros():
    """Strong-baseline solver cost: frozen-coefficient preconditioned Bi-CGSTAB.

    The cost table times the learned forward pass against an unpreconditioned
    Krylov solve and the prose frames that speed-up as an upper bound.  This
    supplies the number that bounds it: the same solve with the standard
    frozen-coefficient preconditioner (contrast replaced by its per-sample
    spatial mean, diagonal in the Fourier basis), measured by
    code/bench_precond.py on the identical problems.  The device is CPU and
    the prose says so beside the numbers; the device-independent headline is
    the iteration count.
    """
    keys = ["PrecondItPlain", "PrecondItPc", "PrecondItFactor",
            "PrecondMsPlain", "PrecondMsPc"]
    p = os.path.join(C.RES, "cost_precond.json")
    if not os.path.exists(p):
        return ["\\newcommand{\\%s}{--}" % k for k in keys]
    try:
        d = json.load(open(p, encoding="utf-8"))
        h = d["headline"]
        s = d["solver"][h["f"]]
        return [
            "\\newcommand{\\PrecondItPlain}{%d}" % int(round(s["plain_iters_med"])),
            "\\newcommand{\\PrecondItPc}{%d}" % int(round(s["pc_iters_med"])),
            "\\newcommand{\\PrecondItFactor}{%s}" % fmt(h["iter_reduction"], 1),
            "\\newcommand{\\PrecondMsPlain}{%s}" % fmt(s["plain_ms"], 1),
            "\\newcommand{\\PrecondMsPc}{%s}" % fmt(s["pc_ms"], 1),
        ]
    except Exception:
        return ["\\newcommand{\\%s}{--}" % k for k in keys]


def cost_macros():
    """Wall-clock cost of the learned operators against the reference solver.

    A study of learned Helmholtz solvers has to say what the learned operator
    costs, otherwise the accuracy numbers cannot be turned into a decision.
    Measured by code/bench_cost.py on the identical 64x64 problem, per sample,
    with an explicit CUDA synchronise.
    """
    keys = ["CostSolverMs", "CostFnoMs", "CostUnetMatchMs", "CostSpeedup"]
    p = os.path.join(C.RES, "cost.json")
    if not os.path.exists(p):
        return ["\\newcommand{\\%s}{--}" % k for k in keys]
    try:
        d = json.load(open(p, encoding="utf-8"))
        sol = max(v["per_sample_ms"] for v in d["solver"].values())
        fno = d["model"]["FNO w40 M20"]["per_sample_ms"]
        um = d["model"]["U-Net base76"]["per_sample_ms"]
        return [
            "\\newcommand{\\CostSolverMs}{%s}" % fmt(sol, 2),
            "\\newcommand{\\CostFnoMs}{%s}" % fmt(fno, 2),
            "\\newcommand{\\CostUnetMatchMs}{%s}" % fmt(um, 2),
            "\\newcommand{\\CostSpeedup}{%s}" % fmt(sol / fno, 1),
        ]
    except Exception:
        return ["\\newcommand{\\%s}{--}" % k for k in keys]


def sci(x, digits=1):
    """LaTeX scientific notation for a magnitude.

    `%.0e` produces `2e-13`, which in running text reads as a code fragment
    rather than as a number.  These macros are cited in prose, so emit
    `$2\\times10^{-13}$` (and `$10^{-9}$` when the mantissa is one).
    """
    if not np.isfinite(x):
        return "--"
    if x == 0:
        return "0"
    ax = abs(x)
    if 1e-2 <= ax < 1e3:
        return "%.*f" % (max(digits, 2), x)
    e = int(np.floor(np.log10(ax)))
    m = x / (10.0 ** e)
    if abs(abs(m) - 10.0) < 1e-4:
        m /= 10.0
        e += 1
    if abs(m - 1.0) < 5e-3:
        return "$10^{%d}$" % e
    return "$%.*f\\times10^{%d}$" % (digits, m, e)


def residual_macros():
    """Residual hygiene over the whole corpus: distribution, not just the max.

    A referee asked for the distribution of the honest backward-substitution
    residual over the entire generated corpus, an independent recomputation of
    it, and the smallest corruption the 1e-9 gate can actually detect.
    Measured by code/residual_dist.py.
    """
    keys = ["ResN", "ResMedian", "ResTail", "ResMax", "ResOverGate", "ResArms",
            "ResRecheckN", "ResRecheckMax", "ResRecheckAgree", "ResFloor",
            "ResRefineMax", "ResPsiStorePert", "ResSilentPct", "ResFrozenTrueMax",
            "ResCleanMax", "ResNonFinitePct", "ResDetectEta"]
    p = os.path.join(C.RES, "residual_dist.json")
    if not os.path.exists(p):
        return ["\\newcommand{\\%s}{--}" % k for k in keys]
    try:
        d = json.load(open(p, encoding="utf-8"))
        ov = d["corpus"]["overall"]
        L = [
            "\\newcommand{\\ResN}{%s}" % ("{:,}".format(ov["n"]).replace(",", "\\,")),
            "\\newcommand{\\ResMedian}{%s}" % sci(ov["p50"]),
            "\\newcommand{\\ResTail}{%s}" % sci(ov["p99"]),
            "\\newcommand{\\ResMax}{%s}" % sci(ov["max"]),
            "\\newcommand{\\ResOverGate}{%d}" % ov["n_over_gate"],
            "\\newcommand{\\ResArms}{%d}" % len(d["corpus"]["per_arm"]),
        ]
        rc = (d.get("recheck") or {}).get("per_arm") or []
        fl = (d.get("recheck") or {}).get("floor") or {}
        if rc and fl:
            n = sum(r["n"] for r in rc)
            L.append("\\newcommand{\\ResRecheckN}{%s}"
                     % ("{:,}".format(n).replace(",", "\\,")))
            L.append("\\newcommand{\\ResRecheckMax}{%s}"
                     % sci(fl["shipped"]["max"], 1))
            L.append("\\newcommand{\\ResRecheckAgree}{%s}"
                     % sci(max(r["agree_max_rel"] for r in rc), 1))
            L.append("\\newcommand{\\ResFloor}{%s}"
                     % sci(fl["roundtrip"]["max"], 1))
            L.append("\\newcommand{\\ResRefineMax}{%s}"
                     % sci(fl["refined"]["max"], 1))
            L.append("\\newcommand{\\ResPsiStorePert}{%s}"
                     % sci(fl["psi_storage_perturbation"]["p50"], 1))
        else:
            for k in ("ResRecheckN", "ResRecheckMax", "ResRecheckAgree",
                      "ResFloor", "ResRefineMax", "ResPsiStorePert"):
                L.append("\\newcommand{\\%s}{--}" % k)
        nv = d.get("naive") or {}
        if nv.get("n_total"):
            # The three quantities below are properties of the naive/frozen
            # comparison itself and do NOT depend on the intact/corrupted split.
            # An earlier version gated all six macros on ``corrupt.n > 0``; when
            # the ruined labels turned out to be non-finite, ``corrupt`` was
            # empty and the silent-failure rate, the frozen residual and the
            # detection threshold -- all measured -- were blanked out with it.
            cz = nv.get("clean") or {}
            L.append("\\newcommand{\\ResCleanMax}{%s}"
                     % (sci(cz["max"]) if cz.get("n") else "--"))
            L.append("\\newcommand{\\ResNonFinitePct}{%.1f}"
                     % (100 * nv["nonfinite_fraction"]))
            L.append("\\newcommand{\\ResSilentPct}{%.1f}"
                     % (100 * nv["silent_fraction"]))
            ft = nv.get("frozen_true") or {}
            L.append("\\newcommand{\\ResFrozenTrueMax}{%s}"
                     % (sci(ft["max"]) if ft.get("max") is not None else "--"))
            # ResCorruptMin / ResSepOrders used to be emitted here and were
            # always blank: the damage in this corpus is non-finite, so the
            # "finite corrupted remainder" they measured never exists.  An
            # unreferenced macro at the `--` fallback passes the audit (nothing
            # cites it) but still ships a placeholder-shaped token in the
            # delivered file, so the two quantities are not emitted at all
            # rather than emitted empty.  Nothing in either manuscript cites
            # them; restore the pair together with a corpus that has finite
            # corrupted labels.
            # smallest injected label error the 1e-9 gate flags for every sample.
            # `>` kept the LARGEST eta with a full flag rate, so the prose
            # printed 0.10 while the archive showed every accepted sample still
            # flagged down to 1e-8 -- the sentence and the data named different
            # thresholds and the macro made the wrong one look authoritative.
            eta = None
            for row in nv.get("injection", []):
                if row["frac_flagged"] >= 0.999 and (eta is None or row["eta"] < eta):
                    eta = row["eta"]
            L.append("\\newcommand{\\ResDetectEta}{%s}"
                     % (sci(eta) if eta is not None else "--"))
        else:
            for k in ("ResCleanMax", "ResNonFinitePct", "ResDetectEta",
                      "ResSilentPct", "ResFrozenTrueMax"):
                L.append("\\newcommand{\\%s}{--}" % k)
        return L
    except Exception:
        return ["\\newcommand{\\%s}{--}" % k for k in keys]


def realgeo_macros():
    """Third evidence rung: real geological structure (results/realgeo.json).

    Emits the few numbers the manuscript quotes.  Everything is derived from
    the JSON, so the prose cannot assert a direction the measurement does not
    support -- in particular ``RealPlateauOverlapArms`` records how many
    contrast arms show overlapping plateau ranges, which is what keeps the text
    from claiming a family-level separation the data does not establish.
    """
    keys = ["RealWinN", "RealGeoN", "RealGeoTestN", "RealFsReal", "RealFsSyn",
            "RealJumpReal", "RealJumpSyn", "RealJumpDev", "RealFsDev",
            "RealPlateauGeoLo", "RealPlateauGeoHi", "RealPlateauSynLo",
            "RealPlateauSynHi", "RealPlateauOverlapArms", "RealArmsN",
            "RealBandGeoPct", "RealBandSynPct", "RealBandRatio", "RealBandMode",
            "RealGradLo", "RealGradHi", "RealExtentXkm", "RealExtentZkm",
            "RealMirrorR", "RealModelNX", "RealModelNZ", "RealSpacingM"]
    p = os.path.join(C.RES, "realgeo.json")
    if not os.path.exists(p):
        return ["\\newcommand{\\%s}{--}" % k for k in keys]
    d = json.load(open(p, encoding="utf-8"))
    s = d.get("summary")
    if not s:
        return ["\\newcommand{\\%s}{--}" % k for k in keys]
    V = d["vmax_list"]
    fam = s["family"]
    kk = "%.2f" % 0.25 if ("%.2f" % 0.25) in fam["real"] else "%.2f" % V[2]
    band = s["bands"]
    geo_names = [g for g in d["order"] if g.startswith("geo_")]
    syn_names = [g for g in d["order"] if g.startswith("syn_")]
    # the mode at which the two families' input fields differ most, in relative
    # terms, is the one worth quoting: at M=4 the gap is already ~100x
    M_REF = 4
    bg = max(band[g][str(M_REF)] for g in geo_names)
    bs = max(band[g][str(M_REF)] for g in syn_names)
    dev = max(abs(fam["real"]["%.2f" % v]["r5"] - fam["synthetic"]["%.2f" % v]["r5"])
              / fam["synthetic"]["%.2f" % v]["r5"] for v in V)
    fsdev = max(abs(fam["real"]["%.2f" % v]["f_star"]
                    - fam["synthetic"]["%.2f" % v]["f_star"]) for v in V)
    overlap = sum(1 for v in V
                  if fam["real"]["%.2f" % v]["plateau_lo"] <= fam["synthetic"]["%.2f" % v]["plateau_hi"]
                  and fam["synthetic"]["%.2f" % v]["plateau_lo"] <= fam["real"]["%.2f" % v]["plateau_hi"])
    rough = d["source"]["window_roughness"]
    xs = d["source"]["x_m"]; zs = d["source"]["z_m"]
    # the whole-sample (integer-lag) agreement with the independently hosted
    # mirror; the resampling-aware fit also lives in the archive record
    mirror_r = d["source"]["cross_check"]["marmousi_mirror_300x298"]["whole_sample_r"]
    return [
        "\\newcommand{\\RealWinN}{%d}" % len(geo_names),
        "\\newcommand{\\RealGeoN}{%d}" % len(syn_names),
        "\\newcommand{\\RealGeoTestN}{%d}" % d["n"],
        "\\newcommand{\\RealFsReal}{%s}" % fmt(fam["real"][kk]["f_star"], 3),
        "\\newcommand{\\RealFsSyn}{%s}" % fmt(fam["synthetic"][kk]["f_star"], 3),
        # The reference arm's own pair difference, and the largest such
        # difference over the arms.  The prose named the first quantity and
        # quoted the second: at v_max = 0.25 the pair is 4.333 against 4.417, a
        # difference of 0.084 (rendered as 0.08), while `RealFsDev` = 0.25 is the maximum over the
        # five arms, attained at v_max = 0.50.  Keep them as separate macros so
        # a sentence cannot silently swap one for the other again.
        "\\newcommand{\\RealFsDiff}{%s}"
        % fmt(abs(fam["real"][kk]["f_star"] - fam["synthetic"][kk]["f_star"]), 2),
        "\\newcommand{\\RealJumpReal}{%s}" % fmt(fam["real"][kk]["r5"], 2),
        "\\newcommand{\\RealJumpSyn}{%s}" % fmt(fam["synthetic"][kk]["r5"], 2),
        "\\newcommand{\\RealJumpDev}{%s}" % fmt(100 * dev, 1),
        "\\newcommand{\\RealFsDev}{%s}" % fmt(fsdev, 2),
        "\\newcommand{\\RealPlateauGeoLo}{%s}" % fmt(fam["real"][kk]["plateau_lo"]),
        "\\newcommand{\\RealPlateauGeoHi}{%s}" % fmt(fam["real"][kk]["plateau_hi"]),
        "\\newcommand{\\RealPlateauSynLo}{%s}" % fmt(fam["synthetic"][kk]["plateau_lo"]),
        "\\newcommand{\\RealPlateauSynHi}{%s}" % fmt(fam["synthetic"][kk]["plateau_hi"]),
        "\\newcommand{\\RealPlateauOverlapArms}{%d}" % overlap,
        "\\newcommand{\\RealArmsN}{%d}" % len(V),
        "\\newcommand{\\RealBandGeoPct}{%.2f}" % (100 * bg),
        "\\newcommand{\\RealBandSynPct}{%.3f}" % (100 * bs),
        "\\newcommand{\\RealBandRatio}{%.0f}" % (bg / bs),
        "\\newcommand{\\RealBandMode}{%d}" % M_REF,
        "\\newcommand{\\RealGradLo}{%.1f}" % min(rough.values()),
        "\\newcommand{\\RealGradHi}{%.1f}" % max(rough.values()),
        "\\newcommand{\\RealExtentXkm}{%.1f}" % ((xs[1] - xs[0]) / 1000.0),
        "\\newcommand{\\RealExtentZkm}{%.1f}" % ((zs[1] - zs[0]) / 1000.0),
        "\\newcommand{\\RealMirrorR}{%s}" % fmt(mirror_r, 2),
        "\\newcommand{\\RealModelNX}{%d}" % d["source"]["used_shape"][1],
        "\\newcommand{\\RealModelNZ}{%d}" % d["source"]["used_shape"][0],
        "\\newcommand{\\RealSpacingM}{%s}" % fmt(d["source"]["spacing_m"], 0),
    ]


# ---------------------------------------------------------------------------
# Training-set-size ladder: the control that was missing behind the paper's
# claim that more data do not help.
#
# The referee's decision rule (expert2 P1-3) was explicit -- if at a fixed band
# the cliff position and the plateau are invariant to n_train, the claim stands;
# otherwise the claim has to go.  Measured here, the *position* is exactly
# invariant (the largest single step of the curve is f = 4 -> 4.25 at every
# rung) and the plateau moves by only a few percent, while the in-band floor
# moves by a factor of three and has NOT saturated.  So the claim survives only
# in the form "data buy in-band accuracy, not out-of-band reach": the flat
# sentence "more data ... do not help" had to be rewritten, and the in-band
# floor can no longer be attributed to the mode budget alone.
#
# The top rung is the reference arm itself, so the ladder is nested inside the
# mode sweep (both at v_max = 0.25, M = 20) and the sample count of every rung
# is read from its run config rather than restated here -- a hand-written
# "12288" would be exactly the kind of count assertion that goes stale silently.
# ---------------------------------------------------------------------------

NDATA_RUNGS = (("ndata_n01536_", 1536), ("ndata_n03072_", 3072),
               ("ndata_n06144_", 6144), ("main_fno_a025_", 12288))

NDATA_MACROS = ("NDataN", "NDataList", "NDataInList", "NDataPlatList",
                "NDataOutList", "NDataSpan", "NDataLastDoubling", "NDataDoublings",
                "NDataPlatSpread",
                "NDataOutSpread", "NDataEdgeSpread", "NDataEdgeLo",
                "NDataEdgeHi", "NDataStepSpread", "NDataTop", "NDataPool",
                "NDataFstarList", "NDataRunsPerRung")


def _ndata_rows(ev):
    """One record per rung of the training-set-size ladder."""
    rungs = []
    for prefix, nominal in NDATA_RUNGS:
        names = runs_for(ev, prefix)
        if not names:
            continue
        cfgp = os.path.join(C.run_dir(names[0], create=False), "config.json")
        n_tr = nominal
        if os.path.exists(cfgp):
            n_tr = int(json.load(open(cfgp, encoding="utf-8"))
                       .get("n_train", nominal))
        F, M_, _ = mean_over_seeds(ev, names, 0.25)
        ib, _ = inband(ev, names, 0.25)

        def at(fv, arr=M_):
            m = np.isclose(F, fv)
            return float(arr[m][0]) if m.any() else np.nan

        # the largest single step of the curve, with the interval it spans
        k = int(np.argmax(np.diff(M_))) if M_.size > 1 else 0
        rungs.append(dict(n=n_tr, runs=len(names), inband=ib, e4=at(4.0),
                          e425=at(4.25), e5=at(5.0),
                          plateau=float(np.mean(M_[F >= 6])),
                          step_lo=float(F[k]), step_hi=float(F[k + 1]),
                          fstar=f_star(F, M_)))
    rungs.sort(key=lambda r: r["n"])
    return rungs


def ndata_macros(ev):
    """Macros for the training-set-size ladder (M = 20, v_max = 0.25)."""
    L = []
    rungs = _ndata_rows(ev)
    if len(rungs) < 2:
        for mac in NDATA_MACROS:
            L.append("\\newcommand{\\%s}{--}" % mac)
        return L
    pct = lambda xs, d=1: fmt(100.0 * (max(xs) - min(xs)) / max(np.mean(xs), 1e-30), d)
    L.append("\\newcommand{\\NDataN}{%d}" % len(rungs))
    # The per-frequency sample count of the main protocol.  It is read out of
    # the delivered evaluation records rather than typed, because the third
    # level's caption has to contrast its own denominator against this one and
    # a hard-coded 200 would silently drift the day the protocol changes.
    _n = None
    for _r in ev.values():
        if isinstance(_r, dict) and _r.get("n"):
            _n = int(_r["n"])
            break
    if _n is None:
        for _pref in ("main_fno_a025_",):
            _ns = runs_for(ev, _pref)
            for _k in _ns:
                _v = ev.get(_k, {}).get("n")
                if _v:
                    _n = int(_v)
                    break
            if _n:
                break
    L.append("\\newcommand{\\MainNMedia}{%s}"
             % ("{:,}".format(_n).replace(",", "\\,") if _n else "--"))
    L.append("\\newcommand{\\NDataList}{%s}"
             % ", ".join("{:,}".format(r["n"]).replace(",", "\\,")
                         for r in rungs))
    L.append("\\newcommand{\\NDataInList}{%s}"
             % ", ".join(fmt(r["inband"]) for r in rungs))
    L.append("\\newcommand{\\NDataPlatList}{%s}"
             % ", ".join(fmt(r["plateau"]) for r in rungs))
    L.append("\\newcommand{\\NDataOutList}{%s}"
             % ", ".join(fmt(r["e425"]) for r in rungs))
    L.append("\\newcommand{\\NDataFstarList}{%s}"
             % ", ".join(na_or(r["fstar"]) for r in rungs))
    L.append("\\newcommand{\\NDataSpan}{%s}"
             % fmt(rungs[0]["inband"] / rungs[-1]["inband"], 1))
    # The last doubling on its own.  Quoted because the ladder's first rungs are
    # cheap and a reader is entitled to ask whether the top rung is still doing
    # anything; if this ever comes out near 1.0 the sentence "has not saturated"
    # stops being true and the macro will say so.
    if len(rungs) >= 2 and rungs[-2]["inband"] > 0:
        L.append("\\newcommand{\\NDataLastDoubling}{%s}"
                 % fmt(rungs[-2]["inband"] / rungs[-1]["inband"], 1))
    else:
        L.append("\\newcommand{\\NDataLastDoubling}{--}")
    L.append("\\newcommand{\\NDataDoublings}{%d}" % (len(rungs) - 1))
    L.append("\\newcommand{\\NDataPlatSpread}{%s}"
             % pct([r["plateau"] for r in rungs]))
    L.append("\\newcommand{\\NDataOutSpread}{%s}" % pct([r["e5"] for r in rungs]))
    L.append("\\newcommand{\\NDataEdgeSpread}{%s}"
             % pct([r["e425"] for r in rungs], 0))
    L.append("\\newcommand{\\NDataEdgeLo}{%s}" % freq_label(rungs[0]["step_lo"]))
    L.append("\\newcommand{\\NDataEdgeHi}{%s}" % freq_label(rungs[0]["step_hi"]))
    L.append("\\newcommand{\\NDataStepSpread}{%s}"
             % fmt(max(r["step_lo"] for r in rungs)
                   - min(r["step_lo"] for r in rungs), 2))
    L.append("\\newcommand{\\NDataTop}{%s}"
             % "{:,}".format(rungs[-1]["n"]).replace(",", "\\,"))
    pool = cache_samples(os.path.join(C.DATA, "train_a025"))
    L.append("\\newcommand{\\NDataPool}{%s}"
             % ("{:,}".format(pool).replace(",", "\\,") if pool else "--"))
    L.append("\\newcommand{\\NDataRunsPerRung}{%d}" % rungs[0]["runs"])
    return L


def uband_macros(ev):
    """Band-width ablation read across two families, not one.

    The band-width claim ("widening the band is the only lever that pushes the
    boundary out") was originally verified on the FNO alone, while the
    architecture table shows the U-Net has no step at all -- so a referee could
    read it as a one-family result.  The wide-band U-Net arm closes that: the
    U-Net has no step in either band, but the *gain* from widening is
    concentrated just beyond the new edge exactly as the FNO's step is, and its
    in-band floor does not move.  Both halves are emitted so the prose cannot
    quote the gain without the missing step.
    """
    L = []

    def arm(prefix):
        names = runs_for(ev, prefix)
        if not names:
            return None
        F, M_, _ = mean_over_seeds(ev, names, 0.25)
        ib, _ = inband(ev, names, 0.25)
        k = int(np.argmax(np.diff(M_))) if M_.size > 1 else 0
        at = lambda fv: (float(M_[np.isclose(F, fv)][0])
                         if np.isclose(F, fv).any() else np.nan)
        return dict(inband=ib, e5=at(5.0), e6=at(6.0), e8=at(8.0),
                    e12=at(12.0), plateau=float(np.mean(M_[F >= 6])),
                    step=float(np.diff(M_)[k]), step_lo=float(F[k]),
                    step_hi=float(F[k + 1]))

    f4, f6 = arm("main_fno_a025_"), arm("band_b6_")
    u4, u6 = arm("base_unet_a025_"), arm("uband_b6_")
    if not (f4 and f6 and u4 and u6):
        for mac in ("UbandInLo", "UbandInHi", "UbandGainFive", "UbandGainSix",
                    "UbandGainEight", "UbandGainTwelve", "UbandPlateauLo",
                    "UbandPlateauHi", "UbandPlateauGain", "UbandStepLo",
                    "UbandStepHi", "FnoStepLo", "FnoStepHi", "FnoPlateauGain",
                    "UbandEdgeHi"):
            L.append("\\newcommand{\\%s}{--}" % mac)
        return L

    def rat(a, b, n=1):
        return fmt(a / b, n) if (np.isfinite(a) and np.isfinite(b) and b > 0) else "--"

    rat2 = lambda a, b: rat(a, b, 2)

    L.append("\\newcommand{\\UbandInLo}{%s}" % fmt(u4["inband"]))
    L.append("\\newcommand{\\UbandInHi}{%s}" % fmt(u6["inband"]))
    L.append("\\newcommand{\\UbandGainFive}{%s}" % rat(u4["e5"], u6["e5"]))
    L.append("\\newcommand{\\UbandGainSix}{%s}" % rat(u4["e6"], u6["e6"]))
    L.append("\\newcommand{\\UbandGainEight}{%s}" % rat(u4["e8"], u6["e8"]))
    L.append("\\newcommand{\\UbandGainTwelve}{%s}" % rat(u4["e12"], u6["e12"]))
    L.append("\\newcommand{\\UbandPlateauLo}{%s}" % fmt(u4["plateau"]))
    L.append("\\newcommand{\\UbandPlateauHi}{%s}" % fmt(u6["plateau"]))
    L.append("\\newcommand{\\UbandPlateauGain}{%s}"
             % rat2(u4["plateau"], u6["plateau"]))
    L.append("\\newcommand{\\UbandStepLo}{%s}" % fmt(u4["step"]))
    L.append("\\newcommand{\\UbandStepHi}{%s}" % fmt(u6["step"]))
    L.append("\\newcommand{\\FnoStepLo}{%s}" % fmt(f4["step"]))
    L.append("\\newcommand{\\FnoStepHi}{%s}" % fmt(f6["step"]))
    L.append("\\newcommand{\\FnoPlateauGain}{%s}"
             % rat2(f4["plateau"], f6["plateau"]))
    L.append("\\newcommand{\\UbandEdgeHi}{%s}" % freq_label(f6["step_hi"]))
    return L


def macros(ev, arms, res):
    lo, hi = [], []
    cliffs, jumps = [], []
    for vmax in arms:
        r = arm_curves(ev, vmax)
        if r is None:
            continue
        F, plateau, e4, e5, cliff, names = r
        ib, _ = inband(ev, names, vmax)
        lo.append(ib); hi.append(plateau)
        # A ratio-based "cliff" is only meaningful for an arm whose band-edge
        # error is still well below its own plateau: an arm already at plateau
        # level at f = f_train has a boundary but no ratio signature (this is
        # what the cross-contrast reading in Fig. "cross" demonstrates).
        # Evaluating arm-matched, every arm passes today, but the rule is stated
        # in the manuscript so the aggregate can never silently absorb an arm
        # that should not contribute.
        if np.isfinite(cliff) and e4 < 0.5 * plateau:
            cliffs.append(cliff)
            jumps.append(e5 - e4)
    # Indexed rather than unpacked: the summary rows gained a seed-SD field, and
    # a positional unpack here would have silently bound `a` to the wrong column
    # the moment that happened.
    alphas = [row[5] for row in res if np.isfinite(row[5])]
    # The same model admits wildly different power-law exponents depending on
    # where the fit starts and stops, because the plateau truncates the rise.
    # Both ends of that range are reported so the manuscript can say so.
    edge = []
    for vmax in arms:
        r = arm_curves(ev, vmax)
        if r is None or not np.isfinite(r[4]) or r[4] <= 0:
            continue
        edge.append(float(np.log(r[4]) / np.log(5.0 / 4.0)))
    L = []
    # Training/validation split and the per-arm cache size.  Read from a real
    # run config and from the cache headers rather than restating literals that
    # can silently drift from train.py (the split is n_train + n_train*val_frac,
    # not a round number, and the cache is larger than the two splits combined).
    _ntr, _nva, _nep = 12288, 1228, 55
    for _n in runs_for(ev, "main_fno_a025_"):
        _cfgp = os.path.join(C.run_dir(_n, create=False), "config.json")
        if os.path.exists(_cfgp):
            _c = json.load(open(_cfgp, encoding="utf-8"))
            _ntr = int(_c.get("n_train", _ntr))
            _nva = int(_c.get("n_val", _nva))
            _nep = int(_c.get("epochs", _nep))
            break
    _arm_cache = cache_samples(os.path.join(C.DATA, "train_a025")) or (_ntr + _nva)
    L.append("\\newcommand{\\NumTrainSamples}{%s}"
             % "{:,}".format(_ntr).replace(",", "\\,"))
    L.append("\\newcommand{\\NumValSamples}{%s}"
             % "{:,}".format(_nva).replace(",", "\\,"))
    L.append("\\newcommand{\\ArmCacheSamples}{%s}"
             % "{:,}".format(_arm_cache).replace(",", "\\,"))
    # The cache is written in 512-sample blocks and the training ladder's top
    # rung is 12 288, so a slice of every cache is generated and never used.
    # Reviewers are entitled to ask what that slice is for, and the answer is
    # "nothing": quoting the unused count rather than leaving it to be derived
    # from the difference is what makes that answer checkable.
    L.append("\\newcommand{\\ArmCacheUnused}{%s}"
             % "{:,}".format(max(0, _arm_cache - _ntr - _nva)).replace(",", "\\,"))
    L.append("\\newcommand{\\NumEpochs}{%d}" % _nep)
    # The three design constants of the protocol.  They are quoted in the
    # protocol section and again by the third evidence rung, and two copies of
    # a constant is exactly the arrangement that drifts, so both sites read
    # from common.py through these macros.
    L.append("\\newcommand{\\ContrastCorrLen}{%s}" % fmt(C.MM_REF, 2))
    L.append("\\newcommand{\\LossTan}{%s}" % fmt(C.GAMMA, 2))
    L.append("\\newcommand{\\GridN}{%d}" % C.N)
    # Test-grid size and the explicit list of test frequencies.  Restating
    # either by hand is how "the test grid is [13 values], i.e. nine
    # frequencies outside the band" silently survives a change to F_TEST.
    L.append("\\newcommand{\\NumTestFreq}{%d}" % len(C.F_TEST))
    L.append("\\newcommand{\\NumOutBandFreq}{%d}"
             % sum(1 for _f in C.F_TEST if _f > C.F_TRAIN[1] + 1e-9))
    L.append("\\newcommand{\\TestFreqList}{%s}"
             % ", ".join("%g" % _f for _f in C.F_TEST))
    # Disk footprint of the generated caches, which the release deliberately
    # omits (they are deterministic functions of their seeds, so the generator
    # ships instead of the data).  Scoped to data/train_*: data/real is not a
    # cache but the downloaded input of the third evidence level, and it does
    # ship.  Measured, not restated, so the sentence in the availability
    # statement cannot drift from what is actually on disk.
    try:
        _tot = 0
        for _sub in sorted(glob.glob(os.path.join(C.DATA, "train_*"))):
            for _root, _dirs, _files in os.walk(_sub):
                for _fn in _files:
                    _tot += os.path.getsize(os.path.join(_root, _fn))
        if _tot > 0:
            _inventory_store("disk_bytes", _tot)
        else:
            # The caches are not on this machine, so read the measurement that
            # was frozen when they were.
            _tot = _frozen_value("disk_bytes")
        L.append("\\newcommand{\\CacheDiskGB}{%s}"
                 % (fmt(_tot / 1e9, 1) if _tot > 0 else "--"))
    except Exception:
        L.append("\\newcommand{\\CacheDiskGB}{--}")
    # `lo`/`hi` are empty whenever no eval record matches the current test grid
    # (e.g. a half-finished re-evaluation after F_TEST changed).  Emit the '--'
    # fallback rather than raising, so the failure surfaces as the audit gate's
    # "macro cited but unresolved" message instead of an opaque traceback.
    L.append("\\newcommand{\\InBandLo}{%s}" % (fmt(min(lo)) if lo else "--"))
    L.append("\\newcommand{\\InBandHi}{%s}" % (fmt(max(lo)) if lo else "--"))
    L.append("\\newcommand{\\PlateauLo}{%s}" % (fmt(min(hi)) if hi else "--"))
    L.append("\\newcommand{\\PlateauHi}{%s}" % (fmt(max(hi)) if hi else "--"))
    # The plateau read off the median and the 90th percentile, per arm and then
    # aggregated.  The manuscript used to support "the plateau is not a metric
    # artefact" by pointing at the summary table, which carries no such columns;
    # these numbers are what that sentence is actually about.
    med, q90 = [], []
    for vmax in arms:
        names = runs_for(ev, "main_fno_%s_" % arm_key(vmax))
        if not names:
            continue
        pm, pq = [], []
        for n in names:
            rows = ev[n]["sweep"]["%.2f" % vmax]
            pm.append(float(np.mean([r["median"] for r in rows if r["f"] >= 6.0])))
            pq.append(float(np.mean([r["q90"] for r in rows if r["f"] >= 6.0])))
        if pm:
            med.append(float(np.mean(pm)))
            q90.append(float(np.mean(pq)))
    for mac, vals in (("MedPlateauLo", med), ("MedPlateauHi", med),
                      ("PctilePlateauLo", q90), ("PctilePlateauHi", q90)):
        L.append("\\newcommand{\\%s}{%s}"
                 % (mac, fmt(min(vals) if "Lo" in mac else max(vals))
                    if vals else "--"))
    L.append("\\newcommand{\\CliffLo}{%s}" % (fmt(min(cliffs), 2) if cliffs else "--"))
    L.append("\\newcommand{\\CliffHi}{%s}" % (fmt(max(cliffs), 2) if cliffs else "--"))
    L.append("\\newcommand{\\JumpLo}{%s}" % (fmt(min(jumps)) if jumps else "--"))
    L.append("\\newcommand{\\JumpHi}{%s}" % (fmt(max(jumps)) if jumps else "--"))
    L.append("\\newcommand{\\NumCliffArms}{%d}" % len(cliffs))
    L.append("\\newcommand{\\AlphaLo}{%s}" % (fmt(min(alphas), 2) if alphas else "--"))
    L.append("\\newcommand{\\AlphaHi}{%s}" % (fmt(max(alphas), 2) if alphas else "--"))
    L.append("\\newcommand{\\AlphaEdge}{%s}" % (fmt(max(edge), 1) if edge else "--"))
    # fraction of its own plateau that the error has reached at the band edge
    rise = []
    for vmax in arms:
        r = arm_curves(ev, vmax)
        if r is None or not np.isfinite(r[2]) or r[1] <= 0:
            continue
        rise.append(float(r[2] / r[1]))
    L.append("\\newcommand{\\RiseLo}{%s}" % (fmt(min(rise), 2) if rise else "--"))
    L.append("\\newcommand{\\RiseHi}{%s}" % (fmt(max(rise), 2) if rise else "--"))
    # Seed-level spread of the two locators, so the arm-to-arm ranges quoted
    # above are read against seed noise rather than by eye.  Both locators are
    # computed per seed (each seed's own e5/e4 and e4/plateau) and the largest
    # within-arm seed SD is emitted beside the arm-to-arm spread.
    r_sd, rise_sd = [], []
    for vmax in arms:
        names = runs_for(ev, "main_fno_%s_" % arm_key(vmax))
        if not names:
            continue
        rs, rises = [], []
        for n in names:
            F1, M1, _ = mean_over_seeds(ev, [n], vmax)
            e4 = M1[np.isclose(F1, 4.0)]
            e5 = M1[np.isclose(F1, 5.0)]
            plat = float(np.mean(M1[F1 >= 6.0])) if (F1 >= 6.0).any() else np.nan
            if e4.size and e4[0] > 0 and e5.size and plat > 0:
                rs.append(float(e5[0] / e4[0]))
                rises.append(float(e4[0] / plat))
        if rs:
            r_sd.append(float(np.std(rs)))
            rise_sd.append(float(np.std(rises)))
    L.append("\\newcommand{\\RatioSeedSDMax}{%s}"
             % (fmt(max(r_sd), 2) if r_sd else "--"))
    L.append("\\newcommand{\\RiseSeedSDMax}{%s}"
             % (fmt(max(rise_sd), 3) if rise_sd else "--"))
    L.append("\\newcommand{\\RatioArmSpread}{%s}"
             % (fmt(max(cliffs) - min(cliffs), 2) if cliffs else "--"))
    L.append("\\newcommand{\\RiseArmSpread}{%s}"
             % (fmt(max(rise) - min(rise), 2) if rise else "--"))
    # The relative-frequency window (in units of f/f_train) that the step is
    # squeezed into: the FIRST test frequency beyond the band edge at which
    # every arm has reached 85% of its own plateau.  This is the number that
    # makes a power-law exponent unidentifiable here, so it is derived from the
    # evaluated grid rather than hardcoded -- if F_TEST moves, the sentence
    # moves with it.
    rw = []
    for vmax in arms:
        names = runs_for(ev, "main_fno_%s_" % arm_key(vmax))
        if not names:
            continue
        Fc, Mc, _ = mean_over_seeds(ev, names, vmax)
        pl = float(np.mean(Mc[Fc >= 6])) if (Fc >= 6).any() else np.nan
        if not np.isfinite(pl) or pl <= 0:
            continue
        out = Fc[Fc > C.F_TRAIN[1] + 1e-9]
        Mo = Mc[Fc > C.F_TRAIN[1] + 1e-9]
        hit = out[Mo >= 0.85 * pl]
        if hit.size:
            rw.append(float((hit.min() - C.F_TRAIN[1]) / C.F_TRAIN[1]))
    L.append("\\newcommand{\\RiseWindow}{%s}" % (fmt(max(rw), 2) if rw else "--"))
    # f/f_train at which the step is complete (= 1 + the window above).
    L.append("\\newcommand{\\PlateauAt}{%s}"
             % (fmt(1.0 + max(rw), 2) if rw else "--"))
    # parameter counts of the four compared architectures (from the a025 runs).
    # Two decimals, matching Table 3's columns: a referee comparing the prose
    # against the table must not find 10.3 in one and 10.25 in the other.
    for mac, pref in [("FnoParams", "main_fno_a025_"),
                      ("UnetParams", "base_unet_a025_"),
                      ("PmatchParams", "pmatch_unet_a025_"),
                      ("DonParams", "base_don_a025_")]:
        names = runs_for(ev, pref)
        v = (ev[names[0]]["params"] / 1e6) if names else float("nan")
        L.append("\\newcommand{\\%s}{%s}" % (mac, fmt(v, 2)))
    # how far the small U-Net sits below the FNO
    _fn = ev[runs_for(ev, "main_fno_a025_")[0]]["params"] if runs_for(ev, "main_fno_a025_") else 0
    names = runs_for(ev, "base_unet_a025_")
    if names and _fn:
        L.append("\\newcommand{\\UnetShortfall}{%.1f}" % (_fn / ev[names[0]]["params"]))
    else:
        L.append("\\newcommand{\\UnetShortfall}{--}")
    names = runs_for(ev, "base_don_a025_")
    if names and _fn:
        L.append("\\newcommand{\\DonShortfall}{%.1f}" % (_fn / ev[names[0]]["params"]))
    else:
        L.append("\\newcommand{\\DonShortfall}{--}")
    # ---- plateau height per architecture -----------------------------------
    # The cliff LOCATION is architecture-independent, but the plateau HEIGHT is
    # not, and on the measured data the U-Net's is *lower* than the FNO's even
    # though the U-Net is several times smaller.  These macros carry those
    # numbers so the prose cannot assert a direction that the data contradicts.
    def _arch_plateau(pref, vmax):
        nm = runs_for(ev, pref)
        if not nm:
            return float("nan")
        Fh, Mh, _ = mean_over_seeds(ev, nm, vmax)
        sel = Mh[Fh >= 6]
        return float(np.mean(sel)) if sel.size else float("nan")

    def _arch_r5(pref, vmax):
        """Band-edge ratio eps(5)/eps(4).  This, not the plateau, is what
        distinguishes a family that steps at the band edge from one that decays
        smoothly, so the prose needs it as a number rather than as a visual
        impression of Fig. 4."""
        nm = runs_for(ev, pref)
        if not nm:
            return float("nan")
        Fh, Mh, _ = mean_over_seeds(ev, nm, vmax)
        e4 = Mh[np.isclose(Fh, 4.0)]
        e5 = Mh[np.isclose(Fh, 5.0)]
        if not (e4.size and e5.size) or e4[0] == 0:
            return float("nan")
        return float(e5[0] / e4[0])

    for _v, _suf in [(0.25, ""), (1.00, "S")]:
        _pf = _arch_plateau("main_fno_%s_" % arm_key(_v), _v)
        _pu = _arch_plateau("base_unet_%s_" % arm_key(_v), _v)
        _pd = _arch_plateau("base_don_%s_" % arm_key(_v), _v)
        _pp = _arch_plateau("pmatch_unet_%s_" % arm_key(_v), _v)
        for _mac, _val in [("ArchFnoPlateau" + _suf, _pf),
                           ("ArchUnetPlateau" + _suf, _pu),
                           ("ArchDonPlateau" + _suf, _pd),
                           ("ArchPmatchPlateau" + _suf, _pp)]:
            L.append("\\newcommand{\\%s}{%s}"
                     % (_mac, fmt(_val) if np.isfinite(_val) else "--"))
        _g = _pf / _pu if (np.isfinite(_pf) and np.isfinite(_pu) and _pu > 0) else float("nan")
        L.append("\\newcommand{\\ArchUnetGain%s}{%s}"
                 % (_suf, ("%.2f" % _g) if np.isfinite(_g) else "--"))
        for _mac, _pr in [("ArchFnoRatioFive" + _suf, "main_fno_%s_" % arm_key(_v)),
                          ("ArchUnetRatioFive" + _suf, "base_unet_%s_" % arm_key(_v)),
                          ("ArchDonRatioFive" + _suf, "base_don_%s_" % arm_key(_v)),
                          ("ArchPmatchRatioFive" + _suf, "pmatch_unet_%s_" % arm_key(_v))]:
            _r = _arch_r5(_pr, _v)
            L.append("\\newcommand{\\%s}{%s}"
                     % (_mac, ("%.2f" % _r) if np.isfinite(_r) else "--"))
        # Point values at the band edge and just past it, plus the in-band mean.
        # Needed because the claim about the architectures is a claim about
        # SHAPE: a step means eps(5) >> eps(4) inside a single grid step, and a
        # smooth decay means it does not.  A plateau height alone cannot
        # distinguish the two, which is how the two got conflated before.
        for _mac, _pr in [("Fno", "main_fno_%s_" % arm_key(_v)),
                          ("Unet", "base_unet_%s_" % arm_key(_v)),
                          ("Don", "base_don_%s_" % arm_key(_v)),
                          ("Pmatch", "pmatch_unet_%s_" % arm_key(_v))]:
            _nm = runs_for(ev, _pr)
            if _nm:
                _F, _M, _ = mean_over_seeds(ev, _nm, _v)
                _e4 = (float(_M[np.isclose(_F, 4.0)][0])
                       if np.isclose(_F, 4.0).any() else np.nan)
                _e5 = (float(_M[np.isclose(_F, 5.0)][0])
                       if np.isclose(_F, 5.0).any() else np.nan)
                _ib = float(np.mean(_M[_F <= C.F_TRAIN[1] + 1e-9]))
            else:
                _e4 = _e5 = _ib = np.nan
            for _sfx, _val in [("EdgeFour", _e4), ("EdgeFive", _e5), ("In", _ib)]:
                L.append("\\newcommand{\\Arch%s%s%s}{%s}"
                         % (_mac, _sfx, _suf,
                            fmt(_val) if np.isfinite(_val) else "--"))

    # ---- the same curves divided by each family's OWN plateau ---------------
    # "No cliff" and "a lower plateau" are two independent claims, and the
    # plateau height cannot establish either of them: a family with a lower
    # plateau would look flat even if it stepped.  The quantity that separates
    # them is the curve normalised by its own plateau, where the reference
    # value is 1 by construction.  A family that STEPS rises past 1; a family
    # that decays smoothly never reaches it.  Panel (c) of Figure 5 draws
    # exactly this, and the macros below are the numbers read off it.
    for _v, _suf in [(0.25, ""), (1.00, "S")]:
        for _mac, _pr in [("Fno", "main_fno_%s_" % arm_key(_v)),
                          ("Unet", "base_unet_%s_" % arm_key(_v)),
                          ("Don", "base_don_%s_" % arm_key(_v)),
                          ("Pmatch", "pmatch_unet_%s_" % arm_key(_v))]:
            _nm = runs_for(ev, _pr)
            if not _nm:
                for _sfx in ("NormIn", "NormFive"):
                    L.append("\\newcommand{\\Arch%s%s%s}{--}"
                             % (_mac, _sfx, _suf))
                continue
            _F, _M, _ = mean_over_seeds(ev, _nm, _v)
            _pl = float(np.mean(_M[_F >= 6.0])) if (_F >= 6.0).any() else np.nan
            if not np.isfinite(_pl) or _pl <= 0:
                for _sfx in ("NormIn", "NormFive"):
                    L.append("\\newcommand{\\Arch%s%s%s}{--}"
                             % (_mac, _sfx, _suf))
                continue
            _ib = float(np.mean(_M[_F <= C.F_TRAIN[1] + 1e-9])) / _pl
            _e5 = (float(_M[np.isclose(_F, 5.0)][0]) / _pl
                   if np.isclose(_F, 5.0).any() else np.nan)
            for _sfx, _val in [("NormIn", _ib), ("NormFive", _e5)]:
                L.append("\\newcommand{\\Arch%s%s%s}{%s}"
                         % (_mac, _sfx, _suf,
                            fmt(_val, 3) if np.isfinite(_val) else "--"))

    # ---- the same quantity per seed: dispersion, and the paired gap ---------
    # The own-plateau ratio has two defensible averaging conventions: divide the
    # seed-averaged curve by the seed-averaged plateau (what Figure 5c draws and
    # what the macros above report), or take the mean of the per-seed ratios.
    # They are different numbers, separated by a Jensen term that grows with the
    # seed spread of the plateau -- so a claim read off panel (c) is only safe
    # if the two agree.  We report the per-seed convention here because the
    # paired comparison needs one number per seed, and check_claims asserts the
    # two conventions agree to 2e-3.
    for _v, _suf in [(0.25, ""), (1.00, "S")]:
        # Per-seed readings keyed by seed index.  The seed-averaged macros
        # above use every run of an arm, but a *paired* comparison needs the
        # two arms to share a seed: W1 extended the v=1.00 anchored arms to
        # six seeds while the architecture baselines stayed at three, so the
        # paired gap is computed on the common seed set (the like-for-like
        # subset) rather than on a mis-aligned broadcast.
        _per, _common = {}, None
        for _mac, _pr in [("Fno", "main_fno_%s_" % arm_key(_v)),
                          ("Unet", "base_unet_%s_" % arm_key(_v)),
                          ("Don", "base_don_%s_" % arm_key(_v)),
                          ("Pmatch", "pmatch_unet_%s_" % arm_key(_v))]:
            _vals = {}
            for _n in runs_for(ev, _pr):
                _sm = re.search(r"_s(\d+)$", _n)
                _rows = ev[_n]["sweep"]["%.2f" % _v]
                _F = np.array([r["f"] for r in _rows], float)
                _M = np.array([r["mean"] for r in _rows], float)
                _pl = _M[_F >= 6.0].mean()
                _vals[int(_sm.group(1))] = float(
                    _M[np.isclose(_F, 5.0)][0] / _pl)
            _per[_mac] = _vals
            _common = (set(_vals) if _common is None
                       else (_common & set(_vals)))
        for _mac in ("Fno", "Unet", "Don", "Pmatch"):
            _own = np.array([_per[_mac][s] for s in sorted(_per[_mac])],
                            float)
            L.append("\\newcommand{\\Arch%sNormFiveSd%s}{%s}"
                     % (_mac, _suf,
                        fmt(_own.std(ddof=1), 3) if len(_own) > 1 else "--"))
        _common = sorted(_common) if _common else []
        if len(_common) > 1 and all(_per[m] for m in _per):
            _d = (np.array([_per["Fno"][s] for s in _common], float)
                  - np.array([_per["Unet"][s] for s in _common], float))
        else:
            _d = np.array([float("nan")])
        L.append("\\newcommand{\\ArchFnoMinusUnet%s}{%s}"
                 % (_suf, fmt(_d.mean(), 3)))
        L.append("\\newcommand{\\ArchFnoMinusUnetLo%s}{%s}"
                 % (_suf, fmt(_d.min(), 3)))
        L.append("\\newcommand{\\ArchFnoMinusUnetHi%s}{%s}"
                 % (_suf, fmt(_d.max(), 3)))
        # Largest per-seed spread over the four families, for the prose to
        # bound the seed-level variability of the normalised reading in one
        # number rather than four.
        L.append("\\newcommand{\\ArchNormFiveSdMax%s}{%s}"
                 % (_suf, fmt(max(np.array(
                     [_per[m][s] for s in sorted(_per[m])], float).std(ddof=1)
                     for m in _per if len(_per[m]) > 1), 3)))

    # ---- the reach of the mode budget, on the object the networks use -------
    # NOTE `u` here is the SCATTERED field, not the total field, and that is a
    # correction rather than a restatement: every network returns
    # psi_inc + u_theta with psi_inc supplied in full, so it is never required to
    # reproduce the low-frequency content of psi, and a bound stated on psi does
    # not bound its error.  On the residual the identity is exact -- writing the
    # prediction as psi_inc + P_M u_theta, the residual splits into the in-band
    # part (P_M u_theta - P_M u) and the out-of-band part (P_M u - u), which are
    # orthogonal, hence rel_err >= ||(I - P_M) u|| / ||psi||.
    sp = os.path.join(C.RES, "spectrum.json")
    spec = None
    if os.path.exists(sp):
        spec = json.load(open(sp, encoding="utf-8"))
    if spec and "u_out_rel" in spec["data"]["0.25"]["f=%g" % C.F_TEST[1]]:
        Fsp = np.array(spec["f"], float)
        arm25 = spec["data"]["0.25"]
        rows25 = [arm25["f=%g" % f] for f in Fsp]
        sol8 = np.array([r["psi_sqrt_mean"]["8"] for r in rows25], float)
        cor8 = np.array([r["u_out_rel"]["8"] for r in rows25], float)
        inb = Fsp <= C.F_TRAIN[1] + 1e-9
        out = Fsp >= 6.0
        # Split the budgets.  The budgets actually used in the main experiments
        # are M >= 20 (M = 20 is the reference arm), and there the constraint is
        # negligible EVERYWHERE, in band and on the plateau alike -- which is what
        # makes "the plateau is a band-limitation effect" untenable rather than
        # merely unproven.  Below M = 8 it is the small budgets that bite, and
        # that is the regime the mode sweep shows to move the cliff.
        big_in = [8, 12, 20, 24, 32]
        big_out = [12, 20, 24, 32]
        cor_in = np.array([[r["u_out_rel"][str(M)] for r in rows25]
                           for M in big_in])
        cor_out = np.array([[r["u_out_rel"][str(M)] for r in rows25]
                            for M in big_out])
        L.append("\\newcommand{\\SolFracInEight}{%s}" % fmt(float(sol8[inb].max())))
        L.append("\\newcommand{\\SolFracOutEight}{%s}" % fmt(float(sol8[-1])))
        L.append("\\newcommand{\\CorrFracInBig}{%s}"
                 % fmt(float(cor_in[:, inb].max())))
        L.append("\\newcommand{\\CorrFracOutBig}{%s}"
                 % fmt(float(cor_out[:, out].max())))
        L.append("\\newcommand{\\CorrFracOutEight}{%s}" % fmt(float(cor8[-1])))
        # the same quantity for the reference arm M = 20, which is the budget the
        # main contrast sweep uses
        cor20 = np.array([r["u_out_rel"]["20"] for r in rows25], float)
        L.append("\\newcommand{\\CorrFracInTwenty}{%s}"
                 % fmt(float(cor20[inb].max())))
        L.append("\\newcommand{\\CorrFracOutTwenty}{%s}" % fmt(float(cor20[-1])))
        _pb = runs_for(ev, "mode_m08_")
        if _pb:
            _Fp, _Mp, _ = mean_over_seeds(ev, _pb, 0.25)
            L.append("\\newcommand{\\MeasInEight}{%s}"
                     % fmt(float(np.mean(_Mp[_Fp <= 4.0]))))
            L.append("\\newcommand{\\MeasOutEight}{%s}" % fmt(float(_Mp[-1])))
        else:
            L.append("\\newcommand{\\MeasInEight}{--}")
            L.append("\\newcommand{\\MeasOutEight}{--}")
        _p20 = runs_for(ev, "main_fno_a025_")
        if _p20:
            _Fq, _Mq, _ = mean_over_seeds(ev, _p20, 0.25)
            L.append("\\newcommand{\\MeasInTwenty}{%s}"
                     % fmt(float(np.mean(_Mq[_Fq <= 4.0]))))
            L.append("\\newcommand{\\MeasOutTwenty}{%s}" % fmt(float(_Mq[-1])))
        else:
            L.append("\\newcommand{\\MeasInTwenty}{--}")
            L.append("\\newcommand{\\MeasOutTwenty}{--}")
    else:
        for mac in ("SolFracInEight", "SolFracOutEight", "CorrFracInBig",
                    "CorrFracOutBig", "CorrFracOutEight", "CorrFracInTwenty",
                    "CorrFracOutTwenty", "MeasInEight", "MeasOutEight",
                    "MeasInTwenty", "MeasOutTwenty"):
            L.append("\\newcommand{\\%s}{--}" % mac)

    # ---- the trivial baseline: predict the incident field, u_theta == 0 -----
    # This is baseline-free by construction (the network adds u_theta to psi_inc)
    # and its relative error is exactly ||u||/||psi||.  It accounts for most of
    # the out-of-band plateau, so leaving it out let the plateau be read as a
    # band-limitation effect.  Reported per arm, in band and on the plateau.
    _k0 = triv_row(C.VMAX_LIST[0])
    have_triv = _k0 is not None and all(
        triv_row(v) is not None for v in C.VMAX_LIST)
    if have_triv:
        ti, to, ratio, exc = [], [], [], []
        for vmax in arms:
            tv = triv_row(vmax)
            if tv is None:
                continue
            # Frequency axis comes from the same loader as the values, so the
            # in-band/plateau masks cannot be applied to a different grid than
            # the one the baseline was evaluated on.
            Fv = np.array(C.F_TEST, float)
            tv = np.asarray(tv, float)
            ti.append(float(tv[Fv <= C.F_TRAIN[1] + 1e-9].mean()))
            tv_out = float(tv[Fv >= 6.0].mean())
            to.append(tv_out)
            r = arm_curves(ev, vmax)
            pl = r[1] if r else np.nan
            if np.isfinite(pl) and tv_out > 0:
                ratio.append(pl / tv_out)
                if pl > tv_out:
                    exc.append((vmax, pl, tv_out))
        L.append("\\newcommand{\\TrivInLo}{%s}" % (fmt(min(ti)) if ti else "--"))
        L.append("\\newcommand{\\TrivInHi}{%s}" % (fmt(max(ti)) if ti else "--"))
        L.append("\\newcommand{\\TrivOutLo}{%s}" % (fmt(min(to)) if to else "--"))
        L.append("\\newcommand{\\TrivOutHi}{%s}" % (fmt(max(to)) if to else "--"))
        L.append("\\newcommand{\\TrivRatioLo}{%s}"
                 % (fmt(min(ratio), 2) if ratio else "--"))
        L.append("\\newcommand{\\TrivRatioHi}{%s}"
                 % (fmt(max(ratio), 2) if ratio else "--"))
        if exc:
            L.append("\\newcommand{\\TrivExceedArm}{%s}" % fmt(exc[0][0], 2))
            L.append("\\newcommand{\\TrivExceedPlat}{%s}" % fmt(exc[0][1]))
            L.append("\\newcommand{\\TrivExceedBase}{%s}" % fmt(exc[0][2]))
            L.append("\\newcommand{\\TrivExceedCount}{%d}" % len(exc))
        else:
            for mac in ("TrivExceedArm", "TrivExceedPlat", "TrivExceedBase"):
                L.append("\\newcommand{\\%s}{--}" % mac)
            L.append("\\newcommand{\\TrivExceedCount}{0}")
    else:
        for mac in ("TrivInLo", "TrivInHi", "TrivOutLo", "TrivOutHi",
                    "TrivRatioLo", "TrivRatioHi", "TrivExceedArm",
                    "TrivExceedPlat", "TrivExceedBase"):
            L.append("\\newcommand{\\%s}{--}" % mac)
        L.append("\\newcommand{\\TrivExceedCount}{0}")

    # ---- what does the model emit, and in which direction? ----------------
    # Two mechanisms can produce a plateau at the baseline, and they are
    # different claims: the correction collapsed toward zero, or it is present
    # but points the wrong way.  Measured by code/diag_corr.py on the same
    # paired media as the error curves.
    #
    # The signed alignment is the quantity that decides the error LEVEL, not
    # just the mechanism.  Per sample,
    #     ||u_hat - u||^2 = ||u||^2 + ||u_hat||^2 - 2 Re<u_hat, u>
    # so ||u_hat-u||/||u|| = sqrt(1 + ratio^2 - 2*ratio*align).  An earlier
    # version of diag_corr.py reported |<u_hat,u>|, which deletes the sign and
    # with it the only term that says whether the model beats or loses to the
    # trivial predictor u_theta == 0.  err_ratio is therefore measured directly
    # per sample rather than composed from the two means: the identity holds
    # per sample, not for the expectations.
    def corr_block(path, prefix, tag):
        L = []
        if not os.path.exists(path):
            return L
        cd = json.load(open(path, encoding="utf-8"))
        Fq = [float(x) for x in cd["f"]]
        rq = [cd["ratio"]["%g" % f] for f in Fq]
        aq = [cd["signed"]["%g" % f] for f in Fq]
        bq = [cd["absalign"]["%g" % f] for f in Fq]
        eq = [cd["err_ratio"]["%g" % f] for f in Fq]
        pq = [cd["proj"]["%g" % f] for f in Fq]
        ii = [i for i, f in enumerate(Fq) if f <= C.F_TRAIN[1] + 1e-9]
        pp = [i for i, f in enumerate(Fq) if f >= 6.0]
        # the peak of err_ratio over the whole sweep: this is where the model is
        # worst relative to predicting nothing, and it is a different frequency
        # from the plateau.
        ipk = int(np.argmax(eq))
        for mac, vals in ((prefix + "EnergyIn", [rq[i] for i in ii]),
                          (prefix + "EnergyOut", [rq[i] for i in pp]),
                          (prefix + "AlignIn", [aq[i] for i in ii]),
                          (prefix + "AlignOut", [aq[i] for i in pp]),
                          (prefix + "AbsAlignOut", [bq[i] for i in pp]),
                          (prefix + "ErrIn", [eq[i] for i in ii]),
                          (prefix + "ErrOut", [eq[i] for i in pp]),
                          (prefix + "ProjOut", [pq[i] for i in pp])):
            L.append("\\newcommand{\\%s}{%s}"
                     % (mac, fmt(float(np.mean(vals))) if vals else "--"))
        L.append("\\newcommand{\\%sErrPeak}{%s}"
                 % (prefix, fmt(float(eq[ipk]))))
        L.append("\\newcommand{\\%sEnergyPeak}{%s}"
                 % (prefix, fmt(float(rq[ipk]))))
        L.append("\\newcommand{\\%sAlignPeak}{%s}"
                 % (prefix, fmt(float(aq[ipk]))))
        L.append("\\newcommand{\\%sPeakF}{%s}"
                 % (prefix, freq_label(Fq[ipk])))
        L.append("\\newcommand{\\%sIdentityResidual}{%s}"
                 % (prefix, fmt(float(cd.get("identity_residual_max",
                                              float("nan"))), 3)))
        return L

    L += corr_block(os.path.join(C.RES, "corr_energy.json"), "Corr", "a025")
    L += corr_block(os.path.join(C.RES, "corr_energy_a100.json"),
                    "CorrStrong", "a100")

    # ---- independent cross-check of the same ratio -------------------------
    # err_ratio above is measured per sample from the checkpoints.  The ratio
    # implied by the delivered table -- mean error divided by the same arm's
    # trivial baseline -- is a different aggregation of the same quantity, and
    # the two must agree.  If they do not, one of the two instruments is wrong,
    # and this is the cheapest place to find out.  The residual is expected to
    # be small and signed: a ratio of means is not a mean of per-sample ratios.
    def table_implied(vmax):
        tb = triv_row(vmax)
        if tb is None:
            return None
        names = runs_for(ev, "main_fno_%s_" % arm_key(vmax))
        if not names:
            return None
        F, M, _ = mean_over_seeds(ev, names, vmax)
        if F is None:
            return None
        F = np.asarray(F, float)
        M = np.asarray(M, float)
        ipk = int(np.argmax(M / tb))
        return dict(peak=float(M[ipk] / tb[ipk]), peak_f=float(F[ipk]),
                    out=float(M[F >= 6.0].mean() / tb[F >= 6.0].mean()))

    for vmax, prefix in ((0.25, "Corr"), (1.00, "CorrStrong")):
        ti = table_implied(vmax)
        if ti is None:
            for suf in ("TablePeak", "TableOut", "TablePeakF"):
                L.append("\\newcommand{\\%s%s}{--}" % (prefix, suf))
            continue
        L.append("\\newcommand{\\%sTablePeak}{%s}" % (prefix, fmt(ti["peak"])))
        L.append("\\newcommand{\\%sTablePeakF}{%s}"
                 % (prefix, freq_label(ti["peak_f"])))
        L.append("\\newcommand{\\%sTableOut}{%s}" % (prefix, fmt(ti["out"])))

    for mac in ("CorrEnergyIn", "CorrEnergyOut", "CorrAlignIn", "CorrAlignOut",
                "CorrAbsAlignOut", "CorrErrIn", "CorrErrOut", "CorrProjOut",
                "CorrErrPeak", "CorrEnergyPeak", "CorrAlignPeak", "CorrPeakF",
                "CorrStrongErrPeak", "CorrStrongEnergyPeak",
                "CorrStrongAlignPeak", "CorrStrongPeakF", "CorrStrongErrOut",
                "CorrStrongAlignOut", "CorrStrongEnergyOut",
                "CorrStrongAbsAlignOut",
                "CorrStrongIdentityResidual", "CorrIdentityResidual"):
        if not any(m.startswith("\\newcommand{\\" + mac + "}") for m in L):
            L.append("\\newcommand{\\%s}{--}" % mac)

    # ---- overshoot of the rise ---------------------------------------------
    # Normalised by its own plateau the error does not settle at once: for
    # v_max >= 0.25 it overshoots near f = 5 and relaxes afterwards.  Calling the
    # transition "one common step" and stopping the account at "plateau
    # attained" turned a non-monotone relaxation into a step, so both the size of
    # the overshoot and the frequency at which the arms finally settle inside
    # +-10% of their plateau are generated here.
    ov_hi, ov_lo, ov_arm, ov_f = np.nan, np.nan, np.nan, np.nan
    settle = []
    for vmax in arms:
        r = arm_curves(ev, vmax)
        if r is None:
            continue
        Fs, Ms, _ = mean_over_seeds(ev, r[5], vmax)
        pl = r[1]
        if not np.isfinite(pl) or pl <= 0:
            continue
        out = Fs > C.F_TRAIN[1] + 1e-9
        if not out.any():
            continue
        rt = Ms / pl
        j = int(np.nanargmax(np.where(out, rt, np.nan)))
        if not np.isfinite(ov_hi) or rt[j] > ov_hi:
            ov_hi, ov_arm, ov_f = float(rt[j]), float(vmax), float(Fs[j])
        # Lower end of the overshoot range is taken over the arms that actually
        # overshoot.  The weakest arm peaks at ~1.06, but it peaks BEFORE the
        # band edge has been cleared, so folding it in made the manuscript's
        # "overshoots by 1.06-1.50" describe a non-overshoot as one.
        if vmax >= 0.25:
            ov_lo = float(rt[j]) if not np.isfinite(ov_lo) else min(ov_lo, float(rt[j]))
        idx = np.where(out)[0]
        for t in idx:
            if np.all(np.abs(Ms[idx[idx >= t]] / pl - 1.0) <= 0.10):
                settle.append(float((Fs[t] - C.F_TRAIN[1]) / C.F_TRAIN[1]))
                break
    L.append("\\newcommand{\\OvershootHi}{%s}" % (fmt(ov_hi, 2) if np.isfinite(ov_hi) else "--"))
    L.append("\\newcommand{\\OvershootLo}{%s}" % (fmt(ov_lo, 2) if np.isfinite(ov_lo) else "--"))
    L.append("\\newcommand{\\OvershootArm}{%s}" % (fmt(ov_arm, 2) if np.isfinite(ov_arm) else "--"))
    L.append("\\newcommand{\\OvershootF}{%s}" % (fmt(ov_f, 1) if np.isfinite(ov_f) else "--"))
    L.append("\\newcommand{\\SettleWindow}{%s}"
             % (fmt(max(settle), 2) if settle else "--"))
    # The same number in absolute frequency, for the sentences that say where the
    # curve is flat again.  Derived, so a change of F_TRAIN moves the sentence.
    L.append("\\newcommand{\\SettleF}{%s}"
             % (fmt(C.F_TRAIN[1] * (1.0 + max(settle)), 1) if settle else "--"))

    # ---- discrete propagator norm ------------------------------------------
    # The design argument for "f is the only variable" was that sup |kappa^2 G|
    # is independent of k.  That holds for the continuum symbol only; on the grid
    # the symbol is sampled at discrete |xi|^2, so the attained norm is frequency
    # dependent.  Measured, not restated.
    #
    # Two constants are carried, not one.  `nominal` is the EXACT continuum sup
    # (1 + gamma^2)/(2 gamma) = 10.025 at gamma = 0.05; `PropNominalLeading` is
    # its gamma -> 0 limit 1/(2 gamma) = 10.0, kept only for the places that quote
    # the leading-order scale.  An earlier revision emitted the leading-order
    # value as `PropNominal` at one decimal, so the text asserted the continuum
    # norm was 10 while its own table printed 10.02 -- a 0.25% = gamma^2 gap that
    # a referee reads as a contradicting pair of numbers.
    pp = os.path.join(C.RES, "propnorm.json")
    if os.path.exists(pp):
        pd = json.load(open(pp, encoding="utf-8"))
        pf = np.array(pd["f"], float)
        pn = np.array(pd["norm"], float)
        ji, jx = int(np.argmin(pn)), int(np.argmax(pn))
        # the densified band-edge point, whichever of 4.25/4.75 the grid carries
        jedge = int(np.argmin(np.abs(pf - 4.75)))
        L.append("\\newcommand{\\PropNominal}{%s}" % fmt(pd["nominal"], 2))
        L.append("\\newcommand{\\PropNominalLeading}{%s}"
                 % fmt(pd.get("nominal_leading", 1.0 / (2.0 * pd["gamma"])), 1))
        L.append("\\newcommand{\\PropMaxF}{%s}"
                 % freq_label(float(pd.get("f_argmax", pf[jx]))))
        L.append("\\newcommand{\\PropLo}{%s}" % fmt(float(pn[ji]), 2))
        L.append("\\newcommand{\\PropLoF}{%s}" % fmt(float(pf[ji]), 1))
        L.append("\\newcommand{\\PropHighest}{%s}" % fmt(float(pn[jx]), 2))
        # The grid maximum does NOT attain the continuum sup exactly: the closest
        # mode still misses the circle |xi| = k sqrt(1-gamma^2) by a hair, so the
        # shortfall is 0.005% at gamma = 0.05. A sentence claiming the discrete
        # symbol "reaches the continuum value" would therefore be false.
        L.append("\\newcommand{\\PropShortfallPct}{%s}"
                 % ("%.3f" % (100.0 * abs(float(pn[jx]) - pd["nominal"])
                              / pd["nominal"]) if pd["nominal"] else "--"))
        L.append("\\newcommand{\\PropEdge}{%s}" % fmt(float(pn[jedge]), 2))
        # `fmt(..., 1)` here printed the densified edge point as 4.8 -- the exact
        # rounding defect that reached Table 1.  Frequency LABELS go through
        # freq_label; only measured quantities go through fmt.
        L.append("\\newcommand{\\PropEdgeF}{%s}" % freq_label(float(pf[jedge])))
        L.append("\\newcommand{\\PropSpread}{%s}"
                 % fmt(float(pn[jx] / pn[ji]), 1))
    else:
        for mac in ("PropNominal", "PropNominalLeading", "PropMaxF",
                    "PropLo", "PropLoF",
                    "PropHighest", "PropShortfallPct", "PropEdge", "PropEdgeF",
                    "PropSpread"):
            L.append("\\newcommand{\\%s}{--}" % mac)
    # strongest-arm in-band error (used in the contrast subsection)
    vmax_max = max(arms)
    names = runs_for(ev, "main_fno_%s_" % arm_key(vmax_max))
    if names:
        ib, _ = inband(ev, names, vmax_max)
        L.append("\\newcommand{\\InBandWorst}{%s}" % fmt(ib))
    else:
        L.append("\\newcommand{\\InBandWorst}{--}")
    n_cache = cache_samples()
    L.append("\\newcommand{\\CacheSamples}{%s}"
             % ("{:,}".format(n_cache).replace(",", "\\,") if n_cache else "--"))
    # the M = 2 arm is the evidence for the mode-budget-limited regime
    m2 = runs_for(ev, "mode_m02_")
    if m2:
        ib2, _ = inband(ev, m2, 0.25)
        F2, M2, _ = mean_over_seeds(ev, m2, 0.25)
        # Anchor this sentence to the ACTUAL ends of the test grid.  The previous
        # version looked up f = 2.0 by name; that silently became NaN (and then a
        # printed en-dash) as soon as the grid was refined to 4.25/4.75 and the
        # integer points 2.0/3.0 stopped existing.
        lo, hi = int(np.argmin(F2)), int(np.argmax(F2))
        L.append("\\newcommand{\\ModeMtwoIn}{%s}" % fmt(ib2))
        L.append("\\newcommand{\\ModeMtwoLo}{%s}" % fmt(M2[lo]))
        L.append("\\newcommand{\\ModeMtwoLoF}{%s}" % fmt(F2[lo], 1))
        L.append("\\newcommand{\\ModeMtwoHi}{%s}" % fmt(M2[hi]))
        L.append("\\newcommand{\\ModeMtwoHiF}{%s}" % fmt(F2[hi], 1))
        # The band-edge step the M = 2 arm still has.  A referee read the arm's
        # monotone in-band rise as "the cliff is pulled in to f ~ 2" and the
        # curve does not say that: the largest single step of the arm is still at
        # the training edge.  Both the two endpoints and their ratio are emitted
        # so the sentence can be checked against the numbers instead of against
        # a shape word like "flat".
        at4 = M2[np.isclose(F2, 4.0)]
        atEdge = M2[np.isclose(F2, C.F_TRAIN[1] + 0.25)]
        if at4.size and atEdge.size:
            L.append("\\newcommand{\\ModeMtwoInF}{%s}" % fmt(float(at4[0])))
            L.append("\\newcommand{\\ModeMtwoEdge}{%s}" % fmt(float(atEdge[0])))
            L.append("\\newcommand{\\ModeMtwoEdgeRatio}{%s}"
                     % fmt(float(atEdge[0] / at4[0]), 2))
        else:
            for mac in ("ModeMtwoInF", "ModeMtwoEdge", "ModeMtwoEdgeRatio"):
                L.append("\\newcommand{\\%s}{--}" % mac)
        # Every budget below the training edge, in one list each, so Eq. (10)'s
        # min(f_train, M) branch can be read off four points rather than one.
        # f* is the table's own rule (first grid point above 6 eps(4)).
        sub_in, sub_plat, sub_fs = [], [], []
        for m in (1, 2, 3, 4):
            nm = runs_for(ev, "mode_m%02d_" % m)
            if not nm:
                continue
            ibm, _ = inband(ev, nm, 0.25)
            Fm, Mm, _ = mean_over_seeds(ev, nm, 0.25)
            sub_in.append(fmt(ibm))
            sub_plat.append(fmt(float(np.mean(Mm[Fm >= 6]))))
            fs = f_star(Fm, Mm)
            sub_fs.append(fmt(fs, 2) if np.isfinite(fs) else "--")
        for mac, vals in (("SubBandInList", sub_in),
                          ("SubBandPlatList", sub_plat),
                          ("SubBandFstarList", sub_fs)):
            L.append("\\newcommand{\\%s}{%s}"
                     % (mac, ", ".join(vals) if vals else "--"))
        L.append("\\newcommand{\\SubBandN}{%d}" % len(sub_in))
        # The three quantities that decide what Eq. (10) may say.  The referee's
        # objection was that its M < f_train branch rested on M = 2 alone and
        # that the M = 2 curve contradicts the story told around it; both halves
        # are answered from the archive now.  (i) The in-band floor falls
        # monotonically with M and saturates near M = 4.  (ii) The locator f*
        # fires at the SAME frequency for every M >= 8 and never fires below
        # that, because there the in-band level has risen to within a factor of
        # EdgeContrastLo of the plateau -- i.e. the mode budget sets the contrast
        # of the transition, not its position.
        if sub_in and len(sub_plat) == len(sub_in):
            try:
                span = float(sub_in[0]) / float(sub_in[-1])
                L.append("\\newcommand{\\SubBandSpan}{%s}" % fmt(span, 1))
            except (ValueError, ZeroDivisionError):
                L.append("\\newcommand{\\SubBandSpan}{--}")
        else:
            L.append("\\newcommand{\\SubBandSpan}{--}")
        e_ratios, e_contras, fs_ge8 = [], [], []
        for m in (1, 2, 3, 4, 8, 12, 16, 20, 24, 32):
            nm = runs_for(ev, "mode_m%02d_" % m) if m != 20 \
                else runs_for(ev, "main_fno_a025_")
            if not nm:
                continue
            Fm, Mm, _ = mean_over_seeds(ev, nm, 0.25)
            a4 = Mm[np.isclose(Fm, 4.0)]
            ae = Mm[np.isclose(Fm, C.F_TRAIN[1] + 0.25)]
            pl = float(np.mean(Mm[Fm >= 6]))
            if a4.size and ae.size and a4[0] > 0:
                e_ratios.append((m, float(ae[0] / a4[0])))
            if a4.size and a4[0] > 0 and pl > 0:
                e_contras.append((m, pl / float(a4[0])))
            fs = f_star(Fm, Mm)
            if np.isfinite(fs):
                fs_ge8.append(fs)
        lo = [c for m, c in e_contras if m <= 4]
        L.append("\\newcommand{\\EdgeContrastLo}{%s}"
                 % (fmt(min(lo), 2) if lo else "--"))
        r4 = [r for m, r in e_ratios if m == 4]
        L.append("\\newcommand{\\SubBandEdgeRatioFour}{%s}"
                 % (fmt(r4[0], 2) if r4 else "--"))
        rr = [r for m, r in e_ratios if m == 20]
        L.append("\\newcommand{\\EdgeRatioRef}{%s}"
                 % (fmt(rr[0], 2) if rr else "--"))
        # One value because the data give one value: every budget from 8 to 32
        # places the locator at the same grid point.  If a later pass ever makes
        # them differ this prints the spread instead of hiding it behind min().
        if fs_ge8 and max(fs_ge8) - min(fs_ge8) < 1e-9:
            L.append("\\newcommand{\\FstarConst}{%s}" % fmt(fs_ge8[0], 2))
            L.append("\\newcommand{\\FstarSpread}{%s}" % fmt(0.0, 2))
        elif fs_ge8:
            L.append("\\newcommand{\\FstarConst}{%s}" % fmt(np.mean(fs_ge8), 2))
            L.append("\\newcommand{\\FstarSpread}{%s}"
                     % fmt(max(fs_ge8) - min(fs_ge8), 2))
        else:
            for mac in ("FstarConst", "FstarSpread"):
                L.append("\\newcommand{\\%s}{--}" % mac)
        ref_m20 = runs_for(ev, "main_fno_a025_")
        L.append("\\newcommand{\\ModeRefIn}{%s}"
                 % (fmt(inband(ev, ref_m20, 0.25)[0]) if ref_m20 else "--"))
    else:
        for mac in ("ModeMtwoIn", "ModeMtwoLo", "ModeMtwoLoF", "ModeMtwoHi",
                    "ModeMtwoHiF", "ModeMtwoInF", "ModeMtwoEdge",
                    "ModeMtwoEdgeRatio", "SubBandInList", "SubBandPlatList",
                    "SubBandFstarList", "ModeRefIn"):
            L.append("\\newcommand{\\%s}{--}" % mac)
        L.append("\\newcommand{\\SubBandN}{0}")
    # first-order Born error in the weakest arm at the lowest test frequency
    weak = min(arms)
    ref = None
    wnames = runs_for(ev, "main_fno_%s_" % arm_key(weak))
    if wnames:
        ref = wnames[0]
    if ref:
        rows = ev[ref]["born_order1"]["%.2f" % weak]
        v = next((r["mean"] for r in rows if abs(r["f"] - 1.5) < 1e-9), np.nan)
        L.append("\\newcommand{\\BornWeakFirst}{%s}" % fmt(v))
    else:
        L.append("\\newcommand{\\BornWeakFirst}{--}")
    # ---- how flat the in-band error really is -------------------------------
    # The manuscript claims the error is "approximately independent of the
    # frequency" for f <= 4.  That is a quantitative statement about the ratio
    # max/min of the mean error over the in-band test frequencies, so it is
    # generated rather than asserted.
    drift = []
    for vmax in arms:
        r = arm_curves(ev, vmax)
        if r is None:
            continue
        Fs, Ms, _ = mean_over_seeds(ev, r[5], vmax)
        sel = Ms[Fs <= 4.0]
        if sel.size and np.nanmin(sel) > 0:
            drift.append(float(np.nanmax(sel) / np.nanmin(sel)))
    L.append("\\newcommand{\\InBandDrift}{%s}"
             % (fmt(max(drift), 1) if drift else "--"))
    # ---- the Born baseline against the learned model in the weakest arm -----
    if wnames:
        rows = ev[wnames[0]]["born_order1"]["%.2f" % weak]
        inb = float(np.mean([r["mean"] for r in rows if r["f"] <= 4.0]))
        out = float(np.mean([r["mean"] for r in rows if r["f"] >= 6.0]))
        rw = arm_curves(ev, weak)
        plat = rw[1] if rw else np.nan
        L.append("\\newcommand{\\BornWeakInBand}{%s}" % fmt(inb))
        L.append("\\newcommand{\\BornWeakOut}{%s}" % fmt(out))
        L.append("\\newcommand{\\BornWeakGain}{%s}"
                 % (fmt(plat / out, 1) if out > 0 and np.isfinite(plat) else "--"))
    else:
        for mac in ("BornWeakInBand", "BornWeakOut", "BornWeakGain"):
            L.append("\\newcommand{\\%s}{--}" % mac)
    # ---- mixed-contrast control -------------------------------------------
    # Trained on the same band [1,4] but with v_max spread over [0.05, 1.00].
    # It separates "the cliff is a property of the frequency band" from "the
    # cliff is an artefact of having trained on a single contrast".
    mx = runs_for(ev, "mixed_fno")
    if mx:
        # The mixed model has no single "own" contrast, so the plateau is
        # reported over the five evaluation arms.  The range used to be taken
        # from the first seed's sweeps only; it is now taken over every seed,
        # so the quoted interval is backed by all three seeds.
        per_seed_arms = []
        for n in mx:
            row = []
            for vm in arms:
                r = ev[n]["sweep"].get("%.2f" % vm)
                if r:
                    row.append(float(np.mean([x["mean"] for x in r if x["f"] >= 6])))
            if row:
                per_seed_arms.append(row)
        allvals = [x for row in per_seed_arms for x in row]
        L.append("\\newcommand{\\MixedPlateau}{%s}"
                 % (fmt(float(np.median(allvals))) if allvals else "--"))
        L.append("\\newcommand{\\MixedPlateauLo}{%s}"
                 % (fmt(min(allvals)) if allvals else "--"))
        L.append("\\newcommand{\\MixedPlateauHi}{%s}"
                 % (fmt(max(allvals)) if allvals else "--"))
        # Band-edge ratio.  The house estimator (Section on the collapse) is
        # per-seed measurement then averaging, NOT composition of the two
        # published seed-mean errors; the earlier macro used the pooled
        # reading and so disagreed with Table "mixed" by more than the seed
        # spread.  Report the per-seed mean with its seed SD as the primary
        # quantity and the pooled reading alongside, so a referee recomputing
        # either one finds it in the text.
        ratios = []
        for n in mx:
            F1, M1, _ = mean_over_seeds(ev, [n], 0.25)
            e5 = M1[np.isclose(F1, 5.0)]
            e4 = M1[np.isclose(F1, 4.0)]
            if e4.size and e4[0] > 0 and e5.size:
                ratios.append(float(e5[0] / e4[0]))
        Fm, Mm, _ = mean_over_seeds(ev, mx, 0.25)
        e5 = Mm[np.isclose(Fm, 5.0)]
        e4 = Mm[np.isclose(Fm, 4.0)]
        pooled = float(e5[0] / e4[0]) if e4.size and e4[0] > 0 else np.nan
        L.append("\\newcommand{\\MixedRatio}{%s}"
                 % (fmt(float(np.mean(ratios)), 2) if ratios else "--"))
        L.append("\\newcommand{\\MixedRatioSD}{%s}"
                 % (fmt(float(np.std(ratios)), 2) if ratios else "--"))
        L.append("\\newcommand{\\MixedRatioPooled}{%s}"
                 % (fmt(pooled, 2) if np.isfinite(pooled) else "--"))
    else:
        for mac in ("MixedPlateau", "MixedPlateauLo", "MixedPlateauHi", "MixedRatio"):
            L.append("\\newcommand{\\%s}{--}" % mac)
    # ---- orders of magnitude by which the Born series misses at the top -----
    b5 = []
    for vmax in arms:
        n = runs_for(ev, "main_fno_%s_" % arm_key(vmax))
        if not n:
            continue
        v5 = np.array([r["mean"] for r in ev[n[0]]["born_order5"]["%.2f" % vmax]])
        b5.append(float(np.nanmax(v5)))
    order = int(np.floor(np.log10(max(b5)))) if b5 and max(b5) >= 10 else 0
    L.append("\\newcommand{\\BornDivOrder}{%d}" % order)
    # ---- arm-matched vs cross-contrast reading of the cliff ratio -----------
    # Each eval JSON carries the weak-contrast model evaluated at every contrast,
    # which is what the confound disclosure in Sec. "cliff" quantifies.
    top = max(arms)
    if wnames:
        rows = ev[wnames[0]]["sweep"]["%.2f" % top]
        Fx = np.array([r["f"] for r in rows])
        Mx = np.array([r["mean"] for r in rows])
        e4x = float(Mx[np.isclose(Fx, 4.0)][0])
        e5x = float(Mx[np.isclose(Fx, 5.0)][0])
        ibx = float(np.mean([r["mean"] for r in ev[wnames[0]]["in_band"]
                             if abs(r["vmax"] - top) < 1e-6]))
        L.append("\\newcommand{\\CrossInBand}{%s}" % fmt(ibx))
        platx = float(Mx[Fx >= 6].mean())
        L.append("\\newcommand{\\CrossPlateau}{%s}" % fmt(platx))
        # How far the weak arm's in-band level sits below its own plateau, as a
        # percentage. The prose must not round this up to "already at its
        # plateau": at 73% the honest statement is "roughly three quarters".
        L.append("\\newcommand{\\CrossInBandFrac}{%s}"
                 % ("%.0f" % (100.0 * ibx / platx) if platx > 0 else "--"))
        L.append("\\newcommand{\\CrossRatio}{%s}"
                 % (fmt(e5x / e4x, 2) if e4x > 0 else "--"))
        rw0 = arm_curves(ev, min(arms))
        L.append("\\newcommand{\\CrossOwnRatio}{%s}"
                 % (fmt(rw0[4], 2) if rw0 and np.isfinite(rw0[4]) else "--"))
        # The converse direction, computed in the same block so the symmetry
        # sentence in the body cannot be supported by one direction in one
        # manuscript and contradicted in the other.
        snames = runs_for(ev, "main_fno_%s_" % arm_key(top))
        if snames:
            srow = ev[snames[0]]["sweep"]["%.2f" % weak]
            Fs = np.array([r["f"] for r in srow])
            Ms = np.array([r["mean"] for r in srow])
            e4s = float(Ms[np.isclose(Fs, 4.0)][0])
            e5s = float(Ms[np.isclose(Fs, 5.0)][0])
            ibs = float(np.mean([r["mean"] for r in ev[snames[0]]["in_band"]
                                 if abs(r["vmax"] - weak) < 1e-6]))
            plats = float(Ms[Fs >= 6].mean())
            own = ev[snames[0]]["sweep"]["%.2f" % top]
            F3 = np.array([r["f"] for r in own])
            M3 = np.array([r["mean"] for r in own])
            L.append("\\newcommand{\\CrossStrongMismatch}{%s}"
                     % (fmt(e5s / e4s, 2) if e4s > 0 else "--"))
            L.append("\\newcommand{\\CrossStrongFrac}{%s}"
                     % ("%.0f" % (100.0 * ibs / plats) if plats > 0 else "--"))
            L.append("\\newcommand{\\CrossStrongOwn}{%s}"
                     % fmt(float(M3[np.isclose(F3, 5.0)][0])
                           / float(M3[np.isclose(F3, 4.0)][0]), 2))
        else:
            for mac in ("CrossStrongMismatch", "CrossStrongFrac",
                        "CrossStrongOwn"):
                L.append("\\newcommand{\\%s}{--}" % mac)
    else:
        for mac in ("CrossInBand", "CrossPlateau", "CrossRatio", "CrossOwnRatio",
                    "CrossInBandFrac", "CrossStrongMismatch", "CrossStrongFrac",
                    "CrossStrongOwn"):
            L.append("\\newcommand{\\%s}{--}" % mac)
    # ---- capacity ablation: same recipe, 2.6x the parameters ---------------
    cap = runs_for(ev, "cap_w64_a025_")
    ref025 = runs_for(ev, "main_fno_a025_")
    # The reference parameter count is reported even while the width-64 run is
    # still training, so the sentence never leans on a stale hard-coded number.
    L.append("\\newcommand{\\BaseParams}{%s}"
             % ("%.1f" % (ev[ref025[0]]["params"] / 1e6) if ref025 else "--"))
    if cap and ref025:
        Fc, Mc, _ = mean_over_seeds(ev, cap, 0.25)
        Fc2, Mr, _ = mean_over_seeds(ev, ref025, 0.25)
        pc = float(Mc[Fc >= 6].mean())
        pr = float(Mr[Fc2 >= 6].mean())
        L.append("\\newcommand{\\CapParams}{%.1f}" % (ev[cap[0]]["params"] / 1e6))
        L.append("\\newcommand{\\CapFactor}{%.1f}"
                 % (ev[cap[0]]["params"] / max(ev[ref025[0]]["params"], 1)))
        L.append("\\newcommand{\\CapPlateau}{%s}" % fmt(pc))
        L.append("\\newcommand{\\PlateauRef}{%s}" % fmt(pr))
        # The in-band column of the same capacity ablation.  It is needed in
        # full: the paragraph that says "a bigger model does not help" is read
        # against the data-volume ladder, which *does* lower the in-band floor,
        # so leaving this number out of the macros would let the two levers be
        # quoted as if they behaved alike.
        ibc, _ = inband(ev, cap, 0.25)
        ibr, _ = inband(ev, ref025, 0.25)
        L.append("\\newcommand{\\CapInBand}{%s}" % fmt(ibc))
        L.append("\\newcommand{\\CapInBandDelta}{%s}"
                 % fmt(100.0 * abs(ibc - ibr) / max(ibr, 1e-30), 1))
    else:
        for mac in ("CapParams", "CapFactor", "CapPlateau", "PlateauRef",
                    "CapInBand", "CapInBandDelta"):
            L.append("\\newcommand{\\%s}{--}" % mac)
    # Parameter span of the mode-budget sweep.  This is the second, independent
    # capacity axis: it runs over two orders of magnitude and, unlike the
    # width ablation, it *does* move the in-band error -- which is exactly the
    # distinction the negative-results paragraph has to keep straight.
    modes = runs_for(ev, "mode_m")
    if modes:
        ps = [ev[n]["params"] / 1e6 for n in modes]
        L.append("\\newcommand{\\ModeSpanLo}{%.2f}" % min(ps))
        L.append("\\newcommand{\\ModeSpanHi}{%.1f}" % max(ps))
        L.append("\\newcommand{\\ModeSpanFactor}{%.0f}" % (max(ps) / min(ps)))
    else:
        for mac in ("ModeSpanLo", "ModeSpanHi", "ModeSpanFactor"):
            L.append("\\newcommand{\\%s}{--}" % mac)
    L += solver_macros()
    L += realgeo_macros()
    L += cost_macros()
    L += precond_macros()
    L += residual_macros()
    L += paired_macros()
    L += dz_macros()
    L += ndata_macros(ev)
    L += uband_macros(ev)
    L += seed_macros(ev)
    L += sd_bound_macros(ev, arms)
    L += c1_macros()
    L += disc_macros()
    L += anchor_macros()
    L += deep_macros()
    L += grid_macros()
    L += loc_macros()
    return "\n".join(L) + "\n"


# The paired rows the prose quotes by value.  Naming them here is not
# cosmetic: these numbers were hand-typed in both manuscripts, so when the
# resampling unit was corrected on 2026-10-12 they would have gone stale in
# the prose while the table updated -- the exact drift the generated-macro
# discipline exists to prevent.
PAIRED_PROSE_ROWS = {
    "PairUnetBaseSoft": "unet_base_vs_fno_a025",
    "PairUnetPmatchSoft": "unet_pmatch_vs_fno_a025",
    "PairUnetBaseStrong": "unet_base_vs_fno_a100",
    "PairWidthPlateau": "capw64_vs_fno_a025",
}

# Rows whose 95% interval STRADDLES ZERO.  The magnitude form above sorts the
# interval endpoints by absolute value and quotes the difference as a positive
# "lower by", which is right only when the interval excludes zero: applied to a
# tie it prints an interval whose printed endpoints are both away from zero and
# therefore reads as a significant improvement.  Tie rows are emitted with their
# sign kept, at the same precision the table uses, plus the permutation p-value,
# and the guard below withholds the macros entirely if the row stops straddling
# zero -- the prose sentence that relies on that straddle would otherwise become
# silently false.
PAIRED_TIE_ROWS = {
    "PairWidthIn": "capw64_vs_fno_a025_in",
}

# Rows the prose quotes with their sign: the band-edge jumps, which are all
# positive and are compared with one another, so a magnitude would lose the
# direction the sentence asserts.
PAIRED_SIGNED_ROWS = {
    "PairJumpFnoSoft": "jump_fno_a025",
    "PairJumpFnoStrong": "jump_fno_a100",
    "PairJumpUnetBase": "jump_unet_base_a025",
    "PairJumpUnetPmatch": "jump_unet_pmatch_a025",
    "PairJumpDon": "jump_don_a025",
}


def paired_macros():
    """Values and 95% intervals for the paired rows quoted in the prose.

    Magnitude rows are emitted as magnitudes with a magnitude-sorted interval,
    because the prose says "lower by X [a, b]" whatever the sign of the
    difference; emitting the signed value would force hand arithmetic at every
    use.  Signed rows keep their sign, since the prose compares their directions.
    """
    keys = []
    for mac in PAIRED_PROSE_ROWS:
        keys += [mac, mac + "Lo", mac + "Hi"]
    for mac in PAIRED_TIE_ROWS:
        keys += [mac + "Delta", mac + "CILo", mac + "CIHi", mac + "P"]
    keys += list(PAIRED_SIGNED_ROWS)
    p = os.path.join(C.RES, "paired.json")
    if not os.path.exists(p):
        return ["\\newcommand{\\%s}{--}" % k for k in sorted(keys)]
    d = json.load(open(p, encoding="utf-8"))
    by = {c["label"]: c for c in d["cases"]}
    L = []
    for mac, label in sorted(PAIRED_PROSE_ROWS.items()):
        c = by.get(label)
        if c is None:
            L += ["\\newcommand{\\%s}{--}" % k
                  for k in (mac, mac + "Lo", mac + "Hi")]
            continue
        lo, hi = sorted((abs(c["lo"]), abs(c["hi"])))
        L.append("\\newcommand{\\%s}{%s}" % (mac, fmt(abs(c["diff"]))))
        L.append("\\newcommand{\\%sLo}{%s}" % (mac, fmt(lo)))
        L.append("\\newcommand{\\%sHi}{%s}" % (mac, fmt(hi)))
    for mac, label in sorted(PAIRED_TIE_ROWS.items()):
        c = by.get(label)
        straddles = c is not None and c["lo"] <= 0.0 <= c["hi"]
        if not straddles:
            L += ["\\newcommand{\\%s}{--}" % (mac + sfx)
                  for sfx in ("Delta", "CILo", "CIHi", "P")]
            continue
        # `%+.3f` in the table prints the positive end as "+0.000"; keeping the
        # prose on the same three decimals makes the two literally the same
        # string, so neither can drift from the other unnoticed.
        L.append("\\newcommand{\\%sDelta}{%s}" % (mac, fmt(c["diff"])))
        L.append("\\newcommand{\\%sCILo}{%s}" % (mac, fmt(c["lo"])))
        L.append("\\newcommand{\\%sCIHi}{%s}"
                 % (mac, ("+" if c["hi"] >= 0 else "") + fmt(c["hi"])))
        L.append("\\newcommand{\\%sP}{%s}" % (mac, fmt(c["p"])))
    for mac, label in sorted(PAIRED_SIGNED_ROWS.items()):
        c = by.get(label)
        L.append("\\newcommand{\\%s}{%s}"
                 % (mac, fmt(c["diff"]) if c else "--"))
    return L


def sd_bound_macros(ev, arms):
    """One measured seed-SD bound per table that cannot carry a second number.

    The reviewer's request is that a reader be able to tell whether a table's
    entries are stable across seeds.  For the 15-column main table and the
    architecture/band/mode/data tables, printing mean +- SD in every cell is not
    an option without destroying the layout, and leaving the numbers bare is
    what the objection was about.  The measured maximum over exactly the cells
    printed is the smallest honest substitute: it is a number from the archive,
    it is checkable, and if a cell ever becomes unstable the bound says so.
    """
    arm_prefs = ["main_fno_%s_" % arm_key(v) for v in arms]
    arch_prefs = ("main_fno_a025_", "base_unet_a025_", "pmatch_unet_a025_",
                  "base_don_a025_", "main_fno_a100_", "base_unet_a100_",
                  "pmatch_unet_a100_", "base_don_a100_")
    band_prefs = ("main_fno_a025_", "band_b2_", "band_b3_", "band_b6_",
                  "band_b12_", "base_unet_a025_", "uband_b6_")
    mode_prefs = tuple("mode_m%02d_" % m
                       for m in (1, 2, 3, 4, 8, 12, 24, 32)) + ("main_fno_a025_",)
    data_prefs = tuple(p for p, _ in NDATA_RUNGS)

    L = []
    for mac, prefs, vlist, use_ib in (
            ("MainSDBound", tuple(arm_prefs), arms, False),
            ("ArchSDBound", arch_prefs, (0.25, 1.00), False),
            ("BandSDBound", band_prefs, (0.25,), True),
            ("ModeSDBound", mode_prefs, (0.25,), True),
            ("NDataSDBound", data_prefs, (0.25,), True)):
        v = sd_bound(ev, prefs, vlist, use_inband=use_ib)
        L.append("\\newcommand{\\%s}{%s}" % (mac, fmt(v)))
    return L


RES_DIR = None          # set by --res; falls back to C.RES inside load()


# ---------------------------------------------------------------------------
# complete arm inventory (appendix)
# ---------------------------------------------------------------------------
# One row per trained arm.  The training-protocol section states that every
# ablation and every baseline was trained from the same seeds; this table is
# what makes that statement checkable by a reader, listing all of them with
# the three numbers a referee would recompute first.  The configuration
# labels are structural properties of the run (architecture, band, budget);
# every numeric cell is read from the eval records, never hardcoded.
def _arm_spec():
    spec = [
        ("main_fno_a005", "FNO, $v_{\\max}=0.05$", 0.05),
        ("main_fno_a010", "FNO, $v_{\\max}=0.10$", 0.10),
        ("main_fno_a025", "FNO, $v_{\\max}=0.25$ (reference)", 0.25),
        ("main_fno_a050", "FNO, $v_{\\max}=0.50$", 0.50),
        ("main_fno_a100", "FNO, $v_{\\max}=1.00$", 1.00),
        ("base_unet_a025", "U-Net (base), $v_{\\max}=0.25$", 0.25),
        ("base_unet_a100", "U-Net (base), $v_{\\max}=1.00$", 1.00),
        ("pmatch_unet_a025", "U-Net (matched), $v_{\\max}=0.25$", 0.25),
        ("pmatch_unet_a100", "U-Net (matched), $v_{\\max}=1.00$", 1.00),
        ("base_don_a025", "DeepONet, $v_{\\max}=0.25$", 0.25),
        ("base_don_a100", "DeepONet, $v_{\\max}=1.00$", 1.00),
        ("cap_w64_a025", "FNO width 64, $v_{\\max}=0.25$", 0.25),
        ("gamma_g02", "FNO, $\\gamma=0.02$, $v_{\\max}=0.25$", 0.25),
        ("gamma_g10", "FNO, $\\gamma=0.10$, $v_{\\max}=0.25$", 0.25),
        ("mixed_fno", "FNO, mixed contrast, eval $v_{\\max}=0.25$", 0.25),
    ]
    for m in (1, 2, 3, 4, 8, 12, 24, 32):
        spec.append(("mode_m%02d" % m, "FNO, $M=%d$, $v_{\\max}=0.25$" % m, 0.25))
    for n in (1536, 3072, 6144):
        spec.append(("ndata_n%06d" % n,
                     "FNO, $n_{\\mathrm{train}}=%d$, $v_{\\max}=0.25$" % n, 0.25))
    for b in (2, 3, 6, 12):
        spec.append(("band_b%d" % b,
                     "FNO, training band $[1,%d]$, $v_{\\max}=0.25$" % b, 0.25))
    spec.append(("uband_b6", "U-Net (base), training band $[1,6]$", 0.25))
    return spec


def table_arms(ev, cap):
    rows = []
    for prefix, label, vmax in _arm_spec():
        names = runs_for(ev, prefix)
        if not names:
            continue
        F, M, _ = mean_over_seeds(ev, names, vmax)
        e4 = float(M[np.isclose(F, 4.0)][0]) if np.isclose(F, 4.0).any() else np.nan
        e5 = float(M[np.isclose(F, 5.0)][0]) if np.isclose(F, 5.0).any() else np.nan
        plat = float(np.mean(M[F >= 6.0])) if (F >= 6.0).any() else np.nan
        params = ev[names[0]].get("params")
        rows.append((prefix, label, params, e4, e5, plat))
    t = ("\\begin{table}[ht]\n\\centering\n\\caption{%s}\n\\label{tab:arms}\n"
         "\\footnotesize\n\\setlength{\\tabcolsep}{3.6pt}\n"
         "\\begin{tabular}{llrrrr}\n\\toprule\n"
         "arm & configuration & params (M) & $\\varepsilon(4)$ & "
         "$\\varepsilon(5)$ & $\\varepsilon(f\\ge6)$ \\\\\n\\midrule\n"
         % cap["arms"])
    for (prefix, label, params, e4, e5, plat) in rows:
        t += ("\\texttt{%s} & %s & %s & $%s$ & $%s$ & $%s$ \\\\\n"
              % (prefix.replace("_", "\\_"), label,
                 ("%.1f" % (params / 1e6)) if params else "--",
                 fmt(e4), fmt(e5), fmt(plat)))
    return t + "\\bottomrule\n\\end{tabular}\n\\end{table}\n"


def c1_load():
    """results/c1_verdict.json, or None.  Written by code/analyze_c1.py."""
    p = os.path.join(C.RES, "c1_verdict.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def _verdict_load(name):
    p = os.path.join(C.RES, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def disc_macros():
    """Prose macros for the representation / loss / transfer discriminators.

    Every value is read from results/c2_verdict.json, c3_verdict.json and
    c4_verdict.json, which analyze_c2.py, analyze_c3.py and analyze_c4.py
    write from the evaluation records (and, for the energy probes, from fresh
    forward passes over the checkpoints).  Nothing here is transcribed.
    """
    d2 = _verdict_load("c2_verdict.json")
    d3 = _verdict_load("c3_verdict.json")
    d4 = _verdict_load("c4_verdict.json")
    keys = ["DiscFFIn", "DiscFFInRatio", "DiscFFEdge", "DiscFFEdgeRatio",
            "DiscFFJump", "DiscFFRatio", "DiscFFPlat", "DiscFFOnsetLo",
            "DiscFFOnsetHi", "DiscFFKLo", "DiscFFKHi",
            "DiscEnIn", "DiscEnOnsetLo", "DiscEnOnsetHi", "DiscEnKLo",
            "DiscEnKHi",
            "DiscObOut", "DiscObPlat", "DiscObRatioHi", "DiscObAlignHi",
            "DiscBaseRatioHi", "DiscBaseAlignHi",
            "DiscTlIn", "DiscTlInRatio", "DiscTlWarmIn", "DiscTlWarmInRatio",
            "DiscTlWarmOut", "DiscTlWarmOutRatio", "DiscTlWarmPlat",
            "DiscTlWarmPlatRatio", "DiscTlOnsetLo", "DiscTlOnsetHi",
            "DiscTlKLo", "DiscTlKHi"]
    dash = ["\\newcommand{\\%s}{--}" % k for k in keys]
    if d2 is None or d3 is None or d4 is None:
        return dash
    ff, base2 = d2["arms"]["fourier"], d2["arms"]["baseline"]
    en, ob, base3 = (d3["arms"]["energy"], d3["arms"]["energy_ob"],
                     d3["arms"]["baseline"])
    tl, tlw, base4 = d4["arms"]["tl"], d4["arms"]["tl_warm"], d4["arms"]["baseline"]
    # out-of-band probe extremes, f = 12 grid point ("12.00" key)
    vals = {
        "DiscFFIn": fmt(ff["in_band"]),
        "DiscFFInRatio": fmt(d2["verdict"]["in_band_ratio"], 1),
        "DiscFFEdge": fmt(ff["e4"]),
        "DiscFFEdgeRatio": fmt(base2["e4"] / ff["e4"], 1),
        "DiscFFJump": fmt(ff["jump"], 1),
        "DiscFFRatio": fmt(ff["ratio"], 1),
        "DiscFFPlat": fmt(ff["plateau"]),
        "DiscFFOnsetLo": freq_label(ff["onset"][0]),
        "DiscFFOnsetHi": freq_label(ff["onset"][1]),
        "DiscFFKLo": fmt(d2["verdict"]["onset_k_lo"], 1),
        "DiscFFKHi": fmt(d2["verdict"]["onset_k_hi"], 1),
        "DiscEnIn": fmt(en["in_band"]),
        "DiscEnOnsetLo": freq_label(en["onset"][0]),
        "DiscEnOnsetHi": freq_label(en["onset"][1]),
        "DiscEnKLo": fmt(d3["verdict"]["onset_k_lo"], 1),
        "DiscEnKHi": fmt(d3["verdict"]["onset_k_hi"], 1),
        "DiscObOut": fmt(ob["e5"]),
        "DiscObPlat": fmt(ob["plateau"]),
        "DiscObRatioHi": fmt(ob["energy"]["12.00"]["energy_ratio"], 2),
        "DiscObAlignHi": sci(ob["energy"]["12.00"]["alignment"], 1),
        "DiscBaseRatioHi": fmt(base3["energy"]["12.00"]["energy_ratio"], 2),
        "DiscBaseAlignHi": sci(base3["energy"]["12.00"]["alignment"], 1),
        "DiscTlIn": fmt(tl["in_band"]),
        "DiscTlInRatio": fmt(d4["verdict"]["in_band_ratio_tl"], 1),
        "DiscTlWarmIn": fmt(tlw["in_band"]),
        "DiscTlWarmInRatio": fmt(d4["verdict"]["in_band_ratio_tl_warm"], 2),
        "DiscTlWarmOut": fmt(tlw["e5"]),
        "DiscTlWarmOutRatio": fmt(d4["verdict"]["e5_ratio_tl_warm"], 2),
        "DiscTlWarmPlat": fmt(tlw["plateau"]),
        "DiscTlWarmPlatRatio": fmt(d4["verdict"]["plateau_ratio_tl_warm"], 2),
        "DiscTlOnsetLo": freq_label(tlw["onset"][0]),
        "DiscTlOnsetHi": freq_label(tlw["onset"][1]),
        "DiscTlKLo": fmt(d4["verdict"]["onset_k_lo"], 1),
        "DiscTlKHi": fmt(d4["verdict"]["onset_k_hi"], 1),
    }
    return ["\\newcommand{\\%s}{%s}" % (k, vals[k]) for k in keys]


def table_disc(cap):
    """The representation / loss / transfer discriminator arms (Section sec:disc).

    Rows are read from results/c2_verdict.json, c3_verdict.json and
    c4_verdict.json; nothing is transcribed.  The baseline row is repeated
    from those files (not from tab:c1) so the table stands alone.
    """
    d2 = _verdict_load("c2_verdict.json")
    d3 = _verdict_load("c3_verdict.json")
    d4 = _verdict_load("c4_verdict.json")
    if d2 is None or d3 is None or d4 is None:
        return ""
    rows = [
        ("baseline (from scratch)", d2["arms"]["baseline"]),
        ("Fourier-feature input (C2)", d2["arms"]["fourier"]),
        ("+ in-band energy penalty (C3)", d3["arms"]["energy"]),
        ("+ out-of-band Born anchor (C3b)", d3["arms"]["energy_ob"]),
        ("TL fine-tune, $10^{-4}$ (C4)", d4["arms"]["tl"]),
        ("TL warm start, $10^{-3}$ (C4b)", d4["arms"]["tl_warm"]),
    ]
    t = ("\\begin{table}[ht]\n\\centering\n\\caption{%s}\n\\label{tab:disc}\n"
         "\\footnotesize\n\\setlength{\\tabcolsep}{3.6pt}\n"
         "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
         "arm & $\\varepsilon_{\\mathrm{in}}$ & $\\varepsilon(4)$ & "
         "$\\varepsilon(5)$ & $r$ & $\\varepsilon_{\\infty}$ & onset \\\\\n"
         "\\midrule\n" % cap["disc"])
    for label, arm in rows:
        t += ("%s & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ & $[%s,%s]$ \\\\\n"
              % (label, fmt(arm["in_band"]), fmt(arm["e4"]), fmt(arm["e5"]),
                 fmt(arm["ratio"], 2), fmt(arm["plateau"]),
                 freq_label(arm["onset"][0]), freq_label(arm["onset"][1])))
    return t + "\\bottomrule\n\\end{tabular}\n\\end{table}\n"


def c1_macros():
    """Prose macros for the conditioning-confound discriminating experiment.

    Every value is read from results/c1_verdict.json, which analyze_c1.py
    writes from the six evaluation records plus a recomputation of the
    attained norms through gamma_comp.  The attained facts deserve the
    care: an earlier revision of the analysis hard-coded "3.44x (2.91 ..
    10.03)" as the BASELINE's spread, but 2.91 is the compensated target,
    not a baseline value -- the baseline spans 2.2x on this grid, and 3.44
    is the ratio of the baseline's band-edge norm to the target.  The three
    numbers are now computed rather than remembered.
    """
    d = c1_load()
    keys = ["CondAttLo", "CondAttHi", "CondAttSpread", "CondAttConst",
            "CondAttLevelRatio", "CondAttConstDev", "CondGammaLo", "CondGammaHi",
            "CondBaseIn", "CondCompIn", "CondBaseAtEdge", "CondCompAtEdge", "CondBaseAtOnset",
            "CondCompAtOnset", "CondBasePlat", "CondCompPlat", "CondBaseRatio",
            "CondCompRatio", "CondBaseJump", "CondCompJump", "CondOnsetLo",
            "CondOnsetHi", "CondOnsetKLo", "CondOnsetKHi", "CondInBandRatio",
            "CondPlatRatio"]
    if d is None:
        return ["\\newcommand{\\%s}{--}" % k for k in keys]
    v, a, g = d["verdict"], d["arms"]["baseline"], d["arms"]["compensated"]
    vals = {
        "CondAttLo": fmt(v["attained_baseline_lo"], 2),
        "CondAttHi": fmt(v["attained_baseline_hi"], 2),
        "CondAttSpread": fmt(v["attained_baseline_spread"], 1),
        "CondAttConst": fmt(v["attained_compensated"], 2),
        "CondAttLevelRatio": fmt(v["attained_level_ratio"], 1),
        "CondAttConstDev": sci(v["attained_compensated_max_dev"], 1),
        "CondGammaLo": fmt(v["gamma_test_lo"], 2),
        "CondGammaHi": fmt(v["gamma_test_hi"], 2),
        "CondBaseIn": fmt(a["in_band"]), "CondCompIn": fmt(g["in_band"]),
        "CondBaseAtEdge": fmt(a["e4"]), "CondCompAtEdge": fmt(g["e4"]),
        "CondBaseAtOnset": fmt(a["e425"]), "CondCompAtOnset": fmt(g["e425"]),
        "CondBasePlat": fmt(a["plateau"]), "CondCompPlat": fmt(g["plateau"]),
        "CondBaseRatio": fmt(a["ratio"], 2), "CondCompRatio": fmt(g["ratio"], 2),
        "CondBaseJump": fmt(a["jump"], 2), "CondCompJump": fmt(g["jump"], 2),
        "CondOnsetLo": freq_label(a["onset"][0]),
        "CondOnsetHi": freq_label(a["onset"][1]),
        "CondOnsetKLo": fmt(v["onset_k_lo"], 1),
        "CondOnsetKHi": fmt(v["onset_k_hi"], 1),
        "CondInBandRatio": fmt(v["in_band_ratio"], 1),
        "CondPlatRatio": fmt(v["plateau_ratio"], 1),
    }
    return ["\\newcommand{\\%s}{%s}" % (k, vals[k]) for k in keys]


def table_c1(cap):
    """The conditioning-confound discriminating experiment (Section sec:c1).

    Both rows are read from results/c1_verdict.json; nothing is transcribed.
    The attained-norm column is the manipulated variable of the experiment and
    is recomputed by analyze_c1.py through the same instrument that built the
    compensation table, so the conditioning claim and the construction cannot
    drift apart.  The onset column is a grid interval, not a point: the test
    grid resolves the band edge to 0.25, and the onset rule (first interval
    whose seed-mean error exceeds 3x its own in-band level) is stated in the
    text that introduces the table.
    """
    d = c1_load()
    if d is None:
        return ""
    v, a, g = d["verdict"], d["arms"]["baseline"], d["arms"]["compensated"]
    rows = [("baseline, $\\gamma=0.05$",
             "$%s$--$%s$" % (fmt(v["attained_baseline_lo"], 2),
                             fmt(v["attained_baseline_hi"], 2)), a),
            ("compensated, $\\gamma(f)$",
             "$%s$ (const.)" % fmt(v["attained_compensated"], 2), g)]
    t = ("\\begin{table}[ht]\n\\centering\n\\caption{%s}\n\\label{tab:c1}\n"
         "\\footnotesize\n\\setlength{\\tabcolsep}{3.6pt}\n"
         "\\begin{tabular}{llrrrrrr}\n\\toprule\n"
         "arm & attained norm & $\\varepsilon_{\\mathrm{in}}$ & "
         "$\\varepsilon(4)$ & $\\varepsilon(4.25)$ & "
         "$\\varepsilon_{\\infty}$ & $r$ & onset \\\\\n\\midrule\n"
         % cap["c1"])
    for label, att, arm in rows:
        t += ("%s & %s & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ & $[%s,%s]$ \\\\\n"
              % (label, att, fmt(arm["in_band"]), fmt(arm["e4"]),
                 fmt(arm["e425"]), fmt(arm["plateau"]), fmt(arm["ratio"], 2),
                 freq_label(arm["onset"][0]), freq_label(arm["onset"][1])))
    return t + "\\bottomrule\n\\end{tabular}\n\\end{table}\n"


def anchor_load():
    """results/anchor_baseline.json, plus the trained arm it is read against.

    The anchor's numbers mean nothing on their own; what makes them a result is
    the comparison with the trained operator on the SAME media and seeds.  Both
    halves are loaded here from their own instrument (code/anchor_baseline.py
    for the anchor, the evaluation records for the trained arm), and the guard
    in check_claims recomputes the comparison independently.
    """
    p = os.path.join(C.RES, "anchor_baseline.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p, encoding="utf-8"))
    ev = load(RES_DIR)
    ref = arm_curves(ev, 0.25) if ev else None
    if ref is not None:
        F, plateau, e4, e5, ratio, names = ref
        ib, _ = inband(ev, names, 0.25)
        d["ref"] = dict(in_band=float(ib), plateau=float(plateau),
                        e4=float(e4), e5=float(e5), ratio=float(ratio))
    return d


def anchor_macros():
    """Prose macros for the analytic anchor (Section sec:anchor).

    Everything is read from results/anchor_baseline.json and from the trained
    arm's own evaluation records.  The one number that carries the argument is
    not the anchor's error level but its FLATNESS: the ratio of its out-of-band
    error to its in-band error is near one, where the trained operator's is
    \CliffLo--\CliffHi.  Both ratios are computed here rather than quoted.
    """
    keys = ["AnchIn", "AnchE", "AnchEdge", "AnchPlat", "AnchRatio",
            "AnchTrivPlat", "AnchTrivRatio", "AnchInOut", "AnchFnoIn",
            "AnchFnoPlat", "AnchFnoRatio", "AnchWinIn", "AnchWinOut",
            "AnchA", "AnchB", "AnchNFit",
            "AnchRatioLo", "AnchRatioHi", "AnchFnoRatioLo", "AnchFnoRatioHi",
            "AnchPlatWeak", "AnchPlatStrong"]
    d = anchor_load()
    if d is None:
        return ["\\newcommand{\\%s}{--}" % k for k in keys]
    ev = load(RES_DIR)
    a = d["arms"]["0.25"]
    v = dict(
        AnchIn=fmt(a["in_band"]),
        AnchE=fmt(a["e4"]),
        AnchEdge=fmt(a["e5"]),
        AnchPlat=fmt(a["plateau"]),
        AnchRatio=fmt(a["e5"] / a["e4"], 2),
        AnchTrivPlat=fmt(a["trivial_plateau"]),
        AnchTrivRatio=fmt(a["trivial_plateau"] / a["plateau"], 1),
        AnchInOut=fmt(a["in_band"] / a["plateau"], 2),
        AnchNFit=str(int(d["n_fit"])),
        AnchA=("$%.3f%+0.3f i$" % (a["a"][0], a["a"][1])),
        AnchB=("$%.3f%+0.3f i$" % (a["b"][0], a["b"][1])),
    )
    r = d.get("ref")
    if r is not None:
        v.update(AnchFnoIn=fmt(r["in_band"]), AnchFnoPlat=fmt(r["plateau"]),
                 AnchFnoRatio=fmt(r["ratio"], 2),
                 # win factors are stated from the winner's side: the trained
                 # operator wins IN band, the anchor wins OUT of band
                 AnchWinIn=fmt(a["in_band"] / r["in_band"], 1),
                 AnchWinOut=fmt(r["plateau"] / a["plateau"], 1))
    # across-contrast facts: the anchor's flatness and the trained operator's
    # step, both as ranges so the prose cannot quote a cherry-picked contrast
    rr, fr, plats = [], [], []
    for vmax in C.VMAX_LIST:
        arm = d["arms"]["%.2f" % vmax]
        rr.append(arm["e5"] / arm["e4"])
        plats.append((vmax, arm["plateau"]))
        if ev:
            q = arm_curves(ev, vmax)
            if q is not None:
                fr.append(q[4])
    if rr:
        v["AnchRatioLo"] = fmt(min(rr), 2)
        v["AnchRatioHi"] = fmt(max(rr), 2)
    if fr:
        v["AnchFnoRatioLo"] = fmt(min(fr), 2)
        v["AnchFnoRatioHi"] = fmt(max(fr), 2)
    if plats:
        v["AnchPlatWeak"] = fmt(min(plats, key=lambda t: t[0])[1])
        v["AnchPlatStrong"] = fmt(max(plats, key=lambda t: t[0])[1])
    return ["\\newcommand{\\%s}{%s}" % (k, v.get(k, "--")) for k in keys]


def table_anchor(cap):
    """The analytic anchor against the trained operator (Section sec:anchor).

    Two rows per contrast: the anchor and, on the same media and seeds, the
    trained operator it is read against.  The anchor's coefficients come from a
    prefix of the training cache, the trained operator's row comes from the
    evaluation records, and neither is transcribed by hand.  The last two
    columns are the whole point: the trained operator steps by an order of
    magnitude at the band edge, the anchor does not step at all.
    """
    d = anchor_load()
    if d is None:
        return ""
    ev = load(RES_DIR)
    t = ("\\begin{table}[ht]\n\\centering\n\\caption{%s}\n\\label{tab:anchor}\n"
         "\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
         "\\begin{tabular}{llrrrrr}\n\\toprule\n"
         "$v_{\\max}$ & predictor & $\\varepsilon_{\\mathrm{in}}$ & "
         "$\\varepsilon(4)$ & $\\varepsilon(5)$ & $\\varepsilon_{\\infty}$ & "
         "$r=\\varepsilon(5)/\\varepsilon(4)$ \\\\\n\\midrule\n" % cap["anchor"])
    for vmax in C.VMAX_LIST:
        a = d["arms"]["%.2f" % vmax]
        ref = None
        if ev:
            rr = arm_curves(ev, vmax)
            if rr is not None:
                F, plateau, e4, e5, ratio, names = rr
                ib, _ = inband(ev, names, vmax)
                ref = (ib, e4, e5, plateau, ratio)
        t += ("$%.2f$ & anchor, $aB_1+bB_2$ & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ \\\\\n"
              % (vmax, fmt(a["in_band"]), fmt(a["e4"]), fmt(a["e5"]),
                 fmt(a["plateau"]), fmt(a["e5"] / a["e4"], 2)))
        if ref is not None:
            t += (" & trained FNO & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ \\\\\n"
                  % (fmt(ref[0]), fmt(ref[1]), fmt(ref[2]), fmt(ref[3]),
                     fmt(ref[4], 2)))
        t += (" & trivial, $\\hat u\\equiv0$ & n/a & n/a & n/a & $%s$ & n/a \\\\\n"
              % fmt(a["trivial_plateau"]))
        if vmax != C.VMAX_LIST[-1]:
            t += "\\midrule\n"
    return t + "\\bottomrule\n\\end{tabular}\n\\end{table}\n"


# ---------------------------------------------------------------- deep anchor
# Everything below is read from three zero-GPU instruments:
#   results/anchor_higher_order.json  -- plateau vs head depth M
#   results/anchor_contraction.json   -- the Born contraction factor rho(v, f)
#   results/anchor_order_selection.json -- M chosen in band, scored out of band
# plus the BA1/BA2 evaluation records for the learned-residual arms.

def deep_load():
    """The three zero-GPU anchor instruments, or None if any is missing.

    Requiring all three is deliberate.  The order scan alone would invite the
    reading "deeper is better"; the contraction measurement is what supplies
    the reason a turning point exists; the order-selection record is what
    makes the depth a choice rather than a dial tuned on the test set.  A
    partial set would support a stronger claim than the evidence carries.
    """
    out = {}
    for key, name in (("order", "anchor_higher_order.json"),
                      ("rho", "anchor_contraction.json"),
                      ("sel", "anchor_order_selection.json")):
        p = os.path.join(C.RES, name)
        if not os.path.exists(p):
            return None
        out[key] = json.load(open(p, encoding="utf-8"))
    return out


def _ba_stats(ev, prefix, vmax):
    """(in_band, e4, e5, plateau, r) for a learned-residual arm, or None.

    Same protocol as the trained arms it is read against: means over the seeds
    whose evaluation record exists on the current grid.  Returns None rather
    than a partial number so a table row is omitted instead of half-written.
    """
    names = runs_for(ev, prefix)
    if not names:
        return None
    F, M, _ = mean_over_seeds(ev, names, vmax)
    ib, _ = inband(ev, names, vmax)
    plateau = float(np.mean(M[F >= 6]))
    e4 = float(M[np.isclose(F, 4.0)][0]) if np.isclose(F, 4.0).any() else np.nan
    e5 = float(M[np.isclose(F, 5.0)][0]) if np.isclose(F, 5.0).any() else np.nan
    ok = np.isfinite(e4) and e4 > 0 and np.isfinite(e5)
    r = (e5 / e4) if ok else float("nan")
    return float(ib), e4, e5, plateau, r, names


def deep_macros():
    """Prose macros for the deep analytic head.

    The macros that carry the argument are the ones a reader can check against
    the tables: the plateau at the in-band-selected depth, the depth itself,
    the gain over the shallow head already in the manuscript, and the measured
    contraction factor with its frequency spread.  The frequency spread is
    included because it is the number that CORRECTS the manuscript's earlier
    "near-frequency-independent" wording: rho does move with f, by a factor
    \RhoSpread, which is real but three times smaller than the cliff.
    """
    keys = ["DAMsel", "DAMselStrong", "DAPlatSel", "DAInSel", "DAPlatMTwo",
            "DAFnoPlat", "DAFnoIn", "DAWinOut", "DAWinIn",
            "DAPlatWeak", "DAPlatStrong", "DAFnoPlatWeak", "DAFnoPlatStrong",
            "DAGainMTwo", "DAGainFno",
            "RhoLevel", "RhoSpread", "RhoWeak", "RhoStrong", "RhoUnit",
            "RhoRatioF",
            "DATurnM", "DATurnA", "DATurnB", "DARefusedCond",
            "BAOneIn", "BAOnePlat", "BAOneR", "BAOneFloor",
            "BAOneWinIn", "BAOneWinOut", "DAGainHeadIn",
            "BATwoIn", "BATwoPlat", "BATwoR", "BATwoFloor",
            "BATwoWinIn", "BATwoWinOut",
            "DAFnoInStrong", "DAPlatMid",
            "BATwoInStrong", "BATwoPlatStrong", "BATwoRStrong",
            "BATwoInMid", "BATwoPlatMid", "BATwoRMid",
            "DAFnoInMid", "DAFnoPlatMid"]
    d = deep_load()
    if d is None:
        return ["\\newcommand{\\%s}{--}" % k for k in keys]
    ev = load(RES_DIR)

    def arm(v):
        return d["sel"]["arms"]["%.2f" % v]

    def ordv(v, M):
        return d["order"]["arms"]["%.2f" % v]["orders"].get(str(M), {})

    m = {}
    v025, v100 = 0.25, 1.00
    m["DAMsel"] = "%d" % arm(v025)["M_sel"]
    m["DAMselStrong"] = "%d" % arm(v100)["M_sel"]
    m["DAPlatSel"] = fmt(arm(v025)["plateau_sel"])
    m["DAInSel"] = fmt(arm(v025)["in_band_sel"])
    m["DAPlatMTwo"] = fmt(arm(v025)["plateau_m2"])
    m["DAPlatWeak"] = fmt(arm(0.05)["plateau_sel"])
    m["DAPlatStrong"] = fmt(arm(v100)["plateau_sel"])

    ref = arm_curves(ev, v025) if ev else None
    if ref is not None:
        F, plateau, e4, e5, ratio, names = ref
        ib, _ = inband(ev, names, v025)
        m["DAFnoPlat"] = fmt(plateau)
        m["DAFnoIn"] = fmt(ib)
        # out-of-band win: trained plateau over head plateau
        m["DAWinOut"] = fmt(plateau / max(arm(v025)["plateau_sel"], 1e-9), 1)
        # in-band: whichever object is better, expressed as the winner's factor
        a_in = arm(v025)["in_band_sel"]
        m["DAWinIn"] = fmt(max(ib, a_in) / min(ib, a_in), 1)
    ref100 = arm_curves(ev, v100) if ev else None
    if ref100 is not None:
        F, plateau, e4, e5, ratio, names = ref100
        # NB: not named `ib` -- that name already holds the v=0.25 in-band
        # value, and rebinding it here silently scaled every later ratio by
        # the wrong contrast's error level.
        _ib100, _ = inband(ev, names, v100)
        del _ib100
        m["DAFnoPlatWeak"] = fmt(arm_curves(ev, 0.05)[1]) if arm_curves(
            ev, 0.05) else "--"
        # NB: DAFnoPlatStrong is set once, in the Strong/Mid loop below --
        # assigning it here too is a duplicate \newcommand in the preamble.
    m["DAGainMTwo"] = fmt(arm(v025)["plateau_m2"] /
                        max(arm(v025)["plateau_sel"], 1e-9), 1)
    if "DAFnoPlat" in m:
        m["DAGainFno"] = fmt(float(m["DAFnoPlat"]) /
                             max(arm(v025)["plateau_sel"], 1e-9), 1)

    rho = d["rho"]["arms"]
    m["RhoLevel"] = fmt(rho["0.25"]["rho_level"])
    m["RhoSpread"] = fmt(float(np.mean([rho["%.2f" % v]["rho_spread"]
                                        for v in C.VMAX_LIST])), 2)
    m["RhoWeak"] = fmt(rho["0.05"]["rho_level"])
    m["RhoStrong"] = fmt(rho["1.00"]["rho_level"])
    # rho ~ c * v with c measured; the rho = 1 boundary follows from it
    c = np.mean([rho["%.2f" % v]["rho_level"] / v for v in C.VMAX_LIST])
    m["RhoUnit"] = fmt(1.0 / c)
    m["RhoRatioF"] = fmt(rho["0.25"]["rho_at_f12"] / rho["0.25"]["rho_at_f4"], 2)

    # the turning point at the highest contrast, read from the scan itself
    ps = [(M, ordv(v100, M).get("plateau")) for M in range(1, 9)]
    ps = [(M, p) for M, p in ps if p is not None]
    turn = None
    for i in range(len(ps) - 1):
        if ps[i + 1][1] > ps[i][1]:
            turn = (ps[i][0], ps[i][1], ps[i + 1][1])
            break
    if turn:
        m["DATurnM"] = "%d" % turn[0]
        m["DATurnA"] = fmt(turn[1])
        m["DATurnB"] = fmt(turn[2])
    refused = [M for M in range(1, 9)
               if ordv(v100, M).get("fitted") is False]
    if refused:
        cond = ordv(v100, refused[0]).get("cond")
        m["DARefusedCond"] = ("$%.1f\\times10^{%d}$" %
                              (cond / 10 ** int(np.floor(np.log10(cond))),
                               int(np.floor(np.log10(cond))))
                              ) if cond else "--"

    if ev:
        b1 = _ba_stats(ev, "ba_fno_a025_s", v025)
        if b1:
            m["BAOneIn"] = fmt(b1[0])
            m["BAOnePlat"] = fmt(b1[3])
            m["BAOneR"] = fmt(b1[4], 2)
        # the floor BA1 is scored against is the M=2 head it was built on
        m["BAOneFloor"] = fmt(arm(v025)["plateau_m2"])
        m["BAOneWinIn"] = fmt(ib / b1[0], 1)
        m["BAOneWinOut"] = fmt(float(m["DAFnoPlat"]) / b1[3], 1)
        b2 = _ba_stats(ev, "ba_fno_a025_m8_s", v025)
        if b2:
            m["BATwoIn"] = fmt(b2[0])
            m["BATwoPlat"] = fmt(b2[3])
            m["BATwoR"] = fmt(b2[4], 2)
            m["BATwoFloor"] = fmt(arm(v025)["plateau_sel"])
            m["DAGainHeadIn"] = fmt(arm(v025)["in_band_sel"] / b2[0], 1)
            if "DAFnoIn" in m:
                m["BATwoWinIn"] = fmt(float(m["DAFnoIn"]) / b2[0], 1)
            if "DAFnoPlat" in m:
                m["BATwoWinOut"] = fmt(float(m["DAFnoPlat"]) / b2[3], 1)
        for vk, tag in (("1.00", "Strong"), ("0.50", "Mid")):
            v = float(vk)
            ref = arm_curves(ev, v) if ev else None
            if ref is not None:
                F0, pl0, e4, e5, ratio, names = ref
                ibf, _ = inband(ev, names, v)
                m["DAFnoIn" + tag] = fmt(ibf)
                m["DAFnoPlat" + tag] = fmt(pl0)
            b3 = None
            for M in (8, 7, 6, 5, 4):
                cand = _ba_stats(ev, "ba_fno_a%s_m%d_s" % (
                    "100" if tag == "Strong" else "050", M), v)
                if cand:
                    b3 = cand
                    break
            if tag == "Mid":
                o8 = d["order"]["arms"]["0.50"]["orders"].get(
                    str(d["sel"]["arms"]["0.50"]["M_sel"]), {})
                if o8.get("plateau") is not None:
                    m["DAPlatMid"] = fmt(o8["plateau"])
            if b3:
                m["BATwoIn" + tag] = fmt(b3[0])
                m["BATwoPlat" + tag] = fmt(b3[3])
                m["BATwoR" + tag] = fmt(b3[4], 2)

    out = []
    for k in keys:
        out.append("\\newcommand{\\%s}{%s}" % (k, m.get(k, "--")))
    return out


def table_deep(cap):
    """Head depth against the trained operator.

    One row per contrast: the head at its IN-BAND-selected depth, the head at
    the manuscript's existing depth M=2, and the trained operator on the same
    media and seeds.  The M=2 and M_sel rows differ only in how many analytic
    terms are summed -- both are fitted on the same in-band prefix, and
    neither is trained -- so the gap between them is the truncation error the
    manuscript's original anchor was carrying.
    """
    d = deep_load()
    if d is None:
        return ""
    ev = load(RES_DIR)
    t = ("\\begin{table}[ht]\n\\centering\n\\caption{%s}\n\\label{tab:deep}\n"
         "\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
         "\\begin{tabular}{llrrrr}\n\\toprule\n"
         "$v_{\\max}$ & predictor & $M$ & $\\varepsilon_{\\mathrm{in}}$ & "
         "$\\varepsilon_{\\infty}$ & $\\varepsilon_{\\infty}$ ratio "
         "to FNO \\\\\n\\midrule\n" % cap["deep"])
    def ratio_to_fno(plateau, ref):
        """Trained plateau over head plateau, with a floor on the head.

        At the two lowest contrasts the head is exact to five decimals and the
        ratio runs past 1e5.  Printing it would claim a precision the
        measurement does not have -- the head's error there is at the level
        where the reference solver's own tolerance matters -- so it is
        reported as a bound instead.
        """
        if ref is None or not np.isfinite(plateau) or plateau <= 0:
            return "n/a"
        r = ref[1] / plateau
        return ">10^{3}" if r > 1000.0 else fmt(r, 1)

    for vmax in C.VMAX_LIST:
        a = d["sel"]["arms"]["%.2f" % vmax]
        ref = None
        if ev:
            rr = arm_curves(ev, vmax)
            if rr is not None:
                F, plateau, e4, e5, ratio, names = rr
                ib, _ = inband(ev, names, vmax)
                ref = (ib, plateau)
        t += ("$%.2f$ & analytic head, $M=M_{\\mathrm{sel}}$ & $%d$ & $%s$ & "
              "$%s$ & $%s$ \\\\\n"
              % (vmax, a["M_sel"], fmt(a["in_band_sel"]),
                 fmt(a["plateau_sel"]),
                 ratio_to_fno(a["plateau_sel"], ref)))
        ib2 = a["in_band"].get("2") if isinstance(a["in_band"], dict) else None
        t += (" & analytic head, $M=2$ (Table~\\ref{tab:anchor}) & $2$ & "
              "$%s$ & $%s$ & $%s$ \\\\\n"
              % (fmt(ib2) if ib2 is not None else "n/a",
                 fmt(a["plateau_m2"]),
                 ratio_to_fno(a["plateau_m2"], ref)))
        if ref is not None:
            t += (" & trained FNO & n/a & $%s$ & $%s$ & $1.0$ \\\\\n"
                  % (fmt(ref[0]), fmt(ref[1])))
        if vmax != C.VMAX_LIST[-1]:
            t += "\\midrule\n"
    return t + "\\bottomrule\n\\end{tabular}\n\\end{table}\n"


def _banddown_load():
    """Load results/banddown.json with a compact per-arm accessor, or None."""
    p = os.path.join(C.RES, "banddown.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p, encoding="utf-8"))
    fL = d["f_test"]
    for x in (0.25, 0.5, 0.75, 1.0):
        if x not in fL:
            return None
    return d, fL


def table_banddown(ev, cap):
    """Below-band behaviour of the FNO: error at the left band edge f=1 and
    below, against the arm's in-band floor and the same-arm trivial baseline.
    New evaluation (results/banddown.json) of the same checkpoints on paired
    media; means over \\NumSeeds\\ seeds, seed SD bounded by \\BandDownSD."""
    ld = _banddown_load()
    if ld is None:
        return "%% banddown.json missing or on an unexpected grid -- table omitted.\n"
    d, fL = ld
    i1, i075, i05, i025 = (fL.index(x) for x in (1.0, 0.75, 0.5, 0.25))
    t = ("\\begin{table}[t]\n\\centering\n"
         "\\caption{" + cap["banddown"] + "}\n"
         "\\label{tab:banddown}\n\\small\n\\setlength{\\tabcolsep}{4pt}\n"
         "\\begin{tabular}{lccccccc}\n\\toprule\n"
         "$v_{\\max}$ & $\\varepsilon_{\\mathrm{in}}$ & $\\varepsilon(1.0)$ & "
         "$\\varepsilon(0.75)$ & $\\varepsilon(0.5)$ & $\\varepsilon(0.25)$ & "
         "$\\|u\\|_2/\\|\\psi\\|_2$ @ $f{=}0.5$ & ratio \\\\\n\\midrule\n")
    for vk, vmax in zip(("a005", "a010", "a025", "a050", "a100"), C.VMAX_LIST):
        names = runs_for(ev, "main_fno_%s_" % vk)
        ib, _ = inband(ev, names, vmax) if names else (float("nan"), None)
        M = np.mean([[d["runs"]["main_fno_%s_s%d" % (vk, s)]
                      ["sweep"]["%.2f" % vmax][i]["mean"]
                      for i in range(len(fL))] for s in range(3)], axis=0)
        T = np.mean([[d["runs"]["main_fno_%s_s%d" % (vk, s)]
                      ["sweep"]["%.2f" % vmax][j]["triv_mean"]
                      for j in range(len(fL))] for s in range(3)], axis=0)
        t += ("$%.2f$ & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ \\\\\n"
              % (vmax, fmt(ib), fmt(M[i1]), fmt(M[i075]), fmt(M[i05]),
                 fmt(M[i025]), fmt(T[i05]), fmt(M[i05] / T[i05], 2)))
    return t + "\\bottomrule\n\\end{tabular}\n\\end{table}\n"


def banddown_macros(ev):
    """Ranges quoted in Section~\\ref{sec:banddown}, computed from banddown.json
    so they cannot drift from the table that carries them."""
    fallback = ["\\newcommand{\\BandDownJumpLo}{--}",
                "\\newcommand{\\BandDownJumpHi}{--}",
                "\\newcommand{\\BandDownRatioLo}{--}",
                "\\newcommand{\\BandDownRatioHi}{--}",
                "\\newcommand{\\BandDownPeakWeak}{--}",
                "\\newcommand{\\BandDownPeakStrong}{--}",
                "\\newcommand{\\BandDownSD}{--}",
                "\\newcommand{\\TrivHalfLo}{--}",
                "\\newcommand{\\TrivHalfHi}{--}"]
    ld = _banddown_load()
    if ld is None:
        return fallback
    d, fL = ld
    i1, i075, i05 = (fL.index(x) for x in (1.0, 0.75, 0.5))
    jumps, ratios, triv050, peaks, sdmax = [], [], [], [], 0.0
    for vk, vmax in zip(("a005", "a010", "a025", "a050", "a100"), C.VMAX_LIST):
        rows = [[d["runs"]["main_fno_%s_s%d" % (vk, s)]
                 ["sweep"]["%.2f" % vmax][i]["mean"]
                 for i in range(len(fL))] for s in range(3)]
        M = np.mean(rows, axis=0)
        S = np.std(rows, axis=0, ddof=1)
        T = np.mean([[d["runs"]["main_fno_%s_s%d" % (vk, s)]
                      ["sweep"]["%.2f" % vmax][j]["triv_mean"]
                      for j in range(len(fL))] for s in range(3)], axis=0)
        sdmax = max(sdmax, float(S[i1]), float(S[i075]), float(S[i05]), float(S[0]))
        jumps.append(M[i075] / M[i1])
        ratios.append(M[i05] / T[i05])
        triv050.append(T[i05])
        peaks.append(max(M[i075], M[i05], M[0]))
    return ["\\newcommand{\\BandDownJumpLo}{%s}" % fmt(min(jumps), 2),
            "\\newcommand{\\BandDownJumpHi}{%s}" % fmt(max(jumps), 2),
            "\\newcommand{\\BandDownRatioLo}{%s}" % fmt(min(ratios), 2),
            "\\newcommand{\\BandDownRatioHi}{%s}" % fmt(max(ratios), 2),
            "\\newcommand{\\BandDownPeakWeak}{%s}" % fmt(min(peaks), 3),
            "\\newcommand{\\BandDownPeakStrong}{%s}" % fmt(max(peaks), 3),
            "\\newcommand{\\BandDownSD}{%s}" % fmt(sdmax, 3),
            "\\newcommand{\\TrivHalfLo}{%s}" % fmt(min(triv050), 3),
            "\\newcommand{\\TrivHalfHi}{%s}" % fmt(max(triv050), 3)]


def conv_macros():
    """Convergence evidence from the training logs: how much the validation
    error still changes over the final epochs of the cosine schedule, per
    family, as a range over the three seeds.  Directly answers the question of
    whether the U-Net/DeepONet plateaus are early-stopped transients."""
    out = ["\\newcommand{\\ConvEpochsTail}{5}",
           "\\newcommand{\\ConvLrFloor}{10^{-5}}"]
    for prefix, tag in (("main_fno_", "ConvChgFno"),
                        ("base_unet_", "ConvChgUnet"),
                        ("pmatch_unet_", "ConvChgPmatch"),
                        ("base_don_", "ConvChgDon")):
        chg = []
        for s in range(3):
            p = os.path.join(C.run_dir(prefix + "a025_s%d" % s, create=False),
                             "train_log.csv")
            if not os.path.exists(p):
                continue
            rows = np.loadtxt(p, delimiter=",", skiprows=1)
            val = rows[:, 2]  # columns: epoch, train_relL2, val_relL2, lr, sec
            v0, v1 = val[-6], val[-1]
            chg.append(abs(v1 - v0) / v0 * 100.0)
        if chg:
            out.append("\\newcommand{\\%sLo}{%.2f}" % (tag, min(chg)))
            out.append("\\newcommand{\\%sHi}{%.2f}" % (tag, max(chg)))
        else:
            out.append("\\newcommand{\\%sLo}{--}" % tag)
            out.append("\\newcommand{\\%sHi}{--}" % tag)
    return out


def plateau_macros(ev):
    """Plateau-to-trivial ratio per arm (seed-mean error over f>=6 against the
    same-arm trivial baseline) and the coupling range, quoted in
    Section~\\ref{sec:contrast}."""
    out = ["\\newcommand{\\PlatCoupLo}{%s}"
           % fmt(min(v / (2 * C.GAMMA) for v in C.VMAX_LIST), 1),
           "\\newcommand{\\PlatCoupHi}{%s}"
           % fmt(max(v / (2 * C.GAMMA) for v in C.VMAX_LIST), 1)]
    # Digits are not allowed in LaTeX control-sequence names, so the per-arm
    # tags are words, keyed by the arm suffix they came from.
    _PLAT_TAG = {"005": "Weak", "010": "Soft", "025": "Mid",
                 "050": "Strong", "100": "Max"}
    for vk, vmax in zip(("a005", "a010", "a025", "a050", "a100"), C.VMAX_LIST):
        r = arm_curves(ev, vmax)
        tr = triv_row(vmax)
        triv = None
        if tr is not None:
            mask = np.array([f for f in C.F_TEST], float) >= 6
            triv = float(np.mean(tr[mask]))
        if r is None or triv is None:
            out.append("\\newcommand{\\PlatTriv%s}{--}" % _PLAT_TAG.get(vk[1:], vk[1:]))
            continue
        plateau = r[1]
        out.append("\\newcommand{\\PlatTriv%s}{%.3f}"
                   % (_PLAT_TAG.get(vk[1:], vk[1:]), plateau / triv))
    # How far the paired denominator sits from the 600-media one it replaced.
    # Quoted in Section~\ref{sec:cliff} so that the change of denominator is
    # visible as a number rather than left to the reader to notice.
    tp = os.path.join(C.RES, "triv_paired.json")
    mx = None
    if os.path.exists(tp):
        _tp = json.load(open(tp, encoding="utf-8"))
        mx = (_tp.get("rel_diff_vs_600") or {}).get("max")
    out.append("\\newcommand{\\BaselinePairedMax}{%s}"
               % (fmt(100.0 * mx, 1) if mx is not None else "--"))
    return out


def _gridthree_load():
    p = os.path.join(C.RES, "grid_conv_three.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def _ci_load():
    p = os.path.join(C.RES, "ci_plateau.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def _grid_rows(d, key, f=None):
    rows = d[key]["rows"]
    return [r for r in rows if (f is None or abs(r["f"] - f) < 1e-9)]


def _plat_mean(rows, field):
    sub = [r[field] for r in rows if r["f"] >= 6.0]
    return float(np.mean(sub)) if sub else None


def grid_macros():
    """Prose macros for the three-grid study and the resampling intervals.

    Two numbers here are deliberately not rounded up into a claim of
    invariance.  The plateau moves by GridPlatShift percent when the target is
    refined two levels, and the media-resampling half-width on the same
    quantity is of the same order; so the honest statement is "determined to a
    few percent by both sources", not "grid-independent".  What IS
    grid-independent is the plateau-to-baseline ratio, because the metric is a
    ratio in which the leading discretisation error largely cancels -- that is
    GridRatioShift, and it is the reason the paper's claims survive the
    refinement at all.
    """
    keys = ["GridOrderLo", "GridOrderHi", "GridDeltaTop", "GridDeltaTopStrong",
            "GridPlatShift", "GridPlatShiftStrong", "GridRatioShift",
            "GridRatioShiftStrong",
            "GridInputPath", "GridInputPathAbs",
            "CIPlatLo", "CIPlatHi", "CIInLo", "CIInHi", "CIJumpLo", "CIJumpHi",
            "CIOnsetShare", "CIUnetJumpLo", "CIUnetJumpHi", "CIUnetOnset",
            "CIInTwoLo", "CIInTwoHi", "CIPlatTwoLo", "CIPlatTwoHi",
            "CIJumpTwoLo", "CIJumpTwoHi", "CIBoot",
            "CIRatioLo", "CIRatioHi", "CIRatioMid", "CIRatioMaxHalf",
            "CIRatioResidHalf"]
    m = {}

    g = _gridthree_load()
    if g is not None:
        ps = [r["p"] for k in ("ref_0.25", "ref_1.00") for r in g[k]["rows"]]
        m["GridOrderLo"] = fmt(min(ps), 2)
        m["GridOrderHi"] = fmt(max(ps), 2)
        a0 = _grid_rows(g, "ref_0.25", 12.0)[0]
        a1 = _grid_rows(g, "ref_1.00", 12.0)[0]
        m["GridDeltaTop"] = fmt(a0["d64v256"])
        m["GridDeltaTopStrong"] = fmt(a1["d64v256"])
        for tag, key, out in (("", "model_0.25", "GridPlatShift"),
                              ("Strong", "model_1.00", "GridPlatShiftStrong")):
            rows = _grid_rows(g, key)
            p64 = _plat_mean(rows, "e64")
            p256 = _plat_mean(rows, "e256")
            t64 = _plat_mean(rows, "b64")
            t256 = _plat_mean(rows, "b256")
            if p64 and p256:
                m[out] = fmt(100.0 * (p256 / p64 - 1.0), 1)
            # One key per arm.  Writing both arms to GridRatioShift silently
            # kept only the second, exactly the rebinding bug that made
            # BAOneWinIn read 12.1 instead of 3.1 earlier in this project.
            if p64 and p256 and t64 and t256:
                m["GridRatioShift" + tag] = fmt(
                    100.0 * ((p256 / t256) / (p64 / t64) - 1.0), 1)
        rows = _grid_rows(g, "model_0.25", 12.0)[0]
        m["GridInputPath"] = fmt(100.0 * (rows["e_native"] / rows["e64"] - 1.0), 1)
        m["GridInputPathAbs"] = fmt(rows["e_native"])

    c = _ci_load()
    if c is not None:
        by = {r["tag"]: r for r in c["rows"]}
        m["CIBoot"] = "%d" % int(c["nboot"])
        r = by.get("main_fno_a025")
        if r:
            m["CIPlatLo"], m["CIPlatHi"] = fmt(r["e_plat_lo"]), fmt(r["e_plat_hi"])
            m["CIInLo"], m["CIInHi"] = fmt(r["e_in_lo"], 4), fmt(r["e_in_hi"], 4)
            m["CIJumpLo"], m["CIJumpHi"] = fmt(r["r_lo"], 2), fmt(r["r_hi"], 2)
            m["CIOnsetShare"] = "%d" % round(
                100 * max(r["onset_dist"].values()))
        u = by.get("base_unet_a025")
        if u:
            m["CIUnetJumpLo"] = fmt(u["r_lo"], 2)
            m["CIUnetJumpHi"] = fmt(u["r_hi"], 2)
            m["CIUnetOnset"] = fmt(u["onset"], 2)
        b = by.get("ba_fno_a025_m8")
        if b:
            m["CIInTwoLo"], m["CIInTwoHi"] = fmt(b["e_in_lo"], 4), fmt(b["e_in_hi"], 4)
            m["CIPlatTwoLo"] = fmt(b["e_plat_lo"])
            m["CIPlatTwoHi"] = fmt(b["e_plat_hi"])
            m["CIJumpTwoLo"] = fmt(b["r_lo"], 2)
            m["CIJumpTwoHi"] = fmt(b["r_hi"], 2)

        # Plateau relative to its own baseline: point value and interval for the
        # reference arm, and the widest interval across all arms as a share of
        # its point value.  This is the scalar the interpretation rests on, so
        # it is the one whose interval has to be quoted rather than implied.
        if r and r.get("ratio") is not None:
            m["CIRatioMid"] = fmt(r["ratio"], 3)
            m["CIRatioLo"] = fmt(r["ratio_lo"], 3)
            m["CIRatioHi"] = fmt(r["ratio_hi"], 3)
        # Tightness is claimed only for the arms whose numerator is of the same
        # order as its denominator.  The head-plus-residual arm has a plateau of
        # 0.026 against a baseline of 0.93, so its ratio is dominated by the
        # denominator and pairing has almost nothing to cancel there: quoting
        # one number for all nine arms would hide a 1.1x against a 40x.
        hw = [0.5 * (x["ratio_hi"] - x["ratio_lo"]) / x["ratio"]
              for x in c["rows"] if x.get("ratio") and x["ratio"] > 0.5]
        if hw:
            m["CIRatioMaxHalf"] = fmt(100 * max(hw), 1)
        hr = [x for x in c["rows"] if x.get("ratio") and x["ratio"] <= 0.5]
        if hr:
            x = hr[0]
            m["CIRatioResidHalf"] = fmt(
                50 * (x["ratio_hi"] - x["ratio_lo"]) / x["ratio"], 0)

    return ["\\newcommand{\\%s}{%s}" % (k, m.get(k, "--")) for k in keys]


def table_gridthree(cap):
    """Three-grid convergence, and the plateau against a refined target."""
    g = _gridthree_load()
    if g is None:
        return ""
    t = ("\\begin{table}[t]\n\\centering\n\\caption{%s}\n"
         "\\label{tab:gridthree}\n\\small\n"
         "\\setlength{\\tabcolsep}{4.6pt}\n"
         "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
         "\\multicolumn{7}{c}{\\textbf{(a) reference-solution convergence, "
         "$\\delta$ between adjacent grids}}\\\\\n\\midrule\n"
         " & \\multicolumn{3}{c}{$v_{\\max}=0.25$} & "
         "\\multicolumn{3}{c}{$v_{\\max}=1.00$}\\\\\n"
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
         "$f$ & $\\delta_{64\\to128}$ & $\\delta_{128\\to256}$ & $p$ & "
         "$\\delta_{64\\to128}$ & $\\delta_{128\\to256}$ & $p$\\\\\n"
         "\\midrule\n" % cap["gridthree"])
    for r0, r1 in zip(_grid_rows(g, "ref_0.25"), _grid_rows(g, "ref_1.00")):
        t += ("$%g$ & %s & %s & %s & %s & %s & %s\\\\\n"
              % (r0["f"], fmt(r0["d64v128"]), fmt(r0["d128v256"]), fmt(r0["p"], 2),
                 fmt(r1["d64v128"]), fmt(r1["d128v256"]), fmt(r1["p"], 2)))
    t += ("\\bottomrule\n\\end{tabular}\n\n\\vspace{6pt}\n\n"
          "\\begin{tabular}{llrrrr}\n\\toprule\n"
          "\\multicolumn{6}{c}{\\textbf{(b) FNO error against the same prediction "
          "scored on three targets, $v_{\\max}=0.25$}}\\\\\n\\midrule\n"
          "$f$ & target & $\\varepsilon$ & $\\varepsilon_{\\mathrm{triv}}$ & "
          "$\\varepsilon/\\varepsilon_{\\mathrm{triv}}$ & $\\sqrt{\\varepsilon^2"
          "+\\delta_{64\\to256}^2}$ \\\\\n\\midrule\n")
    for r, a in zip(_grid_rows(g, "model_0.25"), _grid_rows(g, "ref_0.25")):
        quad = float(np.hypot(r["e64"], a["d64v256"]))
        for lab, e, b in (("$64^2$", r["e64"], r["b64"]),
                          ("$128^2$", r["e128"], r["b128"]),
                          ("$256^2$", r["e256"], r["b256"])):
            q = fmt(quad) if lab == "$64^2$" else ""
            t += ("$%g$ & %s & %s & %s & %s & %s\\\\\n"
                  % (r["f"], lab, fmt(e), fmt(b), fmt(e / b), q))
        t += "\\midrule\n"
    # the loop closes each frequency block with a rule; the last one is
    # immediately followed by \bottomrule and would print two rules in a row
    while t.endswith("\\midrule\n"):
        t = t[:-len("\\midrule\n")]
    return t + "\\bottomrule\n\\end{tabular}\n\\end{table}\n"


def _loc_load():
    p = os.path.join(C.RES, "locators.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def _loc_span(vals):
    """Spread of a locator across the rungs that have a value."""
    v = [x for x in vals if x is not None]
    return (max(v) - min(v)) if len(v) > 1 else 0.0


def loc_macros():
    """Prose macros for the three-locator recomputation.

    The point of this block is that the three locators are NOT the same number
    on the reference arm (4.50, 4.25, [4.00,4.25]), yet each of them is
    constant across the capacity and data ladders.  So the ladder conclusion is
    locator-invariant even though the locators are not interchangeable, and
    both halves have to be said.
    """
    d = _loc_load()
    if d is None:
        return []
    m = {}
    L = d["ladders"]
    ref = [r for r in L["band"]["rows"] if r["label"].startswith("[1,4]")]
    if ref:
        r = ref[0]
        m["LocRefFstar"] = fmt(r["fstar"], 2)
        m["LocRefOnset"] = fmt(r["onset"], 2)
        if r["step"]:
            # A macro carrying `--` must never be used inside $...$: in math
            # mode it typesets as two minus signs, so a range reads as
            # "4.00 - -4.25".  The endpoints are therefore exported separately
            # and the dash is written in text mode at the point of use.
            m["LocRefStepLo"] = fmt(r["step"][0], 2)
            m["LocRefStepHi"] = fmt(r["step"][1], 2)
    # mode ladder: f* fires only where the in-band floor is low enough
    md = L["mode"]["rows"]
    fired = [r for r in md if r["fstar"] is not None]
    silent = [r for r in md if r["fstar"] is None]
    if fired:
        m["LocModeFstar"] = fmt(fired[0]["fstar"], 2)
        m["LocModeFstarSpread"] = fmt(_loc_span([r["fstar"] for r in fired]), 2)
    m["LocModeSilent"] = str(len(silent))
    m["LocModeRungs"] = str(len(md))
    ons = [r["onset"] for r in md if r["onset"] is not None]
    if ons:
        m["LocModeOnsetLo"] = fmt(min(ons), 2)
        m["LocModeOnsetHi"] = fmt(max(ons), 2)
    st = [r["step"] for r in md if r["step"]]
    if st:
        lo, hi = min(s[0] for s in st), max(s[1] for s in st)
        m["LocModeStepLo"] = fmt(lo, 2)
        m["LocModeStepHi"] = fmt(hi, 2)
        m["LocModeStepSame"] = "1" if (len({(round(s[0], 2), round(s[1], 2))
                                          for s in st}) == 1) else "0"
    # capacity and data ladders: all three locators flat
    for k, tag in (("cap", "Cap"), ("ndata", "Data")):
        rows = L[k]["rows"]
        m["Loc%sFstarSpread" % tag] = fmt(
            _loc_span([r["fstar"] for r in rows]), 2)
        m["Loc%sOnsetSpread" % tag] = fmt(
            _loc_span([r["onset"] for r in rows]), 2)
        nst = len({(round(s[0], 2), round(s[1], 2))
                   for s in (r["step"] for r in rows) if s})
        m["Loc%sStepVariants" % tag] = str(nst)
        m["Loc%sSilent" % tag] = str(len([r for r in rows if r["fstar"] is None]))
        ins = [r["inband"] for r in rows if r["inband"]]
        if len(ins) > 1:
            m["Loc%sInLo" % tag] = fmt(min(ins), 3)
            m["Loc%sInHi" % tag] = fmt(max(ins), 3)
            m["Loc%sInRatio" % tag] = fmt(max(ins) / min(ins), 1)
    # band ladder: all three move, and the widest band has no out-of-band node
    bd = L["band"]["rows"]
    fs = [r["fstar"] for r in bd if r["fstar"] is not None]
    on = [r["onset"] for r in bd if r["onset"] is not None]
    if fs:
        m["LocBandFstarLo"] = fmt(min(fs), 2)
        m["LocBandFstarHi"] = fmt(max(fs), 2)
        m["LocBandFstarSpread"] = fmt(max(fs) - min(fs), 2)
    if on:
        m["LocBandOnsetLo"] = fmt(min(on), 2)
        m["LocBandOnsetHi"] = fmt(max(on), 2)
    md_ins = [r["inband"] for r in md if r["inband"]]
    if len(md_ins) > 1:
        m["LocModeInLo"] = fmt(min(md_ins), 3)
        m["LocModeInHi"] = fmt(max(md_ins), 3)
        m["LocModeInRatio"] = fmt(max(md_ins) / min(md_ins), 1)
    # the widest band: the grid lies inside the trained band, so no locator exists
    off = [r for r in bd if r["fstar"] is None and r["onset"] is None
           and r["step"] is None]
    m["LocOffGridCount"] = str(len(off))
    if off:
        m["LocOffGridLabel"] = off[0]["label"].replace("[", "[").replace("]", "]")
    return ["\\newcommand{\\%s}{%s}" % (k, v) for k, v in sorted(m.items())]


def table_locators(cap):
    """Every rung of every control ladder, scored by all three locators.

    Each control section defines its own boundary locator where it is used, so
    the same paper compares an f*-shift for the mode budget against a
    step-shift for capacity.  This table removes the ambiguity by computing all
    three on all rungs from the archived per-seed curves: the locators disagree
    on their absolute value, and agree that capacity and data do not move the
    boundary.
    """
    d = _loc_load()
    if d is None:
        return ""
    t = ("\\begin{table}[t]\n\\centering\n\\caption{%s}\n\\label{tab:locators}\n"
         "\\footnotesize\n\\setlength{\\tabcolsep}{3.6pt}\n"
         "\\begin{tabular}{llcccc}\n\\toprule\n"
         "ladder & rung & $f^{\\star}$ & onset & largest step & "
         "$\\varepsilon_{\\mathrm{in}}$\\\\\n\\midrule\n" % cap["locators"])
    head = {"mode": "mode budget $M$",
            "cap": "capacity",
            "ndata": "training samples",
            "band": "training band"}
    for k in ("mode", "cap", "ndata", "band"):
        rows = d["ladders"][k]["rows"]
        t += ("\\multicolumn{6}{l}{\\emph{%s}}\\\\\n" % head[k])
        for r in rows:
            st = ("n/a" if r["step"] is None
                  else "%.2f--%.2f" % (r["step"][0], r["step"][1]))
            t += (" & \\quad %s & %s & %s & %s & %s\\\\\n"
                  % (r["label"],
                     "n/a" if r["fstar"] is None else "%.2f" % r["fstar"],
                     "n/a" if r["onset"] is None else "%.2f" % r["onset"],
                     st,
                     "n/a" if r["inband"] is None else "%.3f" % r["inband"]))
        if k != "band":
            t += "\\midrule\n"
    return t + ("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def table_ci(cap):
    """The four headline scalars with media-resampling intervals.

    The last column is the plateau relative to its own trivial baseline.  It is
    the quantity the paper's interpretation actually uses, so it cannot be the
    one column without an interval; the denominator comes from
    `baseline_paired.py`, i.e. from the evaluation media themselves, and the
    bootstrap resamples one index for numerator and denominator.  That is why
    its interval is an order of magnitude narrower than the marginal intervals
    of the two components: the media-difficulty component is common to both and
    cancels in the ratio.
    """
    c = _ci_load()
    if c is None:
        return ""
    # \scriptsize, not \footnotesize: this is the densest table in the paper
    # (7 columns, 4 of them carrying "est [lo, hi]" bootstrap intervals).  At
    # \footnotesize the tabular overflows the 470.6pt text block by 26.7pt;
    # shrinking the font keeps every digit at full precision instead of
    # rounding the intervals away.  The sparse tables (5-8 plain columns) are
    # \footnotesize; this one earns the smaller size by its column count.
    t = ("\\begin{table}[t]\n\\centering\n\\caption{%s}\n\\label{tab:ci}\n"
         "\\scriptsize\n\\setlength{\\tabcolsep}{3.0pt}\n"
         "\\begin{tabular}{llccccc}\n\\toprule\n"
         "predictor & $v_{\\max}$ & $\\varepsilon_{\\mathrm{in}}$ & "
         "$\\varepsilon_{\\infty}$ & $r$ & "
         "\\shortstack{$\\varepsilon_{\\infty}/$\\\\"
         "$\\varepsilon^{\\mathrm{triv}}_{\\infty}$} & "
         "onset\\\\\n\\midrule\n"
         % cap["ci"])
    order = ["main_fno_a005", "main_fno_a010", "main_fno_a025",
             "main_fno_a050", "main_fno_a100",
             "base_unet_a025", "base_unet_a100", "pmatch_unet_a025",
             "ba_fno_a025_m8"]
    name = {"main_fno": "FNO", "base_unet": "U-Net", "pmatch_unet": "U-Net (m)",
            "ba_fno": "head + residual"}
    by = {r["tag"]: r for r in c["rows"]}
    for tag in order:
        r = by.get(tag)
        if r is None:
            continue
        fam = tag.rsplit("_a", 1)[0]
        on = "%.2f" % r["onset"]
        if abs(r["onset_hi"] - r["onset_lo"]) > 1e-9:
            on = "%.2f--%.2f" % (r["onset_lo"], r["onset_hi"])
        if r.get("ratio") is not None:
            rat = "%s [%s, %s]" % (fmt(r["ratio"], 3),
                                   fmt(r["ratio_lo"], 3), fmt(r["ratio_hi"], 3))
        else:
            rat = "{--}"
        t += ("%s & $%.2f$ & %s [%s, %s] & %s [%s, %s] & %s [%s, %s] & %s & "
              "%s\\\\\n"
              % (name.get(fam, fam), r["vmax"],
                 fmt(r["e_in"], 4), fmt(r["e_in_lo"], 4), fmt(r["e_in_hi"], 4),
                 fmt(r["e_plat"], 4), fmt(r["e_plat_lo"], 4),
                 fmt(r["e_plat_hi"], 4),
                 fmt(r["r"], 2), fmt(r["r_lo"], 2), fmt(r["r_hi"], 2),
                 rat, on))
        if tag == "main_fno_a100":
            t += "\\midrule\n"
    return t + "\\bottomrule\n\\end{tabular}\n\\end{table}\n"


def _res_load():
    p = os.path.join(C.RES, "res_study.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def _load_results(name):
    """Any archive in results/, or None.  Keeps macro generation degradable."""
    p = os.path.join(C.RES, name)
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None


def _sci(x):
    """Percentages that are numerically zero are printed as powers of ten.

    The synthetic controls are band-limited by construction, so at the
    operators' own mode budget their out-of-band fraction is ~1e-8 %; printing
    that as a fixed-point number would either round it to 0.00 or print machine
    dust.  A macro body is the same string in math or text mode, so it carries
    the exponent explicitly and the prose wraps it in $...$.
    """
    if x == 0.0:
        return "0"
    if abs(x) >= 1e-4:
        return "%.2f" % x
    e = int(np.floor(np.log10(abs(x))))
    mant = x / (10.0 ** e)
    return "%.1f\\times10^{%d}" % (mant, e)


def _loc_cell(v, nd=2):
    return "n/a" if v is None else ("%%.%df" % nd) % v


def _step_cell(v):
    return "n/a" if v is None else "%.2f--%.2f" % (v[0], v[1])


def res_macros():
    """Prose macros for the resolution study.

    Two of the four contributions carry the same obligation -- show that the
    reported boundary is a property of the problem and not of the mesh -- and
    this block is the single machine-readable source for every number the text
    quotes about it.  Coarse is 64^2 (the grid every other table uses), Fine is
    128^2 (the retrained arm).
    """
    d = _res_load()
    if d is None:
        return []
    m = {}
    A = d.get("part_a", {}).get("grids", {})
    B = d.get("part_b", {}).get("grids", {})
    Cc = d.get("part_c", {}).get("grids", {})
    coarse, fine = "64", "128"
    if coarse in A and fine in A:
        a, b = A[coarse], A[fine]
        m["ResFstarCoarse"] = _loc_cell(a["fstar"])
        m["ResFstarFine"] = _loc_cell(b["fstar"])
        m["ResOnsetCoarse"] = _loc_cell(a["onset"])
        m["ResOnsetFine"] = _loc_cell(b["onset"])
        m["ResStepCoarseLo"] = "%.2f" % a["step"][0] if a["step"] else "n/a"
        m["ResStepCoarseHi"] = "%.2f" % a["step"][1] if a["step"] else "n/a"
        m["ResStepFineLo"] = "%.2f" % b["step"][0] if b["step"] else "n/a"
        m["ResStepFineHi"] = "%.2f" % b["step"][1] if b["step"] else "n/a"
        m["ResJumpCoarse"] = "%.2f" % a["jump"]
        m["ResJumpFine"] = "%.2f" % b["jump"]
        m["ResJumpDevPct"] = "%.1f" % (100.0 * abs(b["jump"] - a["jump"]) / a["jump"])
        m["ResPlatCoarse"] = "%.3f" % a["plateau"]
        m["ResPlatFine"] = "%.3f" % b["plateau"]
        m["ResInCoarse"] = "%.3f" % a["in_band"]
        m["ResInFine"] = "%.3f" % b["in_band"]
        if a.get("val") and b.get("val"):
            m["ResValCoarse"] = "%.4f" % a["val"][0]
            m["ResValFine"] = "%.4f" % b["val"][0]
        # the per-seed locator values are what turn "the boundary did not move"
        # into "it did not move by more than the seed spread"
        fs = [v for v in (b.get("fstar_seeds") or []) if v is not None]
        if fs:
            m["ResFstarSeedLo"] = "%.2f" % min(fs)
            m["ResFstarSeedHi"] = "%.2f" % max(fs)
        js = b.get("jump_seeds") or []
        if js:
            m["ResJumpSeedLo"] = "%.2f" % min(js)
            m["ResJumpSeedHi"] = "%.2f" % max(js)
        # The coarse arm's own spread, and the gap between the two arms' seed
        # ranges.  Sign is the point: a positive gap means every refined seed
        # sits above every coarse seed, so the shift is systematic and quoting
        # it as "inside the seed spread" would be true of the magnitude and
        # false of the structure.  The prose reads this macro, not its own
        # arithmetic.
        acs = a.get("jump_seeds") or []
        if acs and js:
            m["ResJumpCoarseSeedLo"] = "%.2f" % min(acs)
            m["ResJumpCoarseSeedHi"] = "%.2f" % max(acs)
            m["ResJumpGap"] = "%.2f" % (min(js) - max(acs))
        for nm, blk in (("Coarse", a), ("Fine", b)):
            vs = blk.get("val") or []
            if vs:
                m["ResVal%sLo" % nm] = "%.4f" % min(vs)
                m["ResVal%sHi" % nm] = "%.4f" % max(vs)
        m["ResPlatDevPct"] = ("%.2f" % (100.0 * abs(b["plateau"] - a["plateau"])
                                        / max(a["plateau"], 1e-30)))
        m["ResInDevPct"] = ("%.1f" % (100.0 * abs(b["in_band"] - a["in_band"])
                                      / max(a["in_band"], 1e-30)))
        m["ResGridN"] = str(len(A))
        m["ResSeeds"] = str(len(b.get("val") or []))
        # observed shift, in units of the test grid step that resolves the edge
        if a["fstar"] is not None and b["fstar"] is not None:
            m["ResFstarShift"] = "%.2f" % abs(b["fstar"] - a["fstar"])
    if coarse in B and fine in B:
        a, b = B[coarse], B[fine]
        m["ResPairFstarCoarse"] = _loc_cell(a["fstar"])
        m["ResPairFstarFine"] = _loc_cell(b["fstar"])
        m["ResPairJumpCoarse"] = "%.2f" % a["jump"]
        m["ResPairJumpFine"] = "%.2f" % b["jump"]
        m["ResPairJumpDevPct"] = ("%.1f" % (100.0 * abs(b["jump"] - a["jump"])
                                            / max(a["jump"], 1e-30)))
        m["ResPairPlatCoarse"] = "%.3f" % a["plateau"]
        m["ResPairPlatFine"] = "%.3f" % b["plateau"]
        m["ResPairOnsetCoarse"] = _loc_cell(a["onset"])
        m["ResPairOnsetFine"] = _loc_cell(b["onset"])
    if coarse in Cc and fine in Cc:
        a, b = Cc[coarse]["family"], Cc[fine]["family"]
        for tag, blk, pre in (("Coarse", a, "C"), ("Fine", b, "F")):
            m["ResGeoFstar%s" % tag] = _loc_cell(blk["real"]["fstar"])
            m["ResGeoSynFstar%s" % tag] = _loc_cell(blk["synthetic"]["fstar"])
            m["ResGeoJump%s" % tag] = "%.2f" % blk["real"]["jump"]
            m["ResGeoSynJump%s" % tag] = "%.2f" % blk["synthetic"]["jump"]
            m["ResGeoPlat%sLo" % tag] = "%.3f" % blk["real"]["plateau5_lo"]
            m["ResGeoPlat%sHi" % tag] = "%.3f" % blk["real"]["plateau5_hi"]
            m["ResGeoSynPlat%sLo" % tag] = "%.3f" % blk["synthetic"]["plateau5_lo"]
            m["ResGeoSynPlat%sHi" % tag] = "%.3f" % blk["synthetic"]["plateau5_hi"]
        # How far apart the two families are at each grid, measured as the gap
        # between the highest geological window and the lowest synthetic
        # control.  Negative means the ranges interpenetrate.  Both grids are
        # emitted because the paper's claim is asymmetric between them and the
        # prose must quote this number instead of asserting separation.
        for tag, key in (("Coarse", coarse), ("Fine", fine)):
            blk = Cc[key]["family"]
            m["ResGeoPlatGap%s" % tag] = "%.3f" % (blk["synthetic"]["plateau5_lo"]
                                                   - blk["real"]["plateau5_hi"])
        m["ResGeoJumpDevPct"] = ("%.1f" % (100.0 * abs(b["real"]["jump"] - a["real"]["jump"])
                                           / max(a["real"]["jump"], 1e-30)))
        m["ResGeoSynJumpDevPct"] = ("%.1f" % (100.0 * abs(b["synthetic"]["jump"]
                                                          - a["synthetic"]["jump"])
                                              / max(a["synthetic"]["jump"], 1e-30)))
        # the band fraction is stated at the network's own mode budget
        mb = "20"
        names = ("west", "centre", "east")
        pg = Cc[fine].get("per_geometry", {})
        if all(("geo_%s" % w) in pg and ("syn_%d" % s) in pg
               for w in names for s in (0,)):
            geo = [pg["geo_%s" % w] for w in names]
            syn = [pg["syn_%d" % s] for s in range(3)]
            if all(mb in g["bands"] for g in geo + syn):
                m["ResBandGeoFinePct"] = "%.2f" % (100.0 * np.mean([g["bands"][mb] for g in geo]))
                m["ResBandSynFinePct"] = _sci(100.0 * np.mean([g["bands"][mb] for g in syn]))
                # No ratio macro: at the operators' own mode budget the control
                # fields are band-limited and carry ~1e-9 %, so a ratio would
                # divide by numerical dust and print a number in the 1e8 range.
                # The two percentages are the honest form of the same statement.
                m["ResBandMode"] = str(mb)
        if coarse in Cc:
            p0 = Cc[coarse].get("per_geometry", {})
            if all(("geo_%s" % w) in p0 and ("syn_%d" % s) in p0
                   for w in names for s in (0,)):
                geo0 = [p0["geo_%s" % w] for w in names]
                syn0 = [p0["syn_%d" % s] for s in range(3)]
                if all(mb in g["bands"] for g in geo0 + syn0):
                    m["ResBandGeoCoarsePct"] = "%.2f" % (100.0 * np.mean([g["bands"][mb] for g in geo0]))
                    m["ResBandSynCoarsePct"] = _sci(100.0 * np.mean([g["bands"][mb] for g in syn0]))
                    # the windows' own fraction is not mesh-invariant; the
                    # coarse windows are area-averaged, which removes exactly
                    # the modes being counted
                    m["ResBandGeoDevPct"] = "%.0f" % (100.0 * abs(
                        np.mean([g["bands"][mb] for g in geo])
                        - np.mean([g["bands"][mb] for g in geo0]))
                        / max(np.mean([g["bands"][mb] for g in geo0]), 1e-30))

    # ---------------------------------------------------------------- (v22)
    # Paired intervals on the two readings that were quoted as point values,
    # and the spectrum-matched control that separates spectral content from
    # geological structure.  Every number is computed by code/p0_checks.py and
    # code/geo_windows.py from archived per-seed and per-window curves.
    pc = _load_results("p0_checks.json") or {}
    ga = pc.get("grid", {}).get("part_a", {})
    jp = ga.get("jump")
    if jp:
        m["ResJumpPairedT"] = "%.2f" % jp["t"]
        m["ResJumpPairedP"] = "%.3f" % jp["p"]
        m["ResJumpPairedLo"] = "%+.2f" % jp["ci"][0]
        m["ResJumpPairedHi"] = "%+.2f" % jp["ci"][1]
    pp = ga.get("plateau5")
    if pp:
        m["ResPlatPairedP"] = "%.2f" % pp["p"]

    # window-level spread of the Marmousi jump, which is the quantity that
    # explains why three windows cannot order two families
    mm = pc.get("marmousi", {})
    for tag, key in (("Coarse", "64"), ("Fine", "128")):
        rec = mm.get(key, {}).get("jump")
        if not rec:
            continue
        m["GeoJumpDiff%s" % tag] = "%.3f" % abs(rec["diff"])
        m["GeoJumpP%s" % tag] = "%.3f" % rec["p"]
        m["GeoJumpCILo%s" % tag] = "%+.3f" % rec["ci"][0]
        m["GeoJumpCIHi%s" % tag] = "%+.3f" % rec["ci"][1]
        rv = np.asarray(rec.get("real") or [], float)
        if rv.size > 1:
            m["ResJumpWindowSpread"] = "%.0f" % (100.0 * (rv.max() - rv.min())
                                                 / rv.mean())

    sg = pc.get("surrogate", {})
    for tag, key in (("", "64"), ("Fine", "128")):
        s = sg.get(key)
        if not s:
            continue
        for pre, name in (("Jump", "jump_all"), ("Plat", "plateau5_all")):
            v = s.get(name)
            if not v:
                continue
            fmt = "%.2f" if pre == "Jump" else "%.3f"
            m["GeoSur%sReal%s" % (pre, tag)] = fmt % v["a"]
            m["GeoSur%sSur%s" % (pre, tag)] = fmt % v["b"]
            m["GeoSur%sDelta%s" % (pre, tag)] = "%.3f" % abs(v["diff"])
            # signed interval, because a bound that crosses zero has to be read
            # as crossing zero and printing |ci| would hide that.  The plateau
            # interval is reported as a magnitude because its sign is fixed on
            # both grids and the prose states the direction in words.
            for side, val in (("Lo", v["ci"][0]), ("Hi", v["ci"][1])):
                if pre == "Plat":
                    val = abs(val)
                sgn = "" if abs(val) < 5e-4 else ("+" if val > 0 else "-")
                m["GeoSur%s%s%s" % (pre, side, tag)] = sgn + ("%.3f" % abs(val))
            if pre == "Plat":
                lo, hi = sorted([abs(v["ci"][0]), abs(v["ci"][1])])
                m["GeoSurPlatLo%s" % tag] = "%.3f" % lo
                m["GeoSurPlatHi%s" % tag] = "%.3f" % hi
            m["GeoSur%sP%s" % (pre, tag)] = "%.3f" % v["p"]
            m["GeoSur%sDevPct%s" % (pre, tag)] = "%.1f" % (
                100.0 * abs(v["diff"]) / max(abs(v["a"]), 1e-30))
        m["GeoSurWinN"] = str(s.get("n_windows"))
        m["GeoSurGradRatio%s" % tag] = "%.2f" % s.get("grad_l1_ratio", float("nan"))
        if s.get("band_match_max_dev"):
            m["GeoSurBandDevPct%s" % tag] = "%.0f" % (100.0 * s["band_match_max_dev"])
            m["GeoSurBandMedPct%s" % tag] = "%.1f" % (100.0 * s["band_match_med_dev"])
        no = s.get("jump_nonoverlap")
        if no:
            m["GeoSurJumpNonOverP%s" % tag] = "%.2f" % no["p"]

    bc = _load_results("bound_cross.json") or {}
    if bc.get("n"):
        m["GridBoundN"] = str(int(bc["n"]))
    for row in bc.get("rows", []):
        tag = "Ffour" if int(row["f"]) == 4 else "Ffive"
        m["GridBoundEps%s" % tag] = "%.3f" % row["e64"]
        m["GridBoundEpsSame%s" % tag] = "%.3f" % row["e64_same_denom"]
        m["GridBoundDelta%s" % tag] = "%.3f" % row["delta"]
        m["GridBoundErr%s" % tag] = "%.3f" % row["e256"]
        m["GridBoundSame%s" % tag] = "%.3f" % row["bound_same_denom"]
        m["GridBoundMixed%s" % tag] = "%.3f" % row["bound_mixed_denom"]
        if row.get("bound_aggregate") is not None:
            m["GridBoundAgg%s" % tag] = "%.3f" % row["bound_aggregate"]
            m["GridBoundAggRatio%s" % tag] = "%.3f" % row["ratio_aggregate"]
        m["GridBoundRatio%s" % tag] = "%.3f" % row["ratio_same"]
        m["GridBoundCos%s" % tag] = "%+.3f" % row["cos_mean"]
        # the pointwise half of the story: a mean that satisfies the bound while
        # half the samples violate it must be reported as an RMS relation, so the
        # violation rate and the worst case are macros, not prose.
        if row.get("viol_same") is not None:
            m["GridBoundViolPct%s" % tag] = "%.0f" % (100.0 * row["viol_same"])
            m["GridBoundRatioMax%s" % tag] = "%.2f" % row["ratio_max_same"]
        if row.get("cos_sd") is not None:
            m["GridBoundCosSd%s" % tag] = "%.2f" % row["cos_sd"]
        if row.get("cos_negative_frac") is not None:
            m["GridBoundCosNegPct%s" % tag] = "%.0f" % (
                100.0 * row["cos_negative_frac"])
        if row.get("e256"):
            m["GridBoundMarginPct%s" % tag] = "%.1f" % (
                100.0 * (row["bound_same_denom"] / row["e256"] - 1.0))

    return ["\\newcommand{\\%s}{%s}" % (k, v) for k, v in sorted(m.items())]


def table_res(cap):
    """The resolution check in one place: is the boundary a property of the mesh?

    Four blocks.  (a) native protocol at both grids -- each model fed the media
    it was trained on; (b) the paired-continuum control, where the media are
    drawn once at 128^2 and block-averaged to 64^2, so the two models see the
    same continuum object and only the mesh differs; (c) and (d) the Marmousi
    windows and their synthetic control, recomputed at both grids by one code
    path.  Every row is scored by the same three locators as
    Table~\ref{tab:locators}, so a "no shift" statement here is the same
    statement that table makes about the control ladders.
    """
    d = _res_load()
    if d is None:
        return ""
    t = ("\\begin{table}[t]\n\\centering\n\\caption{%s}\n\\label{tab:res}\n"
         "\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
         "\\begin{tabular}{lcccccc}\n\\toprule\n"
         "grid & $f^{\\star}$ & onset & largest step & jump & "
         "$\\varepsilon_{\\mathrm{in}}$ & $\\varepsilon_{\\infty}$\\\\\n"
         "\\midrule\n" % cap["res"])

    def row(gridn, lb, pkey="plateau"):
        """One row.  `pkey` selects the plateau averaging window, and it is not
        cosmetic: block (a)/(b) must quote the same plateau as Table main
        (mean over $f>5$, i.e. the plateau proper), while (c)/(d) must quote
        the same plateau as Table realgeo (mean over $f\\ge5$), which is the
        only convention that table was ever generated under.  Printing one
        window in both places would put two numbers for one quantity in the
        paper; printing both without saying so would be worse."""
        return ("%s & %s & %s & %s & %.2f & %.3f & %.3f\\\\\n"
                % (gridn, _loc_cell(lb["fstar"]), _loc_cell(lb["onset"]),
                   _step_cell(lb["step"]), lb["jump"], lb["in_band"],
                   lb.get(pkey) if lb.get(pkey) is not None else lb["plateau"]))

    blocks = (("a", "(a) synthetic, native protocol", "synthetic"),
              ("b", "(b) synthetic, paired continuum", "synthetic"),
              ("c", "(c) Marmousi windows", "real"),
              ("c", "(d) synthetic control", "synthetic"))
    grids = (("64", "$64^{2}$"), ("128", "$128^{2}$"))
    for tag, head, famkey in blocks:
        blk = d.get("part_%s" % tag, {})
        t += "\\multicolumn{7}{l}{\\emph{%s}}\\\\\n" % head
        for gkey, glab in grids:
            if famkey == "synthetic" and tag == "c":
                fam = blk.get("grids", {}).get(gkey, {}).get("family", {}).get("synthetic")
                lb = None if not fam else dict(
                    fstar=fam["fstar"], onset=None, step=None, jump=fam["jump"],
                    in_band=fam["in_band"], plateau=fam["plateau"],
                    plateau5=fam.get("plateau5"))
            elif famkey == "real":
                fam = blk.get("grids", {}).get(gkey, {}).get("family", {}).get("real")
                lb = None if not fam else dict(
                    fstar=fam["fstar"], onset=None, step=None, jump=fam["jump"],
                    in_band=fam["in_band"], plateau=fam["plateau"],
                    plateau5=fam.get("plateau5"))
            else:
                lb = blk.get("grids", {}).get(gkey)
            if lb is not None:
                t += row(glab, lb, pkey=("plateau5" if tag == "c" else "plateau"))
        t += "\\midrule\n"
    while t.endswith("\\midrule\n"):
        t = t[:-len("\\midrule\n")]
    # The averaging windows are a note about the column, not part of the
    # grid, so it goes *under* the tabular: inside a p{} multicolumn it
    # stretches the seven columns to the full text width and the numbers
    # end up scattered across the page (rendered and checked, page 12).
    return (t + "\\bottomrule\n\\end{tabular}\n"
            "\\par\\vspace{2pt}\\parbox{\\linewidth}{\\scriptsize %s}\n"
            "\\end{table}\n" % (cap["res_note"]))


def bax_macros():
    """(v23) The fit-depth/sum-depth defect and the corrected arms.

    Sources: results/ls_head_alone.json (three head variants, formal
    protocol), results/proj_precheck.json (residual-projection probes),
    results/bax_read.json (corrected arms + paired tests + verdicts), and
    results/ba2_frz_read.json is NOT used -- the baf arms were trained on the
    defective head and are diagnostics, quoted only through the re-scored
    archives if at all.
    """
    keys = [
        # head alone, three variants, at the arm's own contrast
        "HeadTruncPlatMid", "HeadFullPlatMid", "HeadShallowPlatMid",
        "HeadTruncPlatStrong", "HeadFullPlatStrong", "HeadShallowPlatStrong",
        "HeadTruncInMid", "HeadFullInMid",
        "HeadTruncInStrong", "HeadFullInStrong",
        # residual-projection probes (as-trained arm, projections only)
        "ProjPlatLoMid", "ProjPlatHiMid",
        # corrected arms
        "BaxInMid", "BaxPlatMid", "BaxInStrong", "BaxPlatStrong",
        "BaxPlatRatioMid", "BaxPlatRatioStrong", "BaxOverHeadMid",
        "BaxOverHeadStrong", "BaxInWorsePctStrong",
        # paired tests bax vs trained FNO
        "BaxVsMainInPMid", "BaxVsMainInLoMid", "BaxVsMainInHiMid",
        "BaxVsMainPlatPMid", "BaxVsMainPlatLoMid", "BaxVsMainPlatHiMid",
        "BaxVsMainInPStrong", "BaxVsMainInLoStrong", "BaxVsMainInHiStrong",
        "BaxVsMainPlatPStrong", "BaxVsMainPlatLoStrong",
        "BaxVsMainPlatHiStrong",
        # pre-registered verdicts, as words
        "BaxVerdictMid", "BaxVerdictStrong",
        "RunInMid", "RunPlatMid", "RunInStrong", "RunPlatStrong",
    ]
    out = []
    ls = _load_results("ls_head_alone.json")
    bx = _load_results("bax_read.json")
    pp = _load_results("proj_precheck.json")
    m = {}
    if ls:
        for tag, vk in (("0.50", "Mid"), ("1.00", "Strong")):
            v = ls["arms"][tag]["variants"]
            own = tag  # the arm's own contrast row
            rows = lambda sw: sw[own]
            # the paper's plateau convention: mean over f >= 6, never the
            # worst point f = 5 on its own -- same as bax_read.py
            plat = lambda sw: float(np.mean([r["mean"] for r in rows(sw)
                                             if r["f"] >= 6.0]))
            inb = lambda sw: float(np.mean([r["mean"] for r in rows(sw)
                                            if r["f"] <= 4.0]))
            m["HeadTruncPlat" + vk] = fmt(plat(v["truncated"]))
            m["HeadFullPlat" + vk] = fmt(plat(v["full"]))
            m["HeadShallowPlat" + vk] = fmt(plat(v["shallow"]))
            m["HeadTruncIn" + vk] = fmt(inb(v["truncated"]))
            m["HeadFullIn" + vk] = fmt(inb(v["full"]))
    if pp:
        arm = pp.get("by_arm", {}).get("a050", {})
        projs = [arm["R%s" % r]["plateau"] for r in ("0", "4", "5", "6", "8")
                 if "R%s" % r in arm]
        if projs:
            m["ProjPlatLoMid"] = fmt(min(projs))
            m["ProjPlatHiMid"] = fmt(max(projs))
    # the arms AS RUN, under the truncated-head semantics they were scored
    # with when the previous draft reported their boundary -- the backups are
    # the only place those numbers still live
    bdir = os.path.join(C.RES, "_archive_truncated_semantics")
    for tag, m8, vk in (("a050", 8, "Mid"), ("a100", 7, "Strong")):
        ibs, pls = [], []
        for s in (0, 1, 2):
            p = os.path.join(bdir, "eval_ba_fno_%s_m%d_s%d.json" %
                             (tag, m8, s))
            if not os.path.exists(p):
                continue
            d = json.load(open(p, encoding="utf-8"))
            ib = next((r["mean"] for r in d["in_band"]
                       if abs(r["vmax"] - float("0.50" if tag == "a050"
                                                  else "1.00")) < 1e-6), None)
            rows = d["sweep"]["%.2f" % float("0.50" if tag == "a050"
                                             else "1.00")]
            pls.append(float(np.mean([r["mean"] for r in rows
                                      if r["f"] >= 6.0])))
            if ib is not None:
                ibs.append(float(ib))
        if ibs:
            m["RunIn" + vk] = fmt(float(np.mean(ibs)))
        if pls:
            m["RunPlat" + vk] = fmt(float(np.mean(pls)))
    if bx:
        for tag, vk in (("a050", "Mid"), ("a100", "Strong")):
            rec = bx["arms"].get(tag)
            if not rec:
                continue
            m["BaxIn" + vk] = fmt(rec["bax"]["in_band"])
            m["BaxPlat" + vk] = fmt(rec["bax"]["plateau"])
            m["BaxPlatRatio" + vk] = fmt(rec["plateau_over_main"], 2)
            m["BaxOverHead" + vk] = fmt(rec["plateau_over_ls_full"], 2)
            m["BaxVerdict" + vk] = rec["verdict"]
            if vk == "Strong":
                main_in = rec["main"]["in_band"]
                m["BaxInWorsePctStrong"] = fmt(
                    100.0 * (rec["bax"]["in_band"] - main_in) / main_in, 0)
            t = rec.get("paired_bax_vs_main", {})
            for what, pre in (("in_band", "In"), ("plateau", "Plat")):
                s = t.get(what)
                if s:
                    m["BaxVsMain%sP%s" % (pre, vk)] = "%.3f" % s["p"]
                    m["BaxVsMain%sLo%s" % (pre, vk)] = "%+.3g" % s["ci"][0]
                    m["BaxVsMain%sHi%s" % (pre, vk)] = "%+.3g" % s["ci"][1]
    for k in keys:
        out.append("\\newcommand{\\%s}{%s}" % (k, m.get(k, "--")))
    return out


def build(lang):
    """Return (macros, {table label: tex body}).

    The tables are emitted one file per label so that main.tex can \\input each
    one at its own first reference.  Concatenating them into a single
    gen_results.tex forced every float into one source location, and LaTeX then
    pushed the overflow past the end of the body -- the counters landed inside
    the reference list while the body pages stayed half empty.
    """
    cap = CAP[lang]
    ev = load(RES_DIR)
    arms = C.VMAX_LIST
    if not ev:
        print("no eval files"); return None, None
    s = table_summary(ev, arms, cap)
    mac = ("%% Auto-generated by code/make_results_tex.py (lang=%s) -- do not edit.\n"
           % lang) + macros(ev, arms, s[1])
    mac += "\n".join(banddown_macros(ev)) + "\n"
    mac += "\n".join(conv_macros()) + "\n"
    mac += "\n".join(plateau_macros(ev)) + "\n"
    mac += "\n".join(res_macros()) + "\n"
    mac += "\n".join(bax_macros()) + "\n"
    tabs = {
        "main": table_main(ev, arms, cap),
        "summary": s[0],
        "banddown": table_banddown(ev, cap),
        "arch": table_arch(ev, arms, cap),
        "band": table_band(ev, arms, cap),
        "mode": table_mode(ev, cap),
        "data": table_data(ev, cap),
        "born": table_born(ev, arms, cap),
        "propnorm": table_propnorm(cap),
        "paired": table_paired(cap),
        "cost": table_cost(cap),
        "realgeo": table_realgeo(arms, cap),
        "arms": table_arms(ev, cap),
        "c1": table_c1(cap),
        "disc": table_disc(cap),
        "anchor": table_anchor(cap),
        "deep": table_deep(cap),
        "gridthree": table_gridthree(cap),
        "ci": table_ci(cap),
        "locators": table_locators(cap),
        "res": table_res(cap),
    }
    if lang == "en":
        # the supplementary material is English-only; its paired table used to
        # be hand-maintained inside SI_main.tex and now shares this source
        tabs["si_paired"] = table_si_paired(CAP["en"])
    return mac, tabs


def main():
    global RES_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--res", default=None,
                    help="folder holding eval_*.json (default results/)")
    a = ap.parse_args()
    RES_DIR = a.res
    if a.out:
        mac, tabs = build("en")
        if tabs:
            open(a.out, "w", encoding="utf-8").write(
                "".join(tabs[k] for k in TAB_ORDER))
            print("wrote", a.out)
        return
    for lang, folder in [("en", "en"), ("cn", "cn")]:
        mac, tabs = build(lang)
        if tabs is None:
            continue
        d = os.path.join(C.ROOT, "paper", folder)
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "gen_macros.tex"), "w",
             encoding="utf-8").write(mac)
        print("wrote", os.path.join(d, "gen_macros.tex"))
        hdr = ("%% Auto-generated by code/make_results_tex.py (lang=%s) "
               "-- do not edit.\n" % lang)
        for label in TAB_ORDER:
            if label not in tabs:
                # a table only one language has (the SI is English-only)
                continue
            p = os.path.join(d, "gen_tab_%s.tex" % label)
            open(p, "w", encoding="utf-8").write(hdr + tabs[label])
            print("wrote", p)
        # the concatenated file is no longer input by main.tex; drop it so a
        # stale copy cannot be compiled by accident
        old = os.path.join(d, "gen_results.tex")
        if os.path.exists(old):
            os.remove(old)
            print("removed", old)


TAB_ORDER = ("main", "summary", "banddown", "propnorm", "paired", "arch", "band",
             "mode", "data", "born", "cost", "realgeo", "arms", "c1", "disc",
             "anchor", "deep", "gridthree", "ci",
             "locators", "res", "si_paired")


if __name__ == "__main__":
    main()

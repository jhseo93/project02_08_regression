"""price(목표변수)의 분포와 설명변수와의 관계를 표/그림으로 정리한다.

산출물:
  outputs/tables/  price 요약통계(전체 / 범주형 열별), 수치형 상관계수
  outputs/figures/ 분포, 범주형 박스플롯, 산포도, 리드타임 추이, 상관 히트맵
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAW_PATH = Path("data/raw/flight_prices.csv")
FIG_DIR = Path("outputs/figures")
TBL_DIR = Path("outputs/tables")

RANDOM_STATE = 42          # 산포도 표본 추출 재현성
SCATTER_SAMPLE = 20_000    # 30만 행 전체는 과밀하므로 표본만 사용

CATEGORICAL = ["class", "stops", "airline", "source_city",
               "destination_city", "departure_time", "arrival_time"]
NUMERIC = ["duration", "days_left"]
TARGET = "price"

STOPS_ORDER = ["zero", "one", "two_or_more"]
TIME_ORDER = ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]
CLASS_COLORS = {"Economy": "#4C72B0", "Business": "#C44E52"}


def setup_style() -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3


def load_raw() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"원본 데이터를 찾을 수 없습니다: {RAW_PATH}")
    return pd.read_csv(RAW_PATH)


def level_order(df: pd.DataFrame, col: str) -> list:
    """범주 표시 순서: 의미상 순서가 있으면 고정, 없으면 price 중앙값 오름차순."""
    if col == "stops":
        return STOPS_ORDER
    if col in ("departure_time", "arrival_time"):
        return TIME_ORDER
    return df.groupby(col, observed=True)[TARGET].median().sort_values().index.tolist()


# ---------------------------------------------------------------- 표

def table_overall(df: pd.DataFrame) -> pd.DataFrame:
    """전체 및 등급별 price 요약통계 (분위수 확장)."""
    qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]

    def stats(s: pd.Series, label: str) -> dict:
        row = {"구분": label, "n": len(s), "평균": s.mean(), "표준편차": s.std(),
               "최솟값": s.min(), "최댓값": s.max(),
               "왜도": s.skew(), "첨도": s.kurtosis(),
               "변동계수": s.std() / s.mean()}
        row.update({f"q{int(q * 100):02d}": s.quantile(q) for q in qs})
        return row

    rows = [stats(df[TARGET], "전체")]
    rows += [stats(g[TARGET], f"class={k}") for k, g in df.groupby("class", observed=True)]
    return pd.DataFrame(rows).round(2)


def table_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """범주형 열 x 수준별 price 요약통계를 long format 으로 누적."""
    rows = []
    for col in CATEGORICAL:
        for level in level_order(df, col):
            s = df.loc[df[col] == level, TARGET]
            rows.append({
                "변수": col, "수준": level, "n": len(s),
                "비율(%)": len(s) / len(df) * 100,
                "평균": s.mean(), "중앙값": s.median(), "표준편차": s.std(),
                "q25": s.quantile(0.25), "q75": s.quantile(0.75),
                "최솟값": s.min(), "최댓값": s.max(),
            })
    return pd.DataFrame(rows).round(2)


def table_correlation(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """수치화 가능한 열 간 Pearson / Spearman 상관계수."""
    num = pd.DataFrame({
        "duration": df.duration,
        "days_left": df.days_left,
        "stops_ord": df.stops.map({v: i for i, v in enumerate(STOPS_ORDER)}),
        "is_business": (df["class"] == "Business").astype(int),
        "price": df.price,
    })
    return num.corr("pearson").round(3), num.corr("spearman").round(3)


# ---------------------------------------------------------------- 그림

def fig_price_distribution(df: pd.DataFrame) -> None:
    """price 분포: 원척도 / 로그척도 / 등급별 분리."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.hist(df[TARGET], bins=80, color="#4C72B0", edgecolor="white", linewidth=0.3)
    ax.axvline(df[TARGET].mean(), color="#C44E52", ls="--", lw=1.6,
               label=f"평균 {df[TARGET].mean():,.0f}")
    ax.axvline(df[TARGET].median(), color="#55A868", ls="-", lw=1.6,
               label=f"중앙값 {df[TARGET].median():,.0f}")
    ax.set_title("전체 price 분포 — 봉우리 2개(이중 분포)")
    ax.set_xlabel("price (INR)")
    ax.set_ylabel("빈도")
    ax.legend()

    ax = axes[0, 1]
    ax.hist(np.log10(df[TARGET]), bins=80, color="#8172B2", edgecolor="white", linewidth=0.3)
    ax.set_title("log10(price) 분포 — 두 집단이 더 뚜렷")
    ax.set_xlabel("log10(price)")
    ax.set_ylabel("빈도")

    ax = axes[1, 0]
    for name, color in CLASS_COLORS.items():
        s = df.loc[df["class"] == name, TARGET]
        ax.hist(s, bins=70, alpha=0.65, color=color, label=f"{name} (n={len(s):,})")
    ax.set_title("등급별 price 분포 — 겹침 구간이 거의 없음")
    ax.set_xlabel("price (INR)")
    ax.set_ylabel("빈도")
    ax.legend()

    ax = axes[1, 1]
    data = [df.loc[df["class"] == n, TARGET].to_numpy() for n in CLASS_COLORS]
    bp = ax.boxplot(data, tick_labels=list(CLASS_COLORS), patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], CLASS_COLORS.values()):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.set_title("등급별 price 상자그림 (이상점 제외)")
    ax.set_ylabel("price (INR)")

    fig.suptitle("목표변수 price 의 분포", fontsize=15, y=1.00)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_price_distribution.png")
    plt.close(fig)


def fig_price_by_categorical(df: pd.DataFrame) -> None:
    """범주형 설명변수별 price 상자그림."""
    fig, axes = plt.subplots(3, 3, figsize=(17, 13))
    axes = axes.ravel()

    for ax, col in zip(axes, CATEGORICAL):
        order = level_order(df, col)
        data = [df.loc[df[col] == lv, TARGET].to_numpy() for lv in order]
        bp = ax.boxplot(data, tick_labels=order, patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor("#4C72B0")
            patch.set_alpha(0.6)
        medians = [np.median(d) for d in data]
        spread = max(medians) / min(medians)
        ax.set_title(f"{col} — 중앙값 최대/최소 배율 {spread:.2f}배")
        ax.set_ylabel("price (INR)")
        ax.tick_params(axis="x", rotation=45, labelsize=8)

    for ax in axes[len(CATEGORICAL):]:
        ax.axis("off")

    fig.suptitle("범주형 설명변수별 price (이상점 제외)", fontsize=15, y=1.00)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_price_by_categorical.png")
    plt.close(fig)


def fig_numeric_scatter(df: pd.DataFrame) -> None:
    """수치형 설명변수와 price 산포도 (등급별 색 구분, 표본 추출)."""
    sample = df.sample(min(SCATTER_SAMPLE, len(df)), random_state=RANDOM_STATE)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for j, col in enumerate(NUMERIC):
        ax = axes[0, j]
        for name, color in CLASS_COLORS.items():
            sub = sample[sample["class"] == name]
            ax.scatter(sub[col], sub[TARGET], s=4, alpha=0.18, color=color, label=name)
        r = df[col].corr(df[TARGET])
        rho = df[col].corr(df[TARGET], method="spearman")
        ax.set_title(f"{col} vs price — Pearson {r:+.3f} / Spearman {rho:+.3f}")
        ax.set_xlabel(col)
        ax.set_ylabel("price (INR)")
        leg = ax.legend(markerscale=4, fontsize=8)
        for lh in leg.legend_handles:
            lh.set_alpha(1)

        # 등급을 분리하면 관계가 달라지는지 확인 (Economy 만)
        ax = axes[1, j]
        eco_sample = sample[sample["class"] == "Economy"]
        ax.scatter(eco_sample[col], eco_sample[TARGET], s=4, alpha=0.18,
                   color=CLASS_COLORS["Economy"])
        eco_full = df[df["class"] == "Economy"]
        r_e = eco_full[col].corr(eco_full[TARGET])
        rho_e = eco_full[col].corr(eco_full[TARGET], method="spearman")
        ax.set_title(f"Economy 만: {col} vs price — Pearson {r_e:+.3f} / Spearman {rho_e:+.3f}")
        ax.set_xlabel(col)
        ax.set_ylabel("price (INR)")

    fig.suptitle(f"수치형 설명변수와 price 의 관계 (표본 {len(sample):,}행, seed={RANDOM_STATE})",
                 fontsize=15, y=1.00)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_price_vs_numeric_scatter.png")
    plt.close(fig)


def fig_days_left_trend(df: pd.DataFrame) -> None:
    """예약 리드타임에 따른 price 중앙값 추이 (등급별 / 경유별)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for name, color in CLASS_COLORS.items():
        grouped = df[df["class"] == name].groupby("days_left", observed=True)[TARGET]
        med = grouped.median()
        ax.plot(med.index, med.to_numpy(), color=color, lw=2, label=f"{name} 중앙값")
        ax.fill_between(med.index, grouped.quantile(0.25), grouped.quantile(0.75),
                        color=color, alpha=0.18)
    ax.set_title("days_left 별 price 중앙값 (음영 = q25~q75)")
    ax.set_xlabel("days_left (출발까지 남은 일수)")
    ax.set_ylabel("price (INR)")
    ax.legend()

    ax = axes[1]
    eco = df[df["class"] == "Economy"]
    for lv, color in zip(STOPS_ORDER, ["#55A868", "#4C72B0", "#C44E52"]):
        med = eco[eco.stops == lv].groupby("days_left", observed=True)[TARGET].median()
        ax.plot(med.index, med.to_numpy(), color=color, lw=2, label=f"stops={lv}")
    # 중앙값 일간 변화량이 가장 큰 지점(15 -> 16일, -2,032 INR)
    ax.axvline(16, color="gray", ls=":", lw=1.5)
    ax.set_title("Economy: 경유 횟수별 price 중앙값 추이 (점선 = 16일, 급락 지점)")
    ax.set_xlabel("days_left")
    ax.set_ylabel("price (INR)")
    ax.legend()

    fig.suptitle("예약 리드타임과 price — 비선형 구간이 뚜렷", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_days_left_trend.png")
    plt.close(fig)


def fig_correlation(pearson: pd.DataFrame, spearman: pd.DataFrame) -> None:
    """Pearson / Spearman 상관 히트맵."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, mat, name in zip(axes, [pearson, spearman], ["Pearson (선형)", "Spearman (순위)"]):
        im = ax.imshow(mat.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(mat)), mat.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(mat)), mat.index, fontsize=9)
        for i in range(len(mat)):
            for j in range(len(mat)):
                v = mat.iloc[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(v) > 0.55 else "black")
        ax.set_title(name)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("수치화 가능한 변수 간 상관계수", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_correlation.png")
    plt.close(fig)


# ---------------------------------------------------------------- 실행

def main() -> None:
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TBL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    print(f"불러온 행 수: {len(df):,}\n")

    overall = table_overall(df)
    by_cat = table_by_category(df)
    pearson, spearman = table_correlation(df)

    overall.to_csv(TBL_DIR / "price_summary_overall.csv", index=False, encoding="utf-8-sig")
    by_cat.to_csv(TBL_DIR / "price_summary_by_category.csv", index=False, encoding="utf-8-sig")
    pearson.to_csv(TBL_DIR / "correlation_pearson.csv", encoding="utf-8-sig")
    spearman.to_csv(TBL_DIR / "correlation_spearman.csv", encoding="utf-8-sig")

    with pd.option_context("display.width", 250, "display.max_columns", None):
        print("=" * 100)
        print("price 요약통계 (전체 / 등급별)")
        print("=" * 100)
        print(overall.to_string(index=False))
        print("\n" + "=" * 100)
        print("범주형 설명변수 수준별 price 요약통계")
        print("=" * 100)
        print(by_cat.to_string(index=False))
        print("\n" + "=" * 100)
        print("Pearson 상관계수")
        print("=" * 100)
        print(pearson.to_string())
        print("\nSpearman 상관계수")
        print(spearman.to_string())

    fig_price_distribution(df)
    fig_price_by_categorical(df)
    fig_numeric_scatter(df)
    fig_days_left_trend(df)
    fig_correlation(pearson, spearman)

    print("\n저장한 그림:")
    for p in sorted(FIG_DIR.glob("*.png")):
        print(f"  {p}  ({p.stat().st_size / 1024:.0f} KB)")
    print("저장한 표:")
    for p in sorted(TBL_DIR.glob("*.csv")):
        print(f"  {p}")


if __name__ == "__main__":
    main()

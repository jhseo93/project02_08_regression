"""의사결정나무(회귀)의 첫 분할 후보를 SSE(잔차제곱합) 기준으로 전수 평가한다.

CART 회귀나무는 각 분할 후보에 대해
    SSE_split = SSE(왼쪽) + SSE(오른쪽),   SSE(g) = sum((y - mean_g)^2)
를 계산하고 SSE_split 이 가장 작은 후보를 고른다.
루트 SSE 대비 감소량이 곧 그 분할의 '이득'이다.

산출물:
  outputs/tables/first_split_candidates.csv   후보별 SSE / 감소량 (내림차순)
  outputs/figures/06_first_split_sse.png      변수별 이득 비교 + 임계값 스캔 곡선
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

RAW_PATH = Path("data/raw/flight_prices.csv")
FIG_DIR = Path("outputs/figures")
TBL_DIR = Path("outputs/tables")

TARGET = "price"
RANDOM_STATE = 42

STOPS_ORDER = ["zero", "one", "two_or_more"]
# 합의된 설계: Unnamed: 0 은 식별자, flight 는 그룹키 -> 입력에서 제외
NOMINAL = ["airline", "source_city", "destination_city", "departure_time", "arrival_time"]
NUMERIC = ["duration", "days_left"]


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


# ---------------------------------------------------------------- SSE 계산

def sse(y: np.ndarray) -> float:
    """한 노드의 잔차제곱합. sum(y^2) - sum(y)^2 / n 형태로 한 번에 계산."""
    if len(y) == 0:
        return 0.0
    return float(y.dot(y) - y.sum() ** 2 / len(y))


def scan_numeric(x: np.ndarray, y: np.ndarray) -> dict:
    """수치형 변수의 모든 임계값을 누적합으로 한 번에 훑어 최적 분할을 찾는다."""
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order], y[order]
    n = len(ys)

    csum = np.cumsum(ys)
    csumsq = np.cumsum(ys * ys)
    total_sum, total_sumsq = csum[-1], csumsq[-1]

    n_left = np.arange(1, n)                 # 왼쪽 노드 크기 1 .. n-1
    n_right = n - n_left
    sum_l, sumsq_l = csum[:-1], csumsq[:-1]

    sse_l = sumsq_l - sum_l ** 2 / n_left
    sse_r = (total_sumsq - sumsq_l) - (total_sum - sum_l) ** 2 / n_right
    total = sse_l + sse_r

    # 값이 같은 지점 사이는 분할할 수 없음
    total = np.where(xs[:-1] != xs[1:], total, np.inf)

    i = int(np.argmin(total))
    return {
        "threshold": float((xs[i] + xs[i + 1]) / 2),
        "n_left": int(n_left[i]), "n_right": int(n_right[i]),
        "sse_left": float(sse_l[i]), "sse_right": float(sse_r[i]),
        "sse_total": float(total[i]),
        "curve_x": (xs[:-1] + xs[1:]) / 2, "curve_y": total,
    }


def eval_binary_mask(y: np.ndarray, mask: np.ndarray) -> dict:
    """참/거짓 마스크 하나로 나눈 분할의 SSE."""
    y_l, y_r = y[mask], y[~mask]
    sse_l, sse_r = sse(y_l), sse(y_r)
    return {"n_left": len(y_l), "n_right": len(y_r),
            "sse_left": sse_l, "sse_right": sse_r, "sse_total": sse_l + sse_r}


# ---------------------------------------------------------------- 후보 열거

def enumerate_candidates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """모든 분할 후보를 평가해 표로 반환한다. 원-핫 인코딩 기준(수준 vs 나머지)."""
    y = df[TARGET].to_numpy(dtype=float)
    root = sse(y)
    rows, curves = [], {}

    def add(variable: str, rule: str, kind: str, res: dict) -> None:
        rows.append({
            "변수": variable, "분할규칙": rule, "유형": kind,
            "n_left": res["n_left"], "n_right": res["n_right"],
            "SSE_left": res["sse_left"], "SSE_right": res["sse_right"],
            "SSE_split": res["sse_total"],
            "감소량": root - res["sse_total"],
            "감소율(%)": (root - res["sse_total"]) / root * 100,
        })

    # 1) 이진 범주형: class
    add("class", "class == Business", "이진",
        eval_binary_mask(y, (df["class"] == "Business").to_numpy()))

    # 2) 순서형: stops (0/1/2 로 매핑 후 임계값 스캔)
    stops_ord = df["stops"].map({v: i for i, v in enumerate(STOPS_ORDER)}).to_numpy(float)
    for thr, label in [(0.5, "stops == zero"), (1.5, "stops in {zero, one}")]:
        add("stops", label, "순서형", eval_binary_mask(y, stops_ord <= thr))

    # 3) 명목형: 각 수준 vs 나머지 (원-핫 후 트리가 실제로 시도하는 분할)
    for col in NOMINAL:
        for level in df[col].unique():
            add(col, f"{col} == {level}", "명목(수준vs나머지)",
                eval_binary_mask(y, (df[col] == level).to_numpy()))

    # 4) 수치형: 전 임계값 스캔
    for col in NUMERIC:
        res = scan_numeric(df[col].to_numpy(float), y)
        curves[col] = (res["curve_x"], res["curve_y"])
        add(col, f"{col} <= {res['threshold']:.4g}", "수치형", res)

    # 5) 참고: 제외하기로 한 식별자를 넣으면 어떻게 되는지 (누수 시연용)
    res = scan_numeric(df["Unnamed: 0"].to_numpy(float), y)
    curves["Unnamed: 0"] = (res["curve_x"], res["curve_y"])
    add("Unnamed: 0 (제외대상)", f"Unnamed: 0 <= {res['threshold']:.0f}", "식별자", res)

    table = pd.DataFrame(rows).sort_values("감소량", ascending=False).reset_index(drop=True)
    table.insert(0, "순위", np.arange(1, len(table) + 1))
    return table, {"root_sse": root, "curves": curves, "y": y}


# ---------------------------------------------------------------- 검증

def verify_with_sklearn(df: pd.DataFrame) -> str:
    """실제 sklearn 회귀나무(깊이 1)가 같은 분할을 고르는지 확인."""
    X = pd.get_dummies(
        df[["class", "stops", *NOMINAL, *NUMERIC]],
        columns=["class", "stops", *NOMINAL], drop_first=False,
    )
    tree = DecisionTreeRegressor(max_depth=1, random_state=RANDOM_STATE)
    tree.fit(X, df[TARGET])
    t = tree.tree_
    feature = X.columns[t.feature[0]]
    thr = t.threshold[0]
    # sklearn 의 impurity 는 MSE 이므로 SSE = impurity * n_samples
    root_sse = t.impurity[0] * t.n_node_samples[0]
    child_sse = (t.impurity[1] * t.n_node_samples[1] + t.impurity[2] * t.n_node_samples[2])
    return (f"선택된 분할: {feature} <= {thr:.4f}\n"
            f"  루트 SSE      : {root_sse:,.0f}\n"
            f"  분할 후 SSE   : {child_sse:,.0f}\n"
            f"  감소율        : {(root_sse - child_sse) / root_sse * 100:.2f}%\n"
            f"  왼쪽 n={t.n_node_samples[1]:,} 예측값={t.value[1][0][0]:,.0f} | "
            f"오른쪽 n={t.n_node_samples[2]:,} 예측값={t.value[2][0][0]:,.0f}")


# ---------------------------------------------------------------- 그림

def fig_first_split(table: pd.DataFrame, ctx: dict, df: pd.DataFrame) -> None:
    root = ctx["root_sse"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # A. 변수별 최고 후보 비교
    ax = axes[0, 0]
    best = table.groupby("변수", as_index=False)["감소율(%)"].max()
    best = best.sort_values("감소율(%)")
    colors = ["#C44E52" if "제외대상" in v else "#4C72B0" for v in best["변수"]]
    ax.barh(best["변수"], best["감소율(%)"], color=colors, alpha=0.85)
    for yi, v in enumerate(best["감소율(%)"]):
        ax.text(v + 0.6, yi, f"{v:.2f}%", va="center", fontsize=9)
    ax.set_xlim(0, max(best["감소율(%)"]) * 1.22)
    ax.set_xlabel("루트 SSE 대비 감소율 (%)")
    ax.set_title("변수별 최적 분할의 이득 — class 가 압도적")

    # B. days_left 임계값 스캔
    ax = axes[0, 1]
    cx, cy = ctx["curves"]["days_left"]
    ok = np.isfinite(cy)
    ax.plot(cx[ok], (root - cy[ok]) / root * 100, color="#55A868", lw=2)
    i = int(np.nanargmin(np.where(ok, cy, np.inf)))
    ax.axvline(cx[i], color="#C44E52", ls="--", lw=1.5,
               label=f"최적 임계값 {cx[i]:.1f} ({(root - cy[i]) / root * 100:.2f}%)")
    ax.set_xlabel("days_left 임계값")
    ax.set_ylabel("SSE 감소율 (%)")
    ax.set_title("days_left: 임계값별 이득 — 최적점이 뚜렷")
    ax.legend()

    # C. duration 임계값 스캔
    ax = axes[1, 0]
    cx, cy = ctx["curves"]["duration"]
    ok = np.isfinite(cy)
    ax.plot(cx[ok], (root - cy[ok]) / root * 100, color="#8172B2", lw=2)
    i = int(np.nanargmin(np.where(ok, cy, np.inf)))
    ax.axvline(cx[i], color="#C44E52", ls="--", lw=1.5,
               label=f"최적 임계값 {cx[i]:.2f} ({(root - cy[i]) / root * 100:.2f}%)")
    ax.set_xlabel("duration 임계값")
    ax.set_ylabel("SSE 감소율 (%)")
    ax.set_title("duration: 임계값별 이득")
    ax.legend()

    # D. 최적 분할 전후의 price 분포
    ax = axes[1, 1]
    ax.hist(df[TARGET], bins=70, color="gray", alpha=0.35, label="루트 (전체)")
    for name, color in [("Economy", "#4C72B0"), ("Business", "#C44E52")]:
        s = df.loc[df["class"] == name, TARGET]
        ax.hist(s, bins=70, alpha=0.7, color=color,
                label=f"{name} 노드 (평균 {s.mean():,.0f})")
    ax.set_xlabel("price (INR)")
    ax.set_ylabel("빈도")
    ax.set_title("class 분할 후: 두 자식 노드가 거의 겹치지 않음")
    ax.legend(fontsize=9)

    fig.suptitle("첫 분할 후보의 SSE 비교", fontsize=15, y=1.00)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_first_split_sse.png")
    plt.close(fig)


# ---------------------------------------------------------------- 실행

def main() -> None:
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TBL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    table, ctx = enumerate_candidates(df)
    root = ctx["root_sse"]
    y = ctx["y"]

    print(f"행 수 {len(df):,} | price 평균 {y.mean():,.2f} | 분산 {y.var():,.2f}")
    print(f"루트 노드 SSE = {root:,.0f}")
    print(f"  (검산: n x 분산 = {len(y):,} x {y.var():,.2f} = {len(y) * y.var():,.0f})\n")

    table.to_csv(TBL_DIR / "first_split_candidates.csv", index=False, encoding="utf-8-sig")

    show = table.copy()
    for c in ["SSE_left", "SSE_right", "SSE_split", "감소량"]:
        show[c] = show[c].map(lambda v: f"{v:,.3e}")
    show["감소율(%)"] = show["감소율(%)"].round(3)
    show["n_left"] = show["n_left"].map("{:,}".format)
    show["n_right"] = show["n_right"].map("{:,}".format)

    with pd.option_context("display.width", 260, "display.max_columns", None,
                           "display.max_rows", None):
        print("=" * 120)
        print(f"분할 후보 {len(table)}개 — SSE 감소량 내림차순 (상위 20개)")
        print("=" * 120)
        print(show.head(20).to_string(index=False))
        print("\n" + "=" * 120)
        print("하위 5개 (이득이 거의 없는 후보)")
        print("=" * 120)
        print(show.tail(5).to_string(index=False))

    print("\n" + "=" * 120)
    print("sklearn DecisionTreeRegressor(max_depth=1) 교차 검증")
    print("=" * 120)
    print(verify_with_sklearn(df))

    fig_first_split(table, ctx, df)
    print(f"\n저장: {TBL_DIR / 'first_split_candidates.csv'}")
    print(f"저장: {FIG_DIR / '06_first_split_sse.png'}")


if __name__ == "__main__":
    main()

"""max_depth x min_samples_leaf 격자 탐색 및 과적합 관찰.

  max_depth        : 2, 4, 6, 8, 12, 제한없음(None)
  min_samples_leaf : 500, 200, 100, 50, 20
  -> 조합 30개를 두 분할 방식에서 각각 학습 (총 60회)

다른 조건은 앞 단계와 동일하게 고정한다.
  범주형 원-핫, 수치형 원본, random_state=42

산출물:
  outputs/tables/hyperparam_grid.csv            전체 격자 지표 (학습/테스트)
  outputs/tables/hyperparam_grid_pivot.csv      test R2 피벗표
  outputs/figures/13_grid_train_test.png        학습 vs 테스트 오차 선그래프
  outputs/figures/14_overfit_gap.png            과적합 격차 추이
"""

from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor

SPLIT_PATH = Path("data/processed/flight_prices_with_splits.csv")
FIG_DIR = Path("outputs/figures")
TBL_DIR = Path("outputs/tables")

RANDOM_STATE = 42
MAX_DEPTHS = [2, 4, 6, 8, 12, None]
MIN_LEAVES = [500, 200, 100, 50, 20]
DEPTH_LABELS = ["2", "4", "6", "8", "12", "제한없음"]

TARGET = "price"
CATEGORICAL = ["airline", "source_city", "destination_city",
               "departure_time", "arrival_time", "class", "stops"]
NUMERIC = ["duration", "days_left"]
SCHEMES = ["split_group", "split_random"]

LEAF_COLOR = {500: "#4C72B0", 200: "#55A868", 100: "#C44E52",
              50: "#8172B2", 20: "#CCB974"}


def setup_style() -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3


def load_and_encode() -> tuple:
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"분할 파일이 없습니다: {SPLIT_PATH}\n"
            "먼저 analysis/04_train_test_split.py 를 실행하세요."
        )
    df = pd.read_csv(SPLIT_PATH)
    dummies = pd.get_dummies(df[CATEGORICAL], columns=CATEGORICAL, drop_first=False)
    return df, pd.concat([dummies.astype(int), df[NUMERIC]], axis=1)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": root_mean_squared_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred)}


def run_grid(df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scheme in SCHEMES:
        is_train = df[scheme] == "train"
        X_tr, X_te = X[is_train], X[~is_train]
        y_tr = df.loc[is_train, TARGET].to_numpy(float)
        y_te = df.loc[~is_train, TARGET].to_numpy(float)

        for depth, label in zip(MAX_DEPTHS, DEPTH_LABELS):
            for leaf in MIN_LEAVES:
                t0 = perf_counter()
                tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=leaf,
                                             random_state=RANDOM_STATE)
                tree.fit(X_tr, y_tr)
                elapsed = perf_counter() - t0

                tr_m = metrics(y_tr, tree.predict(X_tr))
                te_m = metrics(y_te, tree.predict(X_te))
                rows.append({
                    "분할방식": scheme, "max_depth": label, "min_samples_leaf": leaf,
                    "잎 개수": int(tree.get_n_leaves()),
                    "실제 깊이": int(tree.get_depth()),
                    "train_MAE": round(tr_m["MAE"], 1), "test_MAE": round(te_m["MAE"], 1),
                    "train_RMSE": round(tr_m["RMSE"], 1), "test_RMSE": round(te_m["RMSE"], 1),
                    "train_R2": round(tr_m["R2"], 4), "test_R2": round(te_m["R2"], 4),
                    "R2격차": round(tr_m["R2"] - te_m["R2"], 4),
                    "학습초": round(elapsed, 2),
                })
                print(f"  [{scheme}] depth={label:<5} leaf={leaf:<4} "
                      f"잎={tree.get_n_leaves():>6,} "
                      f"test_R2={te_m['R2']:.4f} ({elapsed:.1f}s)")
    return pd.DataFrame(rows)


def fig_train_test(grid: pd.DataFrame) -> None:
    """학습 오차와 테스트 오차를 두 분할 방식에 대해 나란히 그린다."""
    x = np.arange(len(DEPTH_LABELS))
    specs = [("MAE", "MAE (INR)", None), ("RMSE", "RMSE (INR)", None),
             ("R2", "R²", (0.90, 1.005))]

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for r, scheme in enumerate(SCHEMES):
        sub = grid[grid["분할방식"] == scheme]
        for ax, (key, ylabel, ylim) in zip(axes[r], specs):
            for leaf in MIN_LEAVES:
                s = (sub[sub["min_samples_leaf"] == leaf]
                     .set_index("max_depth").reindex(DEPTH_LABELS))
                color = LEAF_COLOR[leaf]
                ax.plot(x, s[f"test_{key}"], marker="o", ms=6, lw=2.2, color=color)
                ax.plot(x, s[f"train_{key}"], marker="^", ms=5, lw=1.6, ls="--",
                        color=color, alpha=0.55)
            ax.set_xticks(x, DEPTH_LABELS)
            ax.set_xlabel("max_depth")
            ax.set_ylabel(ylabel)
            ax.set_title(f"[{scheme}] {ylabel}", fontsize=12)
            if ylim:
                ax.set_ylim(*ylim)

    handles = [plt.Line2D([], [], color=LEAF_COLOR[l], lw=2.4,
                          label=f"min_samples_leaf={l}") for l in MIN_LEAVES]
    handles += [plt.Line2D([], [], color="black", lw=2.2, marker="o",
                           label="테스트 (실선·원)"),
                plt.Line2D([], [], color="black", lw=1.6, ls="--", alpha=0.55,
                           marker="^", label="학습 (점선·삼각)")]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=10,
               bbox_to_anchor=(0.5, -0.045), frameon=False)
    fig.suptitle("max_depth · min_samples_leaf 별 학습/테스트 오차 "
                 f"(random_state={RANDOM_STATE})\n"
                 "위=flight 그룹 분할(편명 겹침 0), 아래=무작위 분할",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "13_grid_train_test.png")
    plt.close(fig)


def fig_gap(grid: pd.DataFrame) -> None:
    """과적합 격차(train R2 - test R2)를 분할 방식별로."""
    x = np.arange(len(DEPTH_LABELS))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, scheme in zip(axes, SCHEMES):
        sub = grid[grid["분할방식"] == scheme]
        for leaf in MIN_LEAVES:
            s = sub[sub["min_samples_leaf"] == leaf].set_index("max_depth")
            s = s.reindex(DEPTH_LABELS)
            ax.plot(x, s["R2격차"], marker="o", ms=6, lw=2.2,
                    color=LEAF_COLOR[leaf], label=f"leaf={leaf}")
        ax.axhline(0, color="gray", lw=1.2, ls=":")
        ax.set_xticks(x, DEPTH_LABELS)
        ax.set_xlabel("max_depth")
        # U+2212 는 Malgun Gothic 에 없어 깨지므로 ASCII 하이픈을 쓴다
        ax.set_ylabel("train R2 - test R2")
        ax.set_title(f"{scheme}  (최대 격차 {sub['R2격차'].max():.4f})\n값이 클수록 과적합",
                     fontsize=12)
        ax.legend(fontsize=9)

    fig.suptitle("과적합 격차 — 깊이를 늘릴수록 학습만 좋아지는 정도\n"
                 "주의: 좌우 y축 눈금 범위가 다르다 (그룹 분할이 약 20배 크다)",
                 fontsize=13, y=1.04)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "14_overfit_gap.png")
    plt.close(fig)


def main() -> None:
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TBL_DIR.mkdir(parents=True, exist_ok=True)

    df, X = load_and_encode()
    print(f"데이터 {len(df):,}행 | 입력변수 {X.shape[1]}개")
    print(f"격자: max_depth {DEPTH_LABELS} x min_samples_leaf {MIN_LEAVES} "
          f"x 분할 {len(SCHEMES)}가지 = {len(MAX_DEPTHS) * len(MIN_LEAVES) * len(SCHEMES)}회\n")

    grid = run_grid(df, X)
    grid.to_csv(TBL_DIR / "hyperparam_grid.csv", index=False, encoding="utf-8-sig")

    cols = ["max_depth", "min_samples_leaf", "잎 개수", "실제 깊이",
            "train_MAE", "test_MAE", "train_RMSE", "test_RMSE",
            "train_R2", "test_R2", "R2격차"]
    with pd.option_context("display.width", 240, "display.max_columns", None,
                           "display.max_rows", None):
        for scheme in SCHEMES:
            print("\n" + "=" * 122)
            print(f"[{scheme}] 격자 결과")
            print("=" * 122)
            print(grid[grid["분할방식"] == scheme][cols].to_string(index=False))

        pivot = grid.pivot_table(index="max_depth", columns=["분할방식", "min_samples_leaf"],
                                 values="test_R2", sort=False)
        print("\n" + "=" * 122)
        print("test R² 피벗")
        print("=" * 122)
        print(pivot.to_string())
        pivot.to_csv(TBL_DIR / "hyperparam_grid_pivot.csv", encoding="utf-8-sig")

        print("\n" + "=" * 122)
        print("분할방식별 최고 test R² 조합")
        print("=" * 122)
        best = grid.loc[grid.groupby("분할방식")["test_R2"].idxmax()]
        print(best[["분할방식", "max_depth", "min_samples_leaf", "잎 개수",
                    "test_MAE", "test_RMSE", "test_R2", "R2격차"]].to_string(index=False))

    fig_train_test(grid)
    fig_gap(grid)
    print(f"\n저장: {TBL_DIR / 'hyperparam_grid.csv'}")
    print(f"저장: {TBL_DIR / 'hyperparam_grid_pivot.csv'}")
    print(f"저장: {FIG_DIR / '13_grid_train_test.png'}")
    print(f"저장: {FIG_DIR / '14_overfit_gap.png'}")


if __name__ == "__main__":
    main()

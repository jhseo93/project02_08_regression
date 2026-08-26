"""max_depth 2 / 4 / 6 성능 비교.

다른 조건은 06 과 동일하게 고정한다.
  범주형 원-핫, 수치형 원본, min_samples_leaf=500, random_state=42

깊이별로 확인하는 것
  - 테스트 지표 개선폭과 수확체감 지점
  - train/test 격차 (과적합 시작 지점)
  - 실제로 사용된 입력변수 수와 변수 중요도 변화
  - 두 분할 방식(무작위 vs flight 그룹)의 점수 격차 변화

산출물:
  outputs/tables/depth_comparison.csv            깊이별 지표
  outputs/tables/depth_feature_importance.csv    깊이별 변수 중요도
  outputs/tables/depth_comparison_by_class.csv   등급별 오차 분해
  outputs/figures/08_depth_comparison.png        성능 곡선
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree

SPLIT_PATH = Path("data/processed/flight_prices_with_splits.csv")
FIG_DIR = Path("outputs/figures")
TBL_DIR = Path("outputs/tables")

RANDOM_STATE = 42
MIN_SAMPLES_LEAF = 500
DEPTHS = [2, 4, 6]

TARGET = "price"
CATEGORICAL = ["airline", "source_city", "destination_city",
               "departure_time", "arrival_time", "class", "stops"]
NUMERIC = ["duration", "days_left"]
SCHEMES = ["split_group", "split_random"]
SCHEME_STYLE = {"split_group": ("#C44E52", "o", "flight 그룹 분할"),
                "split_random": ("#4C72B0", "s", "무작위 분할")}


def setup_style() -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3


def load_splits() -> pd.DataFrame:
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"분할 파일이 없습니다: {SPLIT_PATH}\n"
            "먼저 analysis/04_train_test_split.py 를 실행하세요."
        )
    return pd.read_csv(SPLIT_PATH)


def encode(df: pd.DataFrame) -> pd.DataFrame:
    dummies = pd.get_dummies(df[CATEGORICAL], columns=CATEGORICAL, drop_first=False)
    return pd.concat([dummies.astype(int), df[NUMERIC]], axis=1)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": root_mean_squared_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred)}


def run_all(df: pd.DataFrame, X: pd.DataFrame) -> tuple[list, list, list, dict]:
    """깊이 x 분할방식 조합을 모두 학습하고 결과를 모은다."""
    metric_rows, importance_rows, class_rows = [], [], []
    trees = {}

    for scheme in SCHEMES:
        is_train = df[scheme] == "train"
        X_tr, X_te = X[is_train], X[~is_train]
        y_tr = df.loc[is_train, TARGET].to_numpy(float)
        y_te = df.loc[~is_train, TARGET].to_numpy(float)
        test_class = df.loc[~is_train, "class"].to_numpy()

        base_rmse = metrics(y_te, np.full_like(y_te, y_tr.mean()))["RMSE"]
        base_mae = min(metrics(y_te, np.full_like(y_te, y_tr.mean()))["MAE"],
                       metrics(y_te, np.full_like(y_te, np.median(y_tr)))["MAE"])

        for depth in DEPTHS:
            tree = DecisionTreeRegressor(max_depth=depth,
                                         min_samples_leaf=MIN_SAMPLES_LEAF,
                                         random_state=RANDOM_STATE)
            tree.fit(X_tr, y_tr)
            trees[(scheme, depth)] = (tree, X_tr)

            pred_te, pred_tr = tree.predict(X_te), tree.predict(X_tr)
            te_m, tr_m = metrics(y_te, pred_te), metrics(y_tr, pred_tr)
            used = int((tree.feature_importances_ > 0).sum())

            metric_rows.append({
                "분할방식": scheme, "max_depth": depth,
                "잎 개수": tree.get_n_leaves(), "사용된 변수": used,
                "test_MAE": round(te_m["MAE"], 1),
                "test_RMSE": round(te_m["RMSE"], 1),
                "test_R2": round(te_m["R2"], 4),
                "train_R2": round(tr_m["R2"], 4),
                "R2격차(train-test)": round(tr_m["R2"] - te_m["R2"], 4),
                "MAE개선(vs기준)%": round((base_mae - te_m["MAE"]) / base_mae * 100, 1),
                "RMSE개선(vs기준)%": round((base_rmse - te_m["RMSE"]) / base_rmse * 100, 1),
            })

            for name, imp in zip(X.columns, tree.feature_importances_):
                if imp > 0:
                    importance_rows.append({"분할방식": scheme, "max_depth": depth,
                                            "변수": name, "중요도": round(imp, 5)})

            for cls in ["Economy", "Business"]:
                mask = test_class == cls
                m = metrics(y_te[mask], pred_te[mask])
                class_rows.append({
                    "분할방식": scheme, "max_depth": depth, "class": cls,
                    "n_test": int(mask.sum()),
                    "MAE": round(m["MAE"], 1), "RMSE": round(m["RMSE"], 1),
                    "MAPE(%)": round(np.mean(np.abs(y_te[mask] - pred_te[mask])
                                             / y_te[mask]) * 100, 2),
                })

    return metric_rows, importance_rows, class_rows, trees


def fig_curves(metric_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    specs = [("test_MAE", "테스트 MAE (INR)", "낮을수록 좋음"),
             ("test_RMSE", "테스트 RMSE (INR)", "낮을수록 좋음"),
             ("test_R2", "테스트 R²", "높을수록 좋음")]

    for ax, (col, title, note) in zip(axes, specs):
        for scheme in SCHEMES:
            sub = metric_df[metric_df["분할방식"] == scheme].sort_values("max_depth")
            color, marker, label = SCHEME_STYLE[scheme]
            ax.plot(sub["max_depth"], sub[col], marker=marker, color=color,
                    lw=2, ms=8, label=label)
            for x, v in zip(sub["max_depth"], sub[col]):
                ax.annotate(f"{v:,.0f}" if col != "test_R2" else f"{v:.4f}",
                            (x, v), textcoords="offset points", xytext=(0, 9),
                            ha="center", fontsize=8)
        ax.set_xticks(DEPTHS)
        ax.set_xlabel("max_depth")
        ax.set_title(f"{title}\n({note})", fontsize=11)
        ax.legend(fontsize=9)

    # train/test 격차 패널을 R2 축에 겹쳐 표시
    ax = axes[2]
    for scheme in SCHEMES:
        sub = metric_df[metric_df["분할방식"] == scheme].sort_values("max_depth")
        color, _, label = SCHEME_STYLE[scheme]
        ax.plot(sub["max_depth"], sub["train_R2"], ls="--", lw=1.4, color=color,
                alpha=0.6, label=f"{label} (train)")
    ax.legend(fontsize=8)

    fig.suptitle(f"max_depth 별 성능 (min_samples_leaf={MIN_SAMPLES_LEAF}, "
                 f"random_state={RANDOM_STATE})", fontsize=14, y=1.04)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_depth_comparison.png")
    plt.close(fig)


# 깊이별 그림 크기와 파일명. 깊이 2 그림은 06 단계에서 이미 생성하므로 제외한다.
TREE_FIG_SPEC = {
    4: {"figsize": (30, 11), "fontsize": 8, "name": "09_tree_depth4.png"},
    6: {"figsize": (58, 15), "fontsize": 5, "name": "10_tree_depth6.png"},
}


def fig_tree(trees: dict, metric_df: pd.DataFrame, depth: int,
             scheme: str = "split_group") -> Path:
    """단일 나무를 06 단계와 같은 형식으로 그린다."""
    tree, X_tr = trees[(scheme, depth)]
    spec = TREE_FIG_SPEC[depth]
    row = metric_df[(metric_df["분할방식"] == scheme)
                    & (metric_df["max_depth"] == depth)].iloc[0]

    fig, ax = plt.subplots(figsize=spec["figsize"])
    ax.grid(False)
    plot_tree(tree, feature_names=list(X_tr.columns),
              filled=True, rounded=True, precision=2, fontsize=spec["fontsize"],
              impurity=True, proportion=False, ax=ax)
    ax.set_title(
        f"의사결정나무 회귀 (max_depth={depth}, min_samples_leaf={MIN_SAMPLES_LEAF}, "
        f"random_state={RANDOM_STATE}) — {scheme}\n"
        f"잎 {int(row['잎 개수'])}개 · 사용 변수 {int(row['사용된 변수'])}개 · "
        f"테스트 MAE {row['test_MAE']:,.0f} / RMSE {row['test_RMSE']:,.0f} / "
        f"R2 {row['test_R2']:.4f}",
        fontsize=15, pad=20)
    fig.tight_layout()

    path = FIG_DIR / spec["name"]
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TBL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_splits()
    X = encode(df)
    print(f"데이터 {len(df):,}행 | 입력변수 {X.shape[1]}개")
    print(f"고정 조건: min_samples_leaf={MIN_SAMPLES_LEAF}, random_state={RANDOM_STATE}")
    print(f"비교 깊이: {DEPTHS}\n")

    metric_rows, importance_rows, class_rows, trees = run_all(df, X)
    metric_df = pd.DataFrame(metric_rows)
    imp_df = pd.DataFrame(importance_rows)
    cls_df = pd.DataFrame(class_rows)

    with pd.option_context("display.width", 240, "display.max_columns", None):
        print("=" * 118)
        print("깊이별 테스트 성능")
        print("=" * 118)
        print(metric_df.to_string(index=False))

        print("\n" + "=" * 118)
        print("등급별 오차 분해")
        print("=" * 118)
        print(cls_df.to_string(index=False))

        print("\n" + "=" * 118)
        print("변수 중요도 (split_group, 깊이별 상위 8개)")
        print("=" * 118)
        for depth in DEPTHS:
            sub = (imp_df[(imp_df["분할방식"] == "split_group")
                          & (imp_df["max_depth"] == depth)]
                   .sort_values("중요도", ascending=False).head(8))
            print(f"\n[max_depth={depth}] 사용된 변수 "
                  f"{(imp_df['max_depth'].eq(depth) & imp_df['분할방식'].eq('split_group')).sum()}개")
            print(sub[["변수", "중요도"]].to_string(index=False))

    print("\n" + "=" * 118)
    print("max_depth=4 텍스트 규칙 (split_group)")
    print("=" * 118)
    tree, X_tr = trees[("split_group", 4)]
    print(export_text(tree, feature_names=list(X_tr.columns), decimals=2))

    metric_df.to_csv(TBL_DIR / "depth_comparison.csv", index=False, encoding="utf-8-sig")
    imp_df.to_csv(TBL_DIR / "depth_feature_importance.csv", index=False,
                  encoding="utf-8-sig")
    cls_df.to_csv(TBL_DIR / "depth_comparison_by_class.csv", index=False,
                  encoding="utf-8-sig")

    rules = []
    for depth in DEPTHS:
        t, xt = trees[("split_group", depth)]
        rules.append(f"{'=' * 96}\n[split_group] max_depth={depth}\n{'=' * 96}\n"
                     + export_text(t, feature_names=list(xt.columns), decimals=2))
    (TBL_DIR / "depth_comparison_rules.txt").write_text("\n".join(rules), encoding="utf-8")

    fig_curves(metric_df)
    for depth in TREE_FIG_SPEC:
        path = fig_tree(trees, metric_df, depth)
        print(f"저장: {path}  ({path.stat().st_size / 1024:,.0f} KB)")

    print(f"\n저장: {TBL_DIR / 'depth_comparison.csv'}")
    print(f"저장: {TBL_DIR / 'depth_feature_importance.csv'}")
    print(f"저장: {TBL_DIR / 'depth_comparison_by_class.csv'}")
    print(f"저장: {TBL_DIR / 'depth_comparison_rules.txt'}")
    print(f"저장: {FIG_DIR / '08_depth_comparison.png'}")


if __name__ == "__main__":
    main()

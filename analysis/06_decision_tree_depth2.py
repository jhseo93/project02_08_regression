"""의사결정나무 회귀 (max_depth=2) 학습 및 평가.

전처리
  범주형 -> 원-핫 인코딩 (drop_first=False. 트리는 다중공선성 영향이 없고,
            기준 범주를 없애면 분할 규칙 해석이 어려워지므로 전 수준을 유지한다)
  수치형 -> 변환 없이 그대로 사용

하이퍼파라미터
  max_depth        = 2
  min_samples_leaf = 500
  random_state     = 42

산출물:
  outputs/tables/tree_depth2_metrics.csv   기준모델 대비 개선폭 포함 지표
  outputs/tables/tree_depth2_rules.txt     텍스트 규칙
  outputs/figures/07_tree_depth2.png       나무 그림
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
MAX_DEPTH = 2
MIN_SAMPLES_LEAF = 500

TARGET = "price"
CATEGORICAL = ["airline", "source_city", "destination_city",
               "departure_time", "arrival_time", "class", "stops"]
NUMERIC = ["duration", "days_left"]
SCHEMES = ["split_group", "split_random"]


def setup_style() -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.bbox"] = "tight"


def load_splits() -> pd.DataFrame:
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"분할 파일이 없습니다: {SPLIT_PATH}\n"
            "먼저 analysis/04_train_test_split.py 를 실행하세요."
        )
    return pd.read_csv(SPLIT_PATH)


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """범주형은 원-핫, 수치형은 그대로. 전체에 한 번 적용해 열 순서를 통일한다."""
    dummies = pd.get_dummies(df[CATEGORICAL], columns=CATEGORICAL, drop_first=False)
    return pd.concat([dummies.astype(int), df[NUMERIC]], axis=1)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def fit_and_score(df: pd.DataFrame, X: pd.DataFrame, scheme: str) -> dict:
    """한 분할 방식으로 학습하고 테스트 지표와 기준모델 대비 개선폭을 계산한다."""
    is_train = df[scheme] == "train"
    X_tr, X_te = X[is_train], X[~is_train]
    y_tr = df.loc[is_train, TARGET].to_numpy(float)
    y_te = df.loc[~is_train, TARGET].to_numpy(float)

    tree = DecisionTreeRegressor(
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
    )
    tree.fit(X_tr, y_tr)

    tree_m = metrics(y_te, tree.predict(X_te))
    mean_m = metrics(y_te, np.full_like(y_te, y_tr.mean()))
    median_m = metrics(y_te, np.full_like(y_te, np.median(y_tr)))

    return {
        "scheme": scheme, "tree": tree, "X_tr": X_tr,
        "n_train": int(is_train.sum()), "n_test": int((~is_train).sum()),
        "y_te": y_te, "y_pred": tree.predict(X_te),
        "tree_m": tree_m, "mean_m": mean_m, "median_m": median_m,
        "train_m": metrics(y_tr, tree.predict(X_tr)),
    }


def comparison_table(res: dict) -> pd.DataFrame:
    """기준모델 / 트리 / 개선폭을 한 표로."""
    rows = []
    for label, m in [("기준: 평균 예측", res["mean_m"]),
                     ("기준: 중앙값 예측", res["median_m"]),
                     ("의사결정나무 (depth=2)", res["tree_m"])]:
        rows.append({"분할방식": res["scheme"], "모델": label,
                     "MAE": round(m["MAE"], 1), "RMSE": round(m["RMSE"], 1),
                     "R2": round(m["R2"], 4)})

    tree_m, mean_m, median_m = res["tree_m"], res["mean_m"], res["median_m"]
    best_mae = min(mean_m["MAE"], median_m["MAE"])
    best_rmse = min(mean_m["RMSE"], median_m["RMSE"])
    rows.append({
        "분할방식": res["scheme"], "모델": "개선폭 (vs 최선 기준모델)",
        "MAE": f"-{best_mae - tree_m['MAE']:,.1f} ({(best_mae - tree_m['MAE']) / best_mae * 100:.1f}%)",
        "RMSE": f"-{best_rmse - tree_m['RMSE']:,.1f} ({(best_rmse - tree_m['RMSE']) / best_rmse * 100:.1f}%)",
        "R2": f"+{tree_m['R2'] - mean_m['R2']:.4f}",
    })
    return pd.DataFrame(rows)


def rules_text(res: dict) -> str:
    """텍스트 규칙 + 잎 노드 요약."""
    tree, X_tr = res["tree"], res["X_tr"]
    body = export_text(tree, feature_names=list(X_tr.columns), decimals=2)

    t = tree.tree_
    lines = ["", "잎 노드 요약:"]
    for i in range(t.node_count):
        if t.children_left[i] == -1:
            n = t.n_node_samples[i]
            lines.append(f"  노드 {i:>2}: n={n:>7,} ({n / t.n_node_samples[0] * 100:5.1f}%)"
                         f"  예측값={t.value[i][0][0]:>10,.0f} INR"
                         f"  노드내 표준편차={np.sqrt(t.impurity[i]):>9,.0f}")
    return body + "\n".join(lines)


def fig_tree(res: dict) -> None:
    fig, ax = plt.subplots(figsize=(19, 8))
    # precision=2: 임계값을 반올림하면 텍스트 규칙과 어긋나므로 소수 2자리까지 표시
    plot_tree(
        res["tree"], feature_names=list(res["X_tr"].columns),
        filled=True, rounded=True, precision=2, fontsize=10,
        impurity=True, proportion=False, ax=ax,
    )
    m = res["tree_m"]
    ax.set_title(
        f"의사결정나무 회귀 (max_depth={MAX_DEPTH}, min_samples_leaf={MIN_SAMPLES_LEAF}, "
        f"random_state={RANDOM_STATE}) — {res['scheme']}\n"
        f"테스트 MAE {m['MAE']:,.0f} / RMSE {m['RMSE']:,.0f} / R2 {m['R2']:.4f}",
        fontsize=13, pad=18,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_tree_depth2.png")
    plt.close(fig)


def main() -> None:
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TBL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_splits()
    X = encode(df)
    print(f"데이터 {len(df):,}행")
    print(f"인코딩 후 입력변수 {X.shape[1]}개 "
          f"(원-핫 {X.shape[1] - len(NUMERIC)}개 + 수치형 {len(NUMERIC)}개)")
    print(f"하이퍼파라미터: max_depth={MAX_DEPTH}, "
          f"min_samples_leaf={MIN_SAMPLES_LEAF}, random_state={RANDOM_STATE}\n")

    results = {s: fit_and_score(df, X, s) for s in SCHEMES}

    tables, rule_blocks = [], []
    for scheme in SCHEMES:
        res = results[scheme]
        table = comparison_table(res)
        tables.append(table)

        print("=" * 96)
        print(f"[{scheme}]  train {res['n_train']:,} / test {res['n_test']:,}")
        print("=" * 96)
        print(table.to_string(index=False))
        print(f"\n  학습 데이터 지표 (과적합 점검): "
              f"MAE {res['train_m']['MAE']:,.1f} / RMSE {res['train_m']['RMSE']:,.1f} "
              f"/ R2 {res['train_m']['R2']:.4f}")

        block = (f"{'=' * 96}\n[{scheme}] max_depth={MAX_DEPTH}, "
                 f"min_samples_leaf={MIN_SAMPLES_LEAF}, random_state={RANDOM_STATE}\n"
                 f"{'=' * 96}\n{rules_text(res)}\n")
        rule_blocks.append(block)
        print("\n" + block)

    pd.concat(tables).to_csv(TBL_DIR / "tree_depth2_metrics.csv",
                             index=False, encoding="utf-8-sig")
    (TBL_DIR / "tree_depth2_rules.txt").write_text("\n".join(rule_blocks), encoding="utf-8")
    fig_tree(results["split_group"])

    print(f"저장: {TBL_DIR / 'tree_depth2_metrics.csv'}")
    print(f"저장: {TBL_DIR / 'tree_depth2_rules.txt'}")
    print(f"저장: {FIG_DIR / '07_tree_depth2.png'}  (split_group 기준)")


if __name__ == "__main__":
    main()

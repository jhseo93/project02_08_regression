"""기준 모델(baseline) 평가.

설명변수를 전혀 쓰지 않고 학습 데이터의 대표값 하나만으로 모든 테스트 행을 예측한다.
이후 만드는 의사결정나무는 최소한 이 점수보다 나아야 의미가 있다.

  mean_baseline   : 학습 데이터 price 평균을 모든 테스트 행에 예측
  median_baseline : 학습 데이터 price 중앙값을 모든 테스트 행에 예측

지표
  MAE  = mean(|y - yhat|)                     단위 INR, 이상치에 덜 민감
  RMSE = sqrt(mean((y - yhat)^2))             단위 INR, 큰 오차에 민감
  R^2  = 1 - SS_res / SS_tot                  SS_tot 은 '테스트 데이터의 평균' 기준

산출물:
  outputs/tables/baseline_metrics.csv          분할 x 기준모델 별 지표
  outputs/tables/baseline_metrics_by_class.csv 등급별 분해
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

SPLIT_PATH = Path("data/processed/flight_prices_with_splits.csv")
TBL_DIR = Path("outputs/tables")

TARGET = "price"
SPLIT_SCHEMES = ["split_random", "split_group"]


def load_splits() -> pd.DataFrame:
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"분할 파일이 없습니다: {SPLIT_PATH}\n"
            "먼저 analysis/04_train_test_split.py 를 실행하세요."
        )
    df = pd.read_csv(SPLIT_PATH)
    missing = [c for c in SPLIT_SCHEMES if c not in df.columns]
    if missing:
        raise ValueError(f"분할 라벨 열이 없습니다: {missing}")
    return df


def evaluate(y_true: np.ndarray, prediction: float) -> dict:
    """상수 예측값 하나에 대한 지표를 계산한다."""
    y_pred = np.full_like(y_true, prediction, dtype=float)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def baseline_table(df: pd.DataFrame) -> pd.DataFrame:
    """분할 방식 x 기준모델 조합의 테스트 지표."""
    rows = []
    for scheme in SPLIT_SCHEMES:
        train = df.loc[df[scheme] == "train", TARGET].to_numpy(float)
        test = df.loc[df[scheme] == "test", TARGET].to_numpy(float)

        for name, value in [("평균 예측", train.mean()), ("중앙값 예측", np.median(train))]:
            metrics = evaluate(test, value)
            rows.append({
                "분할방식": scheme, "기준모델": name,
                "학습 대표값": round(value, 2),
                "n_train": len(train), "n_test": len(test),
                "MAE": round(metrics["MAE"], 1),
                "RMSE": round(metrics["RMSE"], 1),
                "R2": round(metrics["R2"], 5),
            })
    return pd.DataFrame(rows)


def by_class_table(df: pd.DataFrame) -> pd.DataFrame:
    """같은 상수 예측을 등급별로 나눠 평가한다 (오차가 어디서 나오는지 확인)."""
    rows = []
    for scheme in SPLIT_SCHEMES:
        train = df.loc[df[scheme] == "train", TARGET].to_numpy(float)
        test_df = df[df[scheme] == "test"]

        for name, value in [("평균 예측", train.mean()), ("중앙값 예측", np.median(train))]:
            for cls in ["Economy", "Business"]:
                y = test_df.loc[test_df["class"] == cls, TARGET].to_numpy(float)
                metrics = evaluate(y, value)
                rows.append({
                    "분할방식": scheme, "기준모델": name, "class": cls,
                    "n_test": len(y),
                    "실제 평균": round(y.mean(), 1),
                    "예측값": round(value, 1),
                    "MAE": round(metrics["MAE"], 1),
                    "RMSE": round(metrics["RMSE"], 1),
                })
    return pd.DataFrame(rows)


def main() -> None:
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    df = load_splits()
    print(f"분할 데이터 {len(df):,}행 로드\n")

    main_table = baseline_table(df)
    cls_table = by_class_table(df)

    main_table.to_csv(TBL_DIR / "baseline_metrics.csv", index=False, encoding="utf-8-sig")
    cls_table.to_csv(TBL_DIR / "baseline_metrics_by_class.csv", index=False,
                     encoding="utf-8-sig")

    with pd.option_context("display.width", 220, "display.max_columns", None):
        print("=" * 100)
        print("기준 모델 테스트 성능")
        print("=" * 100)
        print(main_table.to_string(index=False))
        print("\n" + "=" * 100)
        print("등급별 분해 — 오차가 어디서 발생하는가")
        print("=" * 100)
        print(cls_table.to_string(index=False))

    print("\n" + "=" * 100)
    print("참고: 테스트 데이터 자체 통계 (R2 의 기준선)")
    print("=" * 100)
    for scheme in SPLIT_SCHEMES:
        te = df.loc[df[scheme] == "test", TARGET]
        tr = df.loc[df[scheme] == "train", TARGET]
        print(f"[{scheme}] train 평균 {tr.mean():>9,.1f} / 중앙값 {tr.median():>8,.1f}"
              f" | test 평균 {te.mean():>9,.1f} / 중앙값 {te.median():>8,.1f}"
              f" / 표준편차 {te.std():>9,.1f}")

    print(f"\n저장: {TBL_DIR / 'baseline_metrics.csv'}")
    print(f"저장: {TBL_DIR / 'baseline_metrics_by_class.csv'}")


if __name__ == "__main__":
    main()

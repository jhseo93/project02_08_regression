"""학습/테스트 데이터셋 분할. 난수는 RANDOM_STATE=42 로 고정한다.

두 가지 분할을 모두 만들어 하나의 파일에 라벨 열로 기록한다.
  split_random : 무작위 분할 (class 층화). 운영 상황 = '기존 편의 새 조회 건' 예측
  split_group  : flight 그룹 분할.        일반화 상황 = '처음 보는 편' 예측

입력에서 제외하는 열
  id     : 원본 Unnamed: 0. 등급 순 정렬된 행 번호라 class 의 대리변수가 됨(누수).
           추적용으로 보존만 하고 학습에는 쓰지 않는다.
  flight : 고유값 1,561개 준식별자. 접두어가 airline 과 1:1 중복.
           그룹 분할의 키로만 사용한다.

산출물:
  data/processed/flight_prices_with_splits.csv  원본 + id/split 라벨
  outputs/tables/split_summary.csv              분할별 구성 검증표
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

RAW_PATH = Path("data/raw/flight_prices.csv")
PROCESSED_PATH = Path("data/processed/flight_prices_with_splits.csv")
TBL_DIR = Path("outputs/tables")

RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET = "price"
ID_COLUMN = "id"
GROUP_COLUMN = "flight"
EXCLUDE_FROM_FEATURES = [ID_COLUMN, GROUP_COLUMN, TARGET, "split_random", "split_group"]


def load_raw() -> pd.DataFrame:
    """원본을 읽고 식별자 열 이름만 정리한다. 값은 수정하지 않는다."""
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"원본 데이터를 찾을 수 없습니다: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)
    if "Unnamed: 0" not in df.columns:
        raise ValueError("원본에 'Unnamed: 0' 열이 없습니다. 파일을 확인하세요.")
    return df.rename(columns={"Unnamed: 0": ID_COLUMN})


def add_random_split(df: pd.DataFrame) -> pd.Series:
    """무작위 분할. class 비율을 유지하도록 층화한다."""
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["class"],
    )
    label = pd.Series("train", index=df.index, name="split_random")
    label.loc[test_idx] = "test"
    return label


def add_group_split(df: pd.DataFrame) -> pd.Series:
    """flight 단위 분할. 같은 편명이 train/test 에 동시에 들어가지 않는다."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_pos, test_pos = next(splitter.split(df, groups=df[GROUP_COLUMN]))
    label = pd.Series("train", index=df.index, name="split_group")
    label.iloc[test_pos] = "test"
    return label


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """분할별 크기, 목표변수 분포, 범주 구성비를 한 표로 모은다."""
    rows = []
    for scheme in ["split_random", "split_group"]:
        for part in ["train", "test"]:
            sub = df[df[scheme] == part]
            row = {
                "분할방식": scheme, "구분": part,
                "n": len(sub), "비율(%)": round(len(sub) / len(df) * 100, 2),
                "고유 flight": sub[GROUP_COLUMN].nunique(),
                "price_평균": round(sub[TARGET].mean(), 1),
                "price_중앙값": round(sub[TARGET].median(), 1),
                "price_표준편차": round(sub[TARGET].std(), 1),
            }
            for level in ["Economy", "Business"]:
                row[f"{level}(%)"] = round((sub["class"] == level).mean() * 100, 2)
            rows.append(row)
    return pd.DataFrame(rows)


def check_leakage(df: pd.DataFrame) -> dict:
    """분할 방식별로 train/test 에 겹치는 flight 이 있는지 확인한다."""
    result = {}
    for scheme in ["split_random", "split_group"]:
        tr = set(df.loc[df[scheme] == "train", GROUP_COLUMN])
        te = set(df.loc[df[scheme] == "test", GROUP_COLUMN])
        result[scheme] = {
            "train_flight": len(tr), "test_flight": len(te),
            "겹치는_flight": len(tr & te),
            "test 중 train 에도 있는 편의 행 수": int(
                df[(df[scheme] == "test") & (df[GROUP_COLUMN].isin(tr))].shape[0]
            ),
        }
    return result


def check_category_coverage(df: pd.DataFrame) -> list[str]:
    """test 에 빠진 범주 수준이 있는지 확인한다. 있으면 경고 문자열로 반환."""
    warnings = []
    cat_cols = [c for c in df.columns
                if c not in EXCLUDE_FROM_FEATURES and df[c].dtype == "object"
                or (c not in EXCLUDE_FROM_FEATURES and str(df[c].dtype) == "str")]
    for scheme in ["split_random", "split_group"]:
        te = df[df[scheme] == "test"]
        tr = df[df[scheme] == "train"]
        for col in cat_cols:
            missing_te = set(df[col].unique()) - set(te[col].unique())
            missing_tr = set(df[col].unique()) - set(tr[col].unique())
            if missing_te:
                warnings.append(f"  [{scheme}] test 에 없는 {col} 수준: {sorted(missing_te)}")
            if missing_tr:
                warnings.append(f"  [{scheme}] train 에 없는 {col} 수준: {sorted(missing_tr)}")
    return warnings


def main() -> None:
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    print(f"원본 {len(df):,}행 x {df.shape[1]}열 로드 (Unnamed: 0 -> {ID_COLUMN} 이름만 변경)")
    print(f"난수 고정 RANDOM_STATE={RANDOM_STATE}, TEST_SIZE={TEST_SIZE}\n")

    df["split_random"] = add_random_split(df)
    df["split_group"] = add_group_split(df)

    features = [c for c in df.columns if c not in EXCLUDE_FROM_FEATURES]
    print(f"목표변수 : {TARGET}")
    print(f"입력변수 {len(features)}개 : {features}")
    print(f"보존(입력 제외) : {ID_COLUMN}(추적용), {GROUP_COLUMN}(그룹키)\n")

    summary = summarize(df)
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print("=" * 110)
        print("분할 구성 요약")
        print("=" * 110)
        print(summary.to_string(index=False))

    print("\n" + "=" * 110)
    print("편명 누수 점검 (train/test 에 같은 flight 이 걸쳐 있는가)")
    print("=" * 110)
    for scheme, info in check_leakage(df).items():
        print(f"[{scheme}]")
        for k, v in info.items():
            print(f"    {k:<32} {v:,}")

    print("\n" + "=" * 110)
    print("범주 수준 커버리지 점검")
    print("=" * 110)
    warns = check_category_coverage(df)
    print("\n".join(warns) if warns else "  train/test 양쪽에 모든 범주 수준이 존재합니다.")

    summary.to_csv(TBL_DIR / "split_summary.csv", index=False, encoding="utf-8-sig")
    df.to_csv(PROCESSED_PATH, index=False, encoding="utf-8")

    size_mb = PROCESSED_PATH.stat().st_size / 1024 ** 2
    print(f"\n저장: {PROCESSED_PATH} ({size_mb:.1f} MB)")
    print(f"저장: {TBL_DIR / 'split_summary.csv'}")
    print("\n사용 예:")
    print("    df = pd.read_csv('data/processed/flight_prices_with_splits.csv')")
    print("    train = df[df.split_group == 'train']")
    print("    test  = df[df.split_group == 'test']")


if __name__ == "__main__":
    main()

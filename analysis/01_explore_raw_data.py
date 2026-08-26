"""원본 항공권 데이터 탐색: 구조, 자료형, 결측치, 중복, 범주형 고유값 확인."""

from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/flight_prices.csv")
ID_COLUMN = "Unnamed: 0"


def load_raw(path: Path) -> pd.DataFrame:
    """원본 CSV를 가공 없이 그대로 읽는다."""
    if not path.exists():
        raise FileNotFoundError(f"원본 데이터를 찾을 수 없습니다: {path}")
    return pd.read_csv(path)


def print_section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def report_structure(df: pd.DataFrame) -> None:
    print_section("1. 데이터 구조 (행 x 열)")
    print(f"행 수: {len(df):,}")
    print(f"열 수: {df.shape[1]}")
    print(f"메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    print_section("2. 열별 자료형 및 비결측 개수")
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "null": df.isna().sum(),
        "null_pct": (df.isna().mean() * 100).round(3),
        "nunique": df.nunique(dropna=False),
    })
    print(summary.to_string())


def report_head(df: pd.DataFrame) -> None:
    print_section("3. 상위 5개 행")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.head(5).to_string())


def report_missing(df: pd.DataFrame) -> None:
    print_section("4. 결측치 점검")
    null_counts = df.isna().sum()
    missing = null_counts[null_counts > 0]
    if missing.empty:
        print("결측치 없음 (모든 열의 NaN 개수 = 0)")
    else:
        print(missing.to_string())

    print("\n[문자열 열의 빈 문자열/공백 점검]")
    object_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        blank = df[col].astype("string").str.strip().eq("").sum()
        print(f"  {col:<20} 빈 문자열: {blank}")


def report_duplicates(df: pd.DataFrame) -> None:
    print_section("5. 중복 행 점검")
    print(f"완전 중복 행 (모든 열 기준): {df.duplicated().sum():,}")

    if ID_COLUMN in df.columns:
        without_id = df.drop(columns=[ID_COLUMN])
        print(f"식별자 열 제외 후 중복 행:   {without_id.duplicated().sum():,}")
        print(f"식별자 열 고유값 개수:       {df[ID_COLUMN].nunique():,} / {len(df):,}")


def report_categorical(df: pd.DataFrame) -> None:
    print_section("6. 범주형 열별 고유값")
    cat_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in cat_cols:
        counts = df[col].value_counts(dropna=False)
        print(f"\n[{col}] 고유값 {df[col].nunique(dropna=False):,}개")
        if len(counts) <= 15:
            pct = (counts / len(df) * 100).round(2)
            print(pd.DataFrame({"count": counts, "pct": pct}).to_string())
        else:
            print(f"  (고유값이 많아 상위 5개만 표시)")
            print(counts.head(5).to_string())


def report_numeric(df: pd.DataFrame) -> None:
    print_section("7. 수치형 열 기술통계")
    num_cols = df.select_dtypes(include="number").columns
    with pd.option_context("display.float_format", "{:,.2f}".format):
        print(df[num_cols].describe().T.to_string())


def main() -> None:
    df = load_raw(RAW_PATH)
    report_structure(df)
    report_head(df)
    report_missing(df)
    report_duplicates(df)
    report_categorical(df)
    report_numeric(df)


if __name__ == "__main__":
    main()

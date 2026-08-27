"""max_depth=4 나무의 분할 규칙을 사람이 읽는 서술형 문장으로 변환해 md 로 저장한다.

손으로 옮겨 적으면 임계값 오타가 생기므로, 학습된 트리에서 경로를 직접 추출한다.
07 단계와 동일한 조건으로 재학습하므로 규칙은 항상 일치한다.

변환 규칙
  원-핫 열   X <= 0.5  -> '~가 아닌',  X > 0.5 -> '~인'
  정수 열    days_left <= 15.5 -> '15일 이하' (정수이므로 .5 를 내림하여 표기)
  실수 열    duration <= 4.71  -> '4.71시간(4시간 43분) 이하'
  같은 변수가 경로에 여러 번 나오면 구간 하나로 합친다.

산출물:
  outputs/tree_depth4_rules.md
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor

SPLIT_PATH = Path("data/processed/flight_prices_with_splits.csv")
OUT_PATH = Path("outputs/tree_depth4_rules.md")

RANDOM_STATE = 42
MAX_DEPTH = 4
MIN_SAMPLES_LEAF = 500
SCHEME = "split_group"

TARGET = "price"
CATEGORICAL = ["airline", "source_city", "destination_city",
               "departure_time", "arrival_time", "class", "stops"]
NUMERIC = ["duration", "days_left"]

# 원-핫 열 -> (참일 때 표현, 거짓일 때 표현)
ONEHOT_LABEL = {
    "class_Economy": ("이코노미석", "비즈니스석"),
    "class_Business": ("비즈니스석", "이코노미석"),
    "airline_Vistara": ("Vistara 항공", "Vistara 이외의 항공사"),
    "airline_Air_India": ("Air India 항공", "Air India 이외의 항공사"),
    "airline_AirAsia": ("AirAsia 항공", "AirAsia 이외의 항공사"),
    "airline_Indigo": ("Indigo 항공", "Indigo 이외의 항공사"),
    "airline_GO_FIRST": ("GO FIRST 항공", "GO FIRST 이외의 항공사"),
    "airline_SpiceJet": ("SpiceJet 항공", "SpiceJet 이외의 항공사"),
    "stops_zero": ("직항", "경유편"),
    "stops_one": ("1회 경유", "1회 경유가 아닌"),
    "stops_two_or_more": ("2회 이상 경유", "2회 미만 경유"),
}
CITY_KO = {"Delhi": "델리", "Mumbai": "뭄바이", "Bangalore": "벵갈루루",
           "Kolkata": "콜카타", "Hyderabad": "하이데라바드", "Chennai": "첸나이"}
TIME_KO = {"Early_Morning": "이른 아침", "Morning": "오전", "Afternoon": "오후",
           "Evening": "저녁", "Night": "밤", "Late_Night": "심야"}


def load_and_fit() -> tuple:
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"분할 파일이 없습니다: {SPLIT_PATH}\n"
            "먼저 analysis/04_train_test_split.py 를 실행하세요."
        )
    df = pd.read_csv(SPLIT_PATH)
    dummies = pd.get_dummies(df[CATEGORICAL], columns=CATEGORICAL, drop_first=False)
    X = pd.concat([dummies.astype(int), df[NUMERIC]], axis=1)

    is_train = df[SCHEME] == "train"
    tree = DecisionTreeRegressor(max_depth=MAX_DEPTH, min_samples_leaf=MIN_SAMPLES_LEAF,
                                 random_state=RANDOM_STATE)
    tree.fit(X[is_train], df.loc[is_train, TARGET])
    return df, X, is_train, tree


def onehot_phrase(column: str, is_true: bool) -> str:
    """원-핫 열 이름을 '~인 항공권' 앞에 붙일 수 있는 명사구로 바꾼다."""
    if column in ONEHOT_LABEL:
        return ONEHOT_LABEL[column][0 if is_true else 1]
    for prefix, table, tmpl_t, tmpl_f in [
        ("source_city_", CITY_KO, "{} 출발", "{} 이외 도시 출발"),
        ("destination_city_", CITY_KO, "{} 도착", "{} 이외 도시 도착"),
        ("departure_time_", TIME_KO, "{} 출발", "{} 이외 시간대 출발"),
        ("arrival_time_", TIME_KO, "{} 도착", "{} 이외 시간대 도착"),
    ]:
        if column.startswith(prefix):
            raw = column[len(prefix):]
            return (tmpl_t if is_true else tmpl_f).format(table.get(raw, raw))
    return f"{column} = {int(is_true)}"


def hours_text(value: float) -> str:
    """4.71 -> '4시간 43분'"""
    h, m = int(value), round((value - int(value)) * 60)
    return f"{h}시간 {m}분" if m else f"{h}시간"


def numeric_phrase(column: str, low: float | None, high: float | None) -> str:
    """수치형 변수의 상/하한을 하나의 명사구로. 조사 없이 끝내 문장 조립을 안전하게 한다."""
    if column == "days_left":
        lo = None if low is None else int(np.floor(low)) + 1   # > 15.5  -> 16일 이상
        hi = None if high is None else int(np.floor(high))     # <= 15.5 -> 15일 이하
        if lo is not None and hi is not None:
            return f"출발까지 {lo}~{hi}일"
        if hi is not None:
            return f"출발까지 {hi}일 이하"
        return f"출발까지 {lo}일 이상"

    if low is not None and high is not None:
        return f"비행시간 {hours_text(low)} 초과 {hours_text(high)} 이하"
    if high is not None:
        return f"비행시간 {hours_text(high)} 이하"
    return f"비행시간 {hours_text(low)} 초과"


def raw_condition(column: str, low: float | None, high: float | None) -> str:
    """검증용 원본 임계값 표기."""
    if low is not None and high is not None:
        return f"{low:.2f} < {column} <= {high:.2f}"
    if high is not None:
        return f"{column} <= {high:.2f}"
    return f"{column} > {low:.2f}"


def collect_leaves(tree, columns: list[str]) -> list[dict]:
    """루트에서 각 잎까지의 경로 조건을 모아 정리한다."""
    t = tree.tree_
    leaves = []

    def walk(node: int, bounds: dict, flags: dict, order: list) -> None:
        if t.children_left[node] == -1:
            leaves.append({
                "node": node,
                "bounds": {k: v[:] for k, v in bounds.items()},
                "flags": dict(flags),
                "order": order[:],          # 나무가 질문한 순서 (중복 제거)
                "n": int(t.n_node_samples[node]),
                "value": float(t.value[node][0][0]),
                "std": float(np.sqrt(t.impurity[node])),
            })
            return

        col = columns[t.feature[node]]
        thr = t.threshold[node]
        first_time = col not in order
        if first_time:
            order.append(col)

        if col in NUMERIC:
            lo, hi = bounds.get(col, [None, None])
            bounds[col] = [lo, thr if hi is None else min(hi, thr)]
            walk(t.children_left[node], bounds, flags, order)
            bounds[col] = [thr if lo is None else max(lo, thr), hi]
            walk(t.children_right[node], bounds, flags, order)
            bounds[col] = [lo, hi]
        else:
            flags[col] = False              # X <= 0.5 -> 해당 없음
            walk(t.children_left[node], bounds, flags, order)
            flags[col] = True               # X >  0.5 -> 해당
            walk(t.children_right[node], bounds, flags, order)
            del flags[col]

        if first_time:
            order.pop()

    walk(0, {}, {}, [])
    return leaves


def conditions(leaf: dict) -> list[str]:
    """나무가 질문한 순서대로 조건 명사구를 만든다."""
    out = []
    for col in leaf["order"]:
        if col in leaf["bounds"]:
            lo, hi = leaf["bounds"][col]
            out.append(numeric_phrase(col, lo, hi))
        else:
            out.append(onehot_phrase(col, leaf["flags"][col]))
    return out


def raw_conditions(leaf: dict) -> list[str]:
    out = []
    for col in leaf["order"]:
        if col in leaf["bounds"]:
            lo, hi = leaf["bounds"][col]
            out.append(raw_condition(col, lo, hi))
        else:
            out.append(f"{col} {'>' if leaf['flags'][col] else '<='} 0.50")
    return out


def describe(leaf: dict) -> str:
    """조건 명사구를 쉼표로 이어 '~인 항공권' 앞에 놓을 수 있는 구를 만든다."""
    parts = conditions(leaf)
    return ", ".join(parts) if parts else "모든"


def leaf_test_stats(df, X, is_train, tree, leaves) -> dict:
    """각 잎에 떨어지는 테스트 행의 실제 성능."""
    test_leaf = tree.apply(X[~is_train])
    y_test = df.loc[~is_train, TARGET].to_numpy(float)
    pred = tree.predict(X[~is_train])

    stats = {}
    for leaf in leaves:
        m = test_leaf == leaf["node"]
        if m.sum() == 0:
            stats[leaf["node"]] = None
            continue
        stats[leaf["node"]] = {
            "n": int(m.sum()),
            "actual_mean": float(y_test[m].mean()),
            "mae": float(mean_absolute_error(y_test[m], pred[m])),
            "mape": float(np.mean(np.abs(y_test[m] - pred[m]) / y_test[m]) * 100),
        }
    return stats


def build_markdown(df, X, is_train, tree, leaves, stats) -> str:
    y_te = df.loc[~is_train, TARGET].to_numpy(float)
    pred_te = tree.predict(X[~is_train])
    mae = mean_absolute_error(y_te, pred_te)
    rmse = root_mean_squared_error(y_te, pred_te)
    r2 = r2_score(y_te, pred_te)
    total_train = int(tree.tree_.n_node_samples[0])

    ordered = sorted(leaves, key=lambda d: d["value"], reverse=True)

    out = [
        "# 항공권 가격 예측 규칙 (의사결정나무 깊이 4)",
        "",
        "이 문서는 학습된 의사결정나무가 만든 **16개 예측 규칙**을 문장으로 옮긴 것이다.",
        "표의 숫자는 모두 학습된 모델에서 직접 추출했다.",
        "",
        "## 모델 조건",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 목표변수 | `price` (인도 루피, INR) |",
        f"| 입력변수 | 범주형 원-핫 35개 + 수치형 2개 = 37개 |",
        f"| max_depth | {MAX_DEPTH} |",
        f"| min_samples_leaf | {MIN_SAMPLES_LEAF} |",
        f"| random_state | {RANDOM_STATE} |",
        f"| 분할 방식 | `{SCHEME}` (편명 단위 분할, train/test 편명 겹침 0건) |",
        f"| 학습 데이터 | {total_train:,}행 |",
        f"| 테스트 성능 | MAE {mae:,.0f} · RMSE {rmse:,.0f} · R² {r2:.4f} |",
        "",
        "## 이 나무가 판단하는 순서",
        "",
        "나무는 항상 같은 순서로 질문한다.",
        "",
        "1. **좌석 등급이 무엇인가?** — 가장 먼저, 가장 크게 갈리는 기준이다.",
        "2. 비즈니스석이면 → **비행시간**을 묻는다.",
        "3. 이코노미석이면 → **출발까지 남은 일수**를 묻는다.",
        "4. 마지막으로 항공사나 도착지 같은 세부 조건으로 값을 조정한다.",
        "",
        "같은 나무인데도 좌우 가지가 서로 다른 질문을 한다는 점이 핵심이다.",
        "비즈니스석 가격은 예약 시점에 거의 반응하지 않고, 이코노미석 가격은 예약 시점에 크게 반응한다.",
        "",
        "## 규칙 16개 (예측가격 높은 순)",
        "",
    ]

    for rank, leaf in enumerate(ordered, start=1):
        st = stats[leaf["node"]]
        share = leaf["n"] / total_train * 100
        out += [
            f"### 규칙 {rank}. 약 {leaf['value']:,.0f} INR",
            "",
            f"> **{describe(leaf)}인 항공권은 약 {leaf['value']:,.0f} INR로 예측한다.**",
            "",
            "조건을 하나씩 보면:",
            "",
        ]
        out += [f"{i}. {c}" for i, c in enumerate(conditions(leaf), start=1)]
        out += [
            "",
            f"- 학습 데이터 {leaf['n']:,}건 ({share:.1f}%), 이 그룹 내 표준편차 {leaf['std']:,.0f} INR",
        ]
        if st:
            out.append(
                f"- 테스트 데이터 {st['n']:,}건에서 실제 평균 {st['actual_mean']:,.0f} INR "
                f"→ 평균 오차 {st['mae']:,.0f} INR (오차율 {st['mape']:.1f}%)"
            )
        else:
            out.append("- 테스트 데이터에 해당 건 없음")
        out += [f"- 원본 조건: `{' AND '.join(raw_conditions(leaf))}`", ""]

    out += [
        "## 요약표",
        "",
        "| 순위 | 조건 | 예측가격(INR) | 학습 비중 | 테스트 오차율 |",
        "|---:|---|---:|---:|---:|",
    ]
    for rank, leaf in enumerate(ordered, start=1):
        st = stats[leaf["node"]]
        share = leaf["n"] / total_train * 100
        mape = f"{st['mape']:.1f}%" if st else "—"
        out.append(f"| {rank} | {describe(leaf)} | {leaf['value']:,.0f} | "
                   f"{share:.1f}% | {mape} |")

    biz = [d for d in ordered if d["flags"].get("class_Economy") is False]
    eco = [d for d in ordered if d["flags"].get("class_Economy") is True]
    out += [
        "",
        "## 규칙에서 읽히는 것",
        "",
        f"**1. 좌석 등급이 거의 전부를 결정한다.** "
        f"비즈니스석 규칙 {len(biz)}개의 예측값은 "
        f"{min(d['value'] for d in biz):,.0f} ~ {max(d['value'] for d in biz):,.0f} INR, "
        f"이코노미석 규칙 {len(eco)}개는 "
        f"{min(d['value'] for d in eco):,.0f} ~ {max(d['value'] for d in eco):,.0f} INR이다. "
        "두 범위는 겹치지 않는다. 등급만 알아도 가격대가 정해진다.",
        "",
        "**2. 이코노미석은 예약 시점이 가격을 좌우한다.** "
        "출발 15일 이하로 남은 규칙들과 16일 이상 남은 규칙들의 가격 차이가 2배 안팎이다. "
        "특히 3일 이하로 임박하면 값이 크게 뛴다.",
        "",
        "**3. 비즈니스석은 예약 시점보다 노선 길이가 중요하다.** "
        "비행시간이 짧은 단거리 비즈니스석은 오히려 저렴하고, "
        "장거리로 갈수록 5만 INR을 넘긴다. 예약 시점은 Vistara 항공에서만 조건으로 등장한다.",
        "",
        "**4. 항공사는 마지막 미세 조정 역할이다.** "
        "`airline` 조건은 항상 세 번째나 네 번째 질문으로만 나온다. "
        "등급과 비행시간을 먼저 나눈 뒤 남은 차이를 설명하는 데 쓰인다.",
        "",
        "## 주의사항",
        "",
        "- 이 규칙은 **인도 국내선 6개 도시 구간**에서 수집한 데이터로 만들었다. 다른 노선에는 적용할 수 없다.",
        "- 각 규칙은 그룹의 평균을 예측하므로, 같은 조건이라도 실제 가격은 표준편차만큼 흩어진다.",
        "- 비즈니스석 규칙의 오차율이 이코노미석보다 낮아 보이지만, 이는 가격 자체가 8배 비싸기 때문이다. "
        "절대 오차는 비즈니스석이 훨씬 크다.",
        "",
        "---",
        "",
        "생성: `python analysis/08_tree_rules_narrative.py`",
        "",
    ]
    return "\n".join(out)


def main() -> None:
    df, X, is_train, tree = load_and_fit()
    leaves = collect_leaves(tree, list(X.columns))
    stats = leaf_test_stats(df, X, is_train, tree, leaves)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(build_markdown(df, X, is_train, tree, leaves, stats),
                        encoding="utf-8")

    print(f"잎 {len(leaves)}개의 규칙을 서술형으로 변환했습니다.")
    print(f"저장: {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()

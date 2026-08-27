"""항공권 표시가격 예측 대시보드 (의사결정나무 max_depth=8, min_samples_leaf=200).

실행:
    streamlit run app.py

모델은 앱 시작 시 한 번 학습하고 캐싱한다. 별도의 모델 파일을 저장하지 않으므로
데이터와 코드만 있으면 항상 같은 모델이 재현된다.

학습 조건은 analysis/10_hyperparam_grid.py 의 격자 탐색에서 고른 조합과 동일하다.
  분할        split_group (flight 단위, train/test 편명 겹침 0건)
  max_depth   8
  min_samples_leaf 200
  random_state 42
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor

SPLIT_PATH = Path("data/processed/flight_prices_with_splits.csv")

RANDOM_STATE = 42
MAX_DEPTH = 8
MIN_SAMPLES_LEAF = 200
SCHEME = "split_group"

TARGET = "price"
CATEGORICAL = ["airline", "source_city", "destination_city",
               "departure_time", "arrival_time", "class", "stops"]
NUMERIC = ["duration", "days_left"]

CITY_KO = {"Delhi": "델리", "Mumbai": "뭄바이", "Bangalore": "벵갈루루",
           "Kolkata": "콜카타", "Hyderabad": "하이데라바드", "Chennai": "첸나이"}
TIME_KO = {"Early_Morning": "이른 아침", "Morning": "오전", "Afternoon": "오후",
           "Evening": "저녁", "Night": "밤", "Late_Night": "심야"}
STOPS_KO = {"zero": "직항", "one": "1회 경유", "two_or_more": "2회 이상 경유"}
CLASS_KO = {"Economy": "이코노미석", "Business": "비즈니스석"}
AIRLINE_KO = {"Vistara": "Vistara", "Air_India": "Air India", "Indigo": "Indigo",
              "GO_FIRST": "GO FIRST", "AirAsia": "AirAsia", "SpiceJet": "SpiceJet"}

FEATURE_KO = {"airline": "항공사", "source_city": "출발 도시",
              "destination_city": "도착 도시", "departure_time": "출발 시간대",
              "arrival_time": "도착 시간대", "stops": "경유 횟수",
              "class": "좌석 등급", "duration": "비행시간", "days_left": "예약 리드타임"}

STOPS_ORDER = ["zero", "one", "two_or_more"]
TIME_ORDER = ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]


# ---------------------------------------------------------------- 모델

@st.cache_resource(show_spinner="모델을 학습하는 중입니다...")
def build_model() -> dict:
    """데이터를 읽고 나무를 학습한다. 앱 실행 중 한 번만 수행된다."""
    df = pd.read_csv(SPLIT_PATH)
    dummies = pd.get_dummies(df[CATEGORICAL], columns=CATEGORICAL, drop_first=False)
    X = pd.concat([dummies.astype(int), df[NUMERIC]], axis=1)

    is_train = df[SCHEME] == "train"
    y_tr = df.loc[is_train, TARGET].to_numpy(float)
    y_te = df.loc[~is_train, TARGET].to_numpy(float)

    tree = DecisionTreeRegressor(max_depth=MAX_DEPTH, min_samples_leaf=MIN_SAMPLES_LEAF,
                                 random_state=RANDOM_STATE)
    tree.fit(X[is_train], y_tr)

    pred_te = tree.predict(X[~is_train])
    leaf_of_train = tree.apply(X[is_train])

    return {
        "tree": tree,
        "columns": list(X.columns),
        "train_df": df[is_train].reset_index(drop=True),
        "leaf_of_train": leaf_of_train,
        "train_mean": float(y_tr.mean()),
        "n_train": int(is_train.sum()),
        "n_test": int((~is_train).sum()),
        "test_mae": float(mean_absolute_error(y_te, pred_te)),
        "test_rmse": float(root_mean_squared_error(y_te, pred_te)),
        "test_r2": float(r2_score(y_te, pred_te)),
        "options": {c: sorted(df[c].unique()) for c in CATEGORICAL},
        "duration_range": (float(df.duration.min()), float(df.duration.max())),
        "days_range": (int(df.days_left.min()), int(df.days_left.max())),
        "importance": pd.Series(tree.feature_importances_, index=X.columns),
    }


def to_feature_row(inputs: dict, columns: list[str]) -> pd.DataFrame:
    """사용자 입력을 학습 때와 같은 원-핫 열 순서로 변환한다."""
    row = pd.Series(0, index=columns, dtype=float)
    for col in CATEGORICAL:
        key = f"{col}_{inputs[col]}"
        if key not in row.index:
            raise KeyError(f"학습 데이터에 없는 값입니다: {key}")
        row[key] = 1
    for col in NUMERIC:
        row[col] = float(inputs[col])
    return row.to_frame().T


# ---------------------------------------------------------------- 설명 문구

def hours_text(value: float) -> str:
    h, m = int(value), round((value - int(value)) * 60)
    return f"{h}시간 {m}분" if m else f"{h}시간"


def has_final_consonant(word: str) -> bool:
    """한글 마지막 글자에 받침이 있는지. 조사(을/를) 선택에 쓴다."""
    if not word:
        return False
    code = ord(word[-1])
    return 0xAC00 <= code <= 0xD7A3 and (code - 0xAC00) % 28 != 0


def with_object_particle(word: str) -> str:
    return word + ("을" if has_final_consonant(word) else "를")


def question_text(column: str, threshold: float) -> str:
    if column == "duration":
        return f"비행시간이 {hours_text(threshold)} 이하입니까?"
    if column == "days_left":
        return f"출발까지 {int(np.floor(threshold))}일 이하로 남았습니까?"
    for prefix, table, tmpl in [
        ("class_", CLASS_KO, "좌석 등급이 {}입니까?"),
        ("airline_", AIRLINE_KO, "항공사가 {}입니까?"),
        ("source_city_", CITY_KO, "출발 도시가 {}입니까?"),
        ("destination_city_", CITY_KO, "도착 도시가 {}입니까?"),
        ("departure_time_", TIME_KO, "출발 시간대가 {}입니까?"),
        ("arrival_time_", TIME_KO, "도착 시간대가 {}입니까?"),
        ("stops_", STOPS_KO, "경유가 {}입니까?"),
    ]:
        if column.startswith(prefix):
            raw = column[len(prefix):]
            return tmpl.format(table.get(raw, raw))
    return f"{column} <= {threshold:.2f} 입니까?"


def source_column(onehot: str) -> str:
    """원-핫 열 이름에서 원래 변수명을 되찾는다."""
    for col in CATEGORICAL:
        if onehot.startswith(f"{col}_"):
            return col
    return onehot


def value_text(column: str, inputs: dict) -> str:
    if column == "duration":
        return f"{inputs['duration']:.2f}시간 ({hours_text(inputs['duration'])})"
    if column == "days_left":
        return f"{int(inputs['days_left'])}일"
    base = source_column(column)
    table = {"class": CLASS_KO, "airline": AIRLINE_KO, "source_city": CITY_KO,
             "destination_city": CITY_KO, "departure_time": TIME_KO,
             "arrival_time": TIME_KO, "stops": STOPS_KO}[base]
    return table.get(inputs[base], inputs[base])


def trace_path(model: dict, x_row: pd.DataFrame, inputs: dict) -> tuple[list, int]:
    """루트부터 잎까지 어떤 질문에 어떻게 답했는지 기록한다."""
    tree, columns = model["tree"], model["columns"]
    t = tree.tree_
    x = x_row.iloc[0]

    steps, node = [], 0
    while t.children_left[node] != -1:
        col = columns[t.feature[node]]
        thr = float(t.threshold[node])
        goes_left = float(x[col]) <= thr
        # 수치형 질문은 '임계값 이하입니까?' 라서 왼쪽이 '예',
        # 원-핫 질문은 '해당 항목입니까?' 라서 값이 1 인 오른쪽이 '예'
        says_yes = goes_left if col in NUMERIC else not goes_left
        node = t.children_left[node] if goes_left else t.children_right[node]

        steps.append({
            "질문": question_text(col, thr),
            "입력값": value_text(col, inputs),
            "답": "예" if says_yes else "아니오",
            "변수": FEATURE_KO[source_column(col)],
            "이동 후 평균": float(t.value[node][0][0]),
            "이동 후 건수": int(t.n_node_samples[node]),
        })
    return steps, node


# ---------------------------------------------------------------- 화면

def render_sidebar(model: dict) -> dict:
    opts = model["options"]
    st.sidebar.header("설명변수 입력")
    st.sidebar.caption("9개 항목을 모두 채우면 예측이 갱신됩니다.")

    st.sidebar.subheader("좌석과 항공사")
    cls = st.sidebar.radio("좌석 등급", ["Economy", "Business"],
                           format_func=lambda v: CLASS_KO[v], horizontal=True,
                           help="가격을 가장 크게 좌우하는 항목입니다.")
    airline = st.sidebar.selectbox("항공사", opts["airline"],
                                   format_func=lambda v: AIRLINE_KO.get(v, v))

    st.sidebar.subheader("노선")
    source = st.sidebar.selectbox("출발 도시", opts["source_city"],
                                  format_func=lambda v: CITY_KO.get(v, v))
    dest_opts = [c for c in opts["destination_city"] if c != source]
    dest = st.sidebar.selectbox("도착 도시", dest_opts,
                                format_func=lambda v: CITY_KO.get(v, v),
                                help="출발 도시와 같은 값은 목록에서 제외됩니다.")
    stops = st.sidebar.selectbox("경유 횟수", STOPS_ORDER,
                                 index=1, format_func=lambda v: STOPS_KO[v])

    st.sidebar.subheader("시간대")
    dep = st.sidebar.selectbox("출발 시간대", TIME_ORDER, index=1,
                               format_func=lambda v: TIME_KO[v])
    arr = st.sidebar.selectbox("도착 시간대", TIME_ORDER, index=4,
                               format_func=lambda v: TIME_KO[v])

    st.sidebar.subheader("비행시간과 예약 시점")
    d_lo, d_hi = model["duration_range"]
    duration = st.sidebar.number_input(
        "비행시간 (시간)", min_value=d_lo, max_value=d_hi, value=6.0, step=0.25,
        format="%.2f", help=f"학습 데이터 범위 {d_lo:.2f} ~ {d_hi:.2f}시간. 2.17 = 2시간 10분")
    k_lo, k_hi = model["days_range"]
    days_left = st.sidebar.slider(
        "출발까지 남은 일수", min_value=k_lo, max_value=k_hi, value=30,
        help=f"학습 데이터 범위 {k_lo} ~ {k_hi}일")

    return {"airline": airline, "source_city": source, "destination_city": dest,
            "departure_time": dep, "arrival_time": arr, "stops": stops,
            "class": cls, "duration": duration, "days_left": days_left}


def render_prediction(model: dict, pred: float, leaf: int) -> None:
    t = model["tree"].tree_
    leaf_n = int(t.n_node_samples[leaf])
    diff = pred - model["train_mean"]

    # 잎 내부 분포는 정규분포가 아니라 오른쪽으로 치우쳐 있으므로
    # 평균±표준편차 대신 실제 분위수를 쓴다.
    same = model["train_df"].loc[model["leaf_of_train"] == leaf, TARGET]
    q10, q50, q90 = same.quantile([0.10, 0.50, 0.90])

    c1, c2, c3 = st.columns([2, 1, 1])
    c1.metric("예상 표시가격", f"{pred:,.0f} INR",
              delta=f"{diff:+,.0f} vs 전체 평균 {model['train_mean']:,.0f}")
    c2.metric("이 규칙의 학습 건수", f"{leaf_n:,}건")
    c3.metric("같은 조건의 중앙값", f"{q50:,.0f} INR")

    st.info(
        f"같은 조건에 해당하는 학습 데이터 {leaf_n:,}건의 평균이 **{pred:,.0f} INR**이고, "
        f"이 값이 그대로 예측값이 됩니다.\n\n"
        f"실제로는 그중 80%가 **{q10:,.0f} ~ {q90:,.0f} INR** 사이에 있었고 "
        f"(최소 {same.min():,.0f} / 최대 {same.max():,.0f}), "
        f"중앙값은 {q50:,.0f} INR이었습니다. "
        "조건이 같아도 가격은 이만큼 흩어지므로, 예측값은 하나의 점이 아니라 "
        "이 범위의 중심으로 읽는 편이 맞습니다."
    )


def render_path(model: dict, steps: list, inputs: dict) -> None:
    st.subheader("이 예측에 실제로 사용된 설명변수")
    used = list(dict.fromkeys(s["변수"] for s in steps))
    st.markdown(
        f"나무는 **{len(steps)}번**의 질문을 거쳐 답을 냈고, 그 과정에서 "
        f"**{len(used)}개 변수**({', '.join(used)})만 사용했습니다."
    )

    table = pd.DataFrame([{
        "단계": i,
        "질문": s["질문"],
        "입력한 값": s["입력값"],
        "답": s["답"],
        "이동한 그룹 평균": f"{s['이동 후 평균']:,.0f} INR",
        "그룹 크기": f"{s['이동 후 건수']:,}건",
    } for i, s in enumerate(steps, start=1)])
    st.dataframe(table, hide_index=True, width="stretch")

    unused = [FEATURE_KO[c] for c in CATEGORICAL + NUMERIC if FEATURE_KO[c] not in used]
    if unused:
        listed = with_object_particle(", ".join(unused))
        st.caption(
            f"이번 입력에서는 **{listed}** 묻지 않았습니다. "
            "값을 바꿔도 예측이 달라지지 않으며, 이는 해당 변수가 무의미해서가 아니라 "
            "앞선 질문들로 이미 그룹이 충분히 좁혀졌기 때문입니다."
        )


def render_leaf_samples(model: dict, leaf: int) -> None:
    st.subheader("같은 규칙에 속한 실제 학습 데이터")
    same = model["train_df"][model["leaf_of_train"] == leaf]
    if same.empty:
        st.write("해당 데이터가 없습니다.")
        return

    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(
            same[TARGET].describe()[["count", "mean", "std", "min", "50%", "max"]]
            .rename({"count": "건수", "mean": "평균", "std": "표준편차",
                     "min": "최솟값", "50%": "중앙값", "max": "최댓값"})
            .round(0).astype(int).to_frame("price (INR)"),
            width="stretch")
    with c2:
        st.caption("이 규칙에 속한 학습 데이터의 가격 분포")
        counts = np.histogram(same[TARGET], bins=30)
        st.bar_chart(pd.DataFrame({"건수": counts[0]},
                                  index=np.round(counts[1][:-1]).astype(int)))

    with st.expander("표본 10건 보기"):
        st.dataframe(
            same.sample(min(10, len(same)), random_state=RANDOM_STATE)[
                ["flight", "airline", "source_city", "destination_city",
                 "departure_time", "arrival_time", "stops", "class",
                 "duration", "days_left", "price"]],
            hide_index=True, width="stretch")


def render_model_info(model: dict) -> None:
    st.subheader("모델 정보")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("테스트 R²", f"{model['test_r2']:.4f}")
    c2.metric("테스트 MAE", f"{model['test_mae']:,.0f} INR")
    c3.metric("테스트 RMSE", f"{model['test_rmse']:,.0f} INR")
    c4.metric("잎(규칙) 개수", f"{model['tree'].get_n_leaves():,}개")

    st.markdown(
        f"""
| 항목 | 값 |
|---|---|
| 알고리즘 | 의사결정나무 회귀 (CART, 분할 기준 = 잔차제곱합 최소화) |
| max_depth | {MAX_DEPTH} |
| min_samples_leaf | {MIN_SAMPLES_LEAF} |
| random_state | {RANDOM_STATE} |
| 학습 / 테스트 | {model['n_train']:,}건 / {model['n_test']:,}건 |
| 분할 방식 | `{SCHEME}` — 편명(flight) 단위로 나눠 train/test 편명 겹침 0건 |
| 입력 인코딩 | 범주형 원-핫 35개 + 수치형 2개 = 37개 |
"""
    )
    st.caption(
        "편명 단위로 분할했기 때문에 이 성능은 **처음 보는 항공편**에 대한 값입니다. "
        "같은 설정을 무작위 분할로 평가하면 R²가 0.9518로 조금 더 높게 나오는데, "
        "이는 테스트 항공권의 편명을 학습에서 이미 본 덕분입니다. "
        "깊이를 12 이상으로 늘리면 이 격차가 0.02를 넘어섭니다."
    )

    st.markdown("**모델 전체에서 각 변수가 차지하는 중요도**")
    imp = model["importance"]
    grouped = imp.groupby(source_column).sum().sort_values(ascending=False)
    grouped = grouped[grouped > 0].rename(index=FEATURE_KO)
    st.bar_chart(grouped.to_frame("중요도"), horizontal=True)
    st.caption(
        "좌석 등급이 대부분을 차지합니다. 다만 중요도가 낮다고 쓸모없다는 뜻은 아닙니다 — "
        "예약 리드타임은 이코노미석 안에서는 매우 강한 신호지만, 등급이 전체 변동의 "
        "88%를 먼저 흡수하기 때문에 상대적으로 작게 보입니다."
    )


def render_intro() -> None:
    st.title("항공권 표시가격 예측 대시보드")
    st.markdown(
        "왼쪽에서 항공권 조건을 고르면, 학습된 **의사결정나무(깊이 8)** 가 "
        "예상 표시가격을 계산하고 **어떤 질문을 거쳐 그 값에 도달했는지** 보여줍니다."
    )
    with st.expander("이 모델은 어떻게 예측하나요?"):
        st.markdown(
            """
의사결정나무는 스무고개와 같습니다. 전체 항공권에서 시작해 예/아니오 질문을 던지며
비슷한 가격끼리 모인 작은 그룹으로 좁혀 가고, 마지막 그룹의 **평균 가격**을 답으로 내놓습니다.

- 첫 질문은 항상 **좌석 등급**입니다. 이코노미석과 비즈니스석의 평균 가격이 8배 차이 나기 때문에,
  이 하나로 전체 가격 변동의 약 88%가 설명됩니다.
- 그 다음 질문은 등급에 따라 달라집니다. 이코노미석은 **예약 리드타임**을,
  비즈니스석은 **비행시간**을 먼저 확인합니다.
- 나무는 최대 8번까지만 질문하므로, 입력한 9개 변수를 전부 쓰지는 않습니다.

**데이터 출처**: 인도 국내선 6개 도시(델리·뭄바이·벵갈루루·콜카타·하이데라바드·첸나이)
구간의 항공권 조회 기록 300,153건. 가격 단위는 인도 루피(INR)입니다.
다른 국가나 노선에는 적용할 수 없습니다.
"""
        )


def main() -> None:
    st.set_page_config(page_title="항공권 가격 예측", page_icon="✈️", layout="wide")

    if not SPLIT_PATH.exists():
        st.error(
            f"데이터 파일이 없습니다: `{SPLIT_PATH}`\n\n"
            "프로젝트 루트에서 아래를 먼저 실행하세요.\n\n"
            "```\npython analysis/04_train_test_split.py\n```"
        )
        st.stop()

    model = build_model()
    render_intro()
    inputs = render_sidebar(model)

    try:
        x_row = to_feature_row(inputs, model["columns"])
    except KeyError as exc:
        st.error(f"입력을 변환할 수 없습니다: {exc}")
        st.stop()

    pred = float(model["tree"].predict(x_row)[0])
    steps, leaf = trace_path(model, x_row, inputs)

    st.divider()
    render_prediction(model, pred, leaf)

    st.divider()
    tab_path, tab_data, tab_model = st.tabs(
        ["예측 근거", "같은 규칙의 실제 데이터", "모델 정보"])
    with tab_path:
        render_path(model, steps, inputs)
    with tab_data:
        render_leaf_samples(model, leaf)
    with tab_model:
        render_model_info(model)


if __name__ == "__main__":
    main()

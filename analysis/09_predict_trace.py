"""테스트 데이터 2건(Economy 1건, Business 1건)이 깊이 4 나무를 통과하는 경로를 추적한다.

각 항공권이 루트의 첫 질문부터 어떤 조건을 만족해 어느 가지로 가는지 보여주고,
최종 잎의 예측값과 실제 가격의 차이를 계산한다.

표본 추출도 RANDOM_STATE=42 로 고정해 같은 항공권이 다시 뽑히도록 한다.

산출물:
  outputs/tables/predict_trace_samples.csv   두 표본의 입력 조건과 예측 결과
  outputs/figures/11_trace_on_tree.png       나무 위에 두 경로를 덧칠
  outputs/figures/12_trace_steps.png         질문-답변 단계별 흐름도
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.tree import DecisionTreeRegressor, plot_tree

SPLIT_PATH = Path("data/processed/flight_prices_with_splits.csv")
FIG_DIR = Path("outputs/figures")
TBL_DIR = Path("outputs/tables")

RANDOM_STATE = 42
MAX_DEPTH = 4
MIN_SAMPLES_LEAF = 500
SCHEME = "split_group"

TARGET = "price"
CATEGORICAL = ["airline", "source_city", "destination_city",
               "departure_time", "arrival_time", "class", "stops"]
NUMERIC = ["duration", "days_left"]

CASE_COLOR = {"Economy": "#1f77b4", "Business": "#d62728"}
SHARED_COLOR = "#7A5AA8"


def setup_style() -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.bbox"] = "tight"


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


def pick_samples(df: pd.DataFrame, is_train: pd.Series) -> dict:
    """테스트 데이터에서 등급별 1건씩 무작위 추출."""
    test = df[~is_train]
    return {cls: test[test["class"] == cls].sample(1, random_state=RANDOM_STATE).iloc[0]
            for cls in ["Economy", "Business"]}


# ---------------------------------------------------------------- 질문 서술

def question_text(column: str, threshold: float) -> str:
    """분할 조건을 사람이 읽는 질문으로."""
    if column == "duration":
        h, m = int(threshold), round((threshold - int(threshold)) * 60)
        return f"비행시간이 {threshold:.2f}시간({h}시간 {m}분) 이하입니까?"
    if column == "days_left":
        return f"출발까지 {int(np.floor(threshold))}일 이하로 남았습니까?"
    pretty = {"class_Economy": "이코노미석입니까?",
              "class_Business": "비즈니스석입니까?"}
    if column in pretty:
        return pretty[column]
    if column.startswith("airline_"):
        return f"{column[len('airline_'):].replace('_', ' ')} 항공입니까?"
    if column.startswith("destination_city_"):
        return f"도착지가 {column[len('destination_city_'):]}입니까?"
    if column.startswith("source_city_"):
        return f"출발지가 {column[len('source_city_'):]}입니까?"
    if column.startswith("departure_time_"):
        return f"출발 시간대가 {column[len('departure_time_'):]}입니까?"
    if column.startswith("arrival_time_"):
        return f"도착 시간대가 {column[len('arrival_time_'):]}입니까?"
    if column.startswith("stops_"):
        return f"경유가 {column[len('stops_'):]}입니까?"
    return f"{column} <= {threshold:.2f} 입니까?"


def value_text(column: str, value: float, row: pd.Series) -> str:
    """이 항공권의 해당 항목 실제 값."""
    if column == "duration":
        h, m = int(row.duration), round((row.duration - int(row.duration)) * 60)
        return f"{row.duration:.2f}시간 ({h}시간 {m}분)"
    if column == "days_left":
        return f"{int(row.days_left)}일"
    for prefix, col in [("class_", "class"), ("airline_", "airline"),
                        ("destination_city_", "destination_city"),
                        ("source_city_", "source_city"),
                        ("departure_time_", "departure_time"),
                        ("arrival_time_", "arrival_time"), ("stops_", "stops")]:
        if column.startswith(prefix):
            return f"{row[col]}"
    return f"{value:g}"


def trace(tree, X: pd.DataFrame, row_idx, row: pd.Series) -> list[dict]:
    """루트부터 잎까지의 질문/답변/이동 기록."""
    t = tree.tree_
    columns = list(X.columns)
    x = X.loc[row_idx]

    steps, node = [], 0
    while t.children_left[node] != -1:
        col = columns[t.feature[node]]
        thr = float(t.threshold[node])
        goes_left = float(x[col]) <= thr
        nxt = t.children_left[node] if goes_left else t.children_right[node]

        # 수치형 질문은 '임계값 이하입니까?' 이므로 왼쪽이 '예'.
        # 원-핫 질문은 '해당 항목입니까?' 이므로 값이 1 인 오른쪽이 '예'.
        says_yes = goes_left if col in NUMERIC else not goes_left

        steps.append({
            "node": node, "next": nxt, "feature": col, "threshold": thr,
            "question": question_text(col, thr),
            "value": value_text(col, float(x[col]), row),
            "answer": "예" if says_yes else "아니오",
            "direction": "왼쪽" if goes_left else "오른쪽",
            "node_mean": float(t.value[node][0][0]),
            "node_n": int(t.n_node_samples[node]),
            "next_mean": float(t.value[nxt][0][0]),
            "next_n": int(t.n_node_samples[nxt]),
        })
        node = nxt
    return steps, node


# ---------------------------------------------------------------- 그림

def fig_trace_on_tree(tree, X: pd.DataFrame, paths: dict) -> None:
    """나무 전체 그림 위에 두 경로의 노드를 색으로 강조한다."""
    fig, ax = plt.subplots(figsize=(30, 11))
    ax.grid(False)
    anns = plot_tree(tree, feature_names=list(X.columns), filled=True, rounded=True,
                     precision=2, fontsize=8, impurity=False, ax=ax)

    # plot_tree 는 루트 자식에 'True'/'False' 라벨 주석을 끼워 넣는다. 제거해야 노드 id 와 맞는다.
    node_anns = [a for a in anns if a.get_text().strip() not in ("True", "False")]
    if len(node_anns) != tree.tree_.node_count:
        raise RuntimeError(
            f"주석 {len(node_anns)}개 != 노드 {tree.tree_.node_count}개. 매핑을 확인하세요."
        )

    for cls, nodes in paths.items():
        for nid in nodes:
            shared = any(nid in other for c, other in paths.items() if c != cls)
            color = SHARED_COLOR if shared else CASE_COLOR[cls]
            box = node_anns[nid].get_bbox_patch()
            box.set_edgecolor(color)
            box.set_linewidth(4.5)

    handles = [plt.Line2D([], [], color=CASE_COLOR[c], lw=4, label=f"{c} 표본 경로")
               for c in paths]
    handles.append(plt.Line2D([], [], color=SHARED_COLOR, lw=4, label="두 경로가 공유하는 노드"))
    ax.legend(handles=handles, loc="upper left", fontsize=12, framealpha=0.95)
    ax.set_title(
        f"테스트 표본 2건이 지나간 경로 (max_depth={MAX_DEPTH}, "
        f"min_samples_leaf={MIN_SAMPLES_LEAF}, random_state={RANDOM_STATE})",
        fontsize=16, pad=18)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "11_trace_on_tree.png")
    plt.close(fig)


def draw_box(ax, y: float, text: str, color: str, height: float,
             face: str = "white", fontsize: int = 10, weight: str = "normal") -> None:
    ax.add_patch(FancyBboxPatch((0.03, y), 0.94, height,
                                boxstyle="round,pad=0.012", linewidth=2.2,
                                edgecolor=color, facecolor=face, zorder=2))
    ax.text(0.5, y + height / 2, text, ha="center", va="center",
            fontsize=fontsize, zorder=3, weight=weight, linespacing=1.5)


def fig_trace_steps(cases: dict) -> None:
    """질문 -> 답변 -> 이동 을 세로 흐름도로."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 11))

    for ax, (cls, case) in zip(axes, cases.items()):
        color = CASE_COLOR[cls]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        row, steps = case["row"], case["steps"]
        ax.set_title(f"{cls} 표본 (id={int(row.id)}, {row.flight})\n"
                     f"실제 가격 {row.price:,.0f} INR",
                     fontsize=14, color=color, pad=14, weight="bold")

        # 위에서 아래로 커서를 내리며 배치한다 (상자 높이 + 간격을 누적)
        gap, cursor = 0.048, 0.965
        h_start, h_step, h_result = 0.072, 0.125, 0.092

        cursor -= h_start
        draw_box(ax, cursor, f"시작: 전체 항공권\n"
                             f"학습 평균 {steps[0]['node_mean']:,.0f} INR "
                             f"(n={steps[0]['node_n']:,})",
                 "#888888", h_start, "#F2F2F2", 10)

        for i, s in enumerate(steps):
            prev_bottom = cursor
            cursor -= gap + h_step
            mark = "O" if s["answer"] == "예" else "X"
            body = (f"질문 {i + 1}. {s['question']}\n"
                    f"이 항공권: {s['value']}   ->   {mark} {s['answer']} "
                    f"({s['direction']} 가지)\n"
                    f"이동한 노드 평균 {s['next_mean']:,.0f} INR (n={s['next_n']:,})")
            draw_box(ax, cursor, body, color, h_step, "white", 10)
            ax.annotate("", xy=(0.5, cursor + h_step), xytext=(0.5, prev_bottom),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2), zorder=1)

        prev_bottom = cursor
        cursor -= gap + h_result
        diff = case["actual"] - case["pred"]
        draw_box(ax, cursor,
                 f"예측 {case['pred']:,.0f} INR   vs   실제 {case['actual']:,.0f} INR\n"
                 f"차이 {diff:+,.0f} INR  (오차율 {abs(diff) / case['actual'] * 100:.1f}%)",
                 color, h_result, "#FFF6E5", 12, "bold")
        ax.annotate("", xy=(0.5, cursor + h_result), xytext=(0.5, prev_bottom),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2), zorder=1)

    fig.suptitle("깊이 4 나무의 질문을 따라간 예측 과정", fontsize=16, y=0.99)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "12_trace_steps.png")
    plt.close(fig)


# ---------------------------------------------------------------- 실행

def main() -> None:
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TBL_DIR.mkdir(parents=True, exist_ok=True)

    df, X, is_train, tree = load_and_fit()
    samples = pick_samples(df, is_train)

    show_cols = ["id", "flight", "airline", "source_city", "destination_city",
                 "departure_time", "arrival_time", "stops", "class",
                 "duration", "days_left", "price"]

    print("=" * 92)
    print(f"테스트 데이터에서 무작위 추출한 2건 (random_state={RANDOM_STATE})")
    print("=" * 92)
    print(pd.DataFrame([s[show_cols] for s in samples.values()]).to_string(index=False))

    cases, paths = {}, {}
    for cls, row in samples.items():
        steps, leaf = trace(tree, X, row.name, row)
        pred = float(tree.tree_.value[leaf][0][0])
        cases[cls] = {"row": row, "steps": steps, "leaf": leaf,
                      "pred": pred, "actual": float(row.price)}
        paths[cls] = [s["node"] for s in steps] + [leaf]

        print("\n" + "=" * 92)
        print(f"[{cls}] id={int(row.id)} / 편명 {row.flight} / 실제 가격 {row.price:,.0f} INR")
        print("=" * 92)
        print(f"  시작 — 전체 항공권 {steps[0]['node_n']:,}건, "
              f"평균 {steps[0]['node_mean']:,.0f} INR")
        for i, s in enumerate(steps, start=1):
            mark = "O" if s["answer"] == "예" else "X"
            print(f"\n  질문 {i}. {s['question']}")
            print(f"     이 항공권: {s['value']}  ->  {mark} {s['answer']} "
                  f"({s['direction']} 가지로 이동)")
            print(f"     도착 노드: 평균 {s['next_mean']:,.0f} INR (n={s['next_n']:,})")
        diff = cases[cls]["actual"] - pred
        print(f"\n  최종 잎 노드 {leaf} -> 예측 {pred:,.0f} INR")
        print(f"  실제 {cases[cls]['actual']:,.0f} INR / 예측 {pred:,.0f} INR "
              f"= 차이 {diff:+,.0f} INR (오차율 {abs(diff) / cases[cls]['actual'] * 100:.1f}%)")

    out = pd.DataFrame([{
        **{c: samples[cls][c] for c in show_cols},
        "예측값": round(cases[cls]["pred"], 2),
        "차이(실제-예측)": round(cases[cls]["actual"] - cases[cls]["pred"], 2),
        "오차율(%)": round(abs(cases[cls]["actual"] - cases[cls]["pred"])
                        / cases[cls]["actual"] * 100, 2),
        "잎 노드": cases[cls]["leaf"],
    } for cls in cases])
    out.to_csv(TBL_DIR / "predict_trace_samples.csv", index=False, encoding="utf-8-sig")

    fig_trace_on_tree(tree, X, paths)
    fig_trace_steps(cases)

    print(f"\n저장: {TBL_DIR / 'predict_trace_samples.csv'}")
    print(f"저장: {FIG_DIR / '11_trace_on_tree.png'}")
    print(f"저장: {FIG_DIR / '12_trace_steps.png'}")


if __name__ == "__main__":
    main()

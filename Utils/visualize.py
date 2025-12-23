import os
import colorsys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from matplotlib.patches import Patch
from matplotlib import patheffects
from matplotlib.ticker import MultipleLocator, AutoLocator

plt.rcParams['font.family'] = 'Times New Roman'


def draw_scatter_plot(save=False):
    problem_sizes = ["10-5", "15-5", "15-10", "20-10", "20-15", "25-15"]
    # algorithms = ["SPT+SETT", "RL+SETT", "MWKR+RL (comm.)", "RL+RL (comm.)", "CTCE",
    #               "DTDE (comm.)", "DTDE with pt (comm.)", "CTDE with CL (comm.)", "CTDE with EP (comm.)"]
    algorithms = ["DTDE with pt (comm.)", "DTDE (comm.)", "CTDE with CL (comm.)", "CTDE with EP (comm.)", "CTCE",
                  "RL+RL (comm.)", "RL+SETT", "MWKR+RL (comm.)", "MWKR+SETT"]

    df_avg = pd.DataFrame(index=problem_sizes, columns=algorithms)
    for i, size in enumerate(problem_sizes):
        df = pd.read_excel('../output/test/results_scaled_by_rand+rand.xlsx',
                           sheet_name=size, engine="openpyxl", index_col=0).iloc[:-1]
        df_selected = df[algorithms]
        df_avg.iloc[i] = df_selected.mean()

    mapping = {
        "MWKR+SETT": "MWKR+SETT",
        "RL+SETT": "RL+SETT",
        "MWKR+RL (comm.)": "MWKR+RL",
        "RL+RL (comm.)": "RL+RL",
        "CTCE": "CTCE",
        "DTDE (comm.)": "DTDE",
        "DTDE with pt (comm.)": "DIA-DTDE",
        "CTDE with CL (comm.)": "CTDE with CL",
        "CTDE with EP (comm.)": "CTDE with EP"
    }

    colors = {
        "MWKR+SETT": "#440154",  # Deep Purple
        "RL+SETT": "#3B528B",  # Indigo Blue
        "MWKR+RL": "#21908C",  # Teal Green
        "RL+RL": "#5DC863",  # Soft Green
        "CTCE": "#FDE725",  # Golden Yellow
        "DTDE": "#F9844A",  # Warm Orange
        "DIA-DTDE": "#D43E4F",  # Coral Red
        "CTDE with CL": "#728EA3",  # Slate Blue
        "CTDE with EP": "#A6AD00",  # Olive Green
    }

    markers = {
        'MWKR+SETT': 'x',
        "RL+SETT": '^',
        "MWKR+RL": 'v',
        "RL+RL": 's',
        'CTCE': 'd',
        'DTDE': 'o',
        'DIA-DTDE': '*',
        "CTDE with CL": 'p',
        "CTDE with EP": 'h'
    }

    positions = range(len(problem_sizes))

    # 그래프 생성
    fig, ax = plt.subplots(figsize=(12, 10))

    for algorithm in algorithms:
        # 색상과 마커 설정
        color = colors.get(mapping[algorithm], 'black')
        marker = markers.get(mapping[algorithm], 'o')

        # 데이터 추출
        y_values = []
        for size in problem_sizes:
            y_values.append(df_avg.loc[size, algorithm])

        # 그래프 그리기
        ax.plot(positions,
                y_values,
                color=color,
                marker=marker,
                label=mapping[algorithm],
                linestyle="None",
                markersize=15)

    # 그래프 설정
    ax.set_xlabel('No. of Machines X No. of Jobs', fontsize=20, labelpad=10)
    ax.set_ylabel('Performance Measure ($\sigma$)', fontsize=20)

    # x축 설정
    ax.set_xlim(-0.5, len(problem_sizes) - 0.5)
    ax.set_xticks(positions)
    xticklabels = ["5X10", "5X15", "10X15", "10X20", "15X20", "15X25"]
    ax.set_xticklabels(xticklabels, ha='center', fontsize=20)

    # y축 범위 설정 (그림에 맞게)
    ax.set_ylim(0, 35)
    yticklabels = ax.get_yticklabels()
    ax.set_yticklabels(yticklabels, fontsize=20)

    # 격자 추가
    # ax.grid(True, alpha=0.3)
    ax.minorticks_on()
    ax.xaxis.set_major_locator(MultipleLocator(1))  # 주 눈금 간격: 2
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))  # 보조 눈금 간격: 1
    ax.grid(True, axis='y', linestyle=':', color='gray', alpha=0.3)
    ax.grid(True, which='major', axis='x', linestyle=':', color='gray', alpha=0.3)
    ax.grid(True, which='minor', axis='x', linestyle='-', color='gray', alpha=0.5)

    # 범례 추가
    ax.legend(
        loc='upper center',  # 기준 위치 (anchor 기준에서 위쪽 가운데)
        bbox_to_anchor=(0.5, -0.1),  # (x, y) → x는 가운데(0.5), y는 아래쪽 바깥 (-0.15)
        ncol=5,  # 범례 항목을 가로로 배열할 때 열 수 지정 (선택)
        frameon=False,  # 범례 박스 테두리 없애기 (선택)
        fontsize=15
    )

    plt.tight_layout()
    plt.show()

    if save:
        file_dir = '../output/test/figures/MARL/'
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        fig.savefig(file_dir + 'scatter plot.png', dpi=300, bbox_inches='tight')


def draw_boxplot_SARL(problem_size="10-5", tag="FJSP", save=False):
    # 1. 데이터 불러오기
    df = pd.read_excel('../output/test/results_scaled_by_rand+rand.xlsx',
                       sheet_name=problem_size, engine="openpyxl", index_col=0).iloc[:-1]

    # 2. 열 정의
    if tag == "FJSP":
        columns = ["RL+SETT", "SPT+SETT", "MOR+SETT", "MWKR+SETT",
                   "RL+TDD", "SPT+TDD", "MOR+TDD", "MWKR+TDD",
                   "RL+TDT", "SPT+TDT", "MOR+TDT", "MWKR+TDT"]
    else:
        columns = ["SPT+RL", "SPT+SETT", "SPT+TDD", "SPT+TDT",
                   "MOR+RL", "MOR+SETT", "MOR+TDD", "MOR+TDT",
                   "MWKR+RL", "MWKR+SETT", "MWKR+TDD", "MWKR+TDT"]

    # 3. 컬럼별 데이터 수집
    data = [df[col].dropna().values for col in columns]

    # 4. x축 위치 지정 (그룹 간 간격 포함)
    fig_width = 12
    fig_height = 8
    num_groups = 3
    boxes_per_group = 4

    # 총 x축 길이를 기준으로 spacing 비율 설정
    total_width = fig_width
    box_width = 0.5
    group_spacing_ratio = 0.08  # 그룹 간 간격 비율
    intra_group_spacing_ratio = 0.2  # 그룹 내 box 간격 비율

    # 전체 위치 계산
    positions = []
    base = 0
    for i in range(num_groups):
        for j in range(boxes_per_group):
            positions.append(base + j * (box_width + intra_group_spacing_ratio))
        base = positions[-1] + group_spacing_ratio * total_width  # 그룹 간 간격 반영

    # 5. 색상 지정 (Sub_Category 기준)
    colors = {
        "RL": "#440154",
        "SPT": "#3B528B",
        "MOR": "#21908C",
        "MWKR": "#5DC863",
        "SETT": "#35B779",
        "TDD": "#FDE725",
        "TDT": "#29798E"
    }

    labels = columns
    if tag == "FJSP":
        box_colors = [colors[col.split('+')[0]] for col in columns]
    else:
        box_colors = [colors[col.split('+')[1]] for col in columns]

    # 6. boxplot 그리기
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=box_width,
        notch=True,
        patch_artist=True  # 색상 채우기 허용
    )

    # 7. 색상 입히기
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)
        patch.set_linewidth(0.8)

    # 8. 그룹 경계선 추가
    boundary1 = (positions[3] + positions[4]) / 2
    boundary2 = (positions[7] + positions[8]) / 2

    ax.axvline(x=boundary1, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(x=boundary2, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    # 9. x축 설정
    # ax.set_xlabel("Algorithms", fontsize=20)
    ax.set_ylabel("Performance Measure ($\sigma$)", fontsize=20)
    # ax.set_title(problem_size, fontsize=25)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='center', fontsize=20)

    if tag == "FJSP":
        ax.set_ylim(-10, 40)
        yticklabels = ax.get_yticklabels()
        ax.set_yticklabels(yticklabels, fontsize=20)
    else:
        ax.set_ylim(-10, 40)
        yticklabels = ax.get_yticklabels()
        ax.set_yticklabels(yticklabels, fontsize=20)

    # 10. 범례 수동 생성
    # from matplotlib.patches import Patch
    # legend_handles = [Patch(facecolor=c, label=sub) for sub, c in colors.items()]
    # ax.legend(handles=legend_handles, title="Sub Category")

    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    if save:
        file_dir = '../output/test/figures/SARL/%s/' %tag
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        fig.savefig(file_dir + 'box_plot(%s).png' % problem_size, dpi=300, bbox_inches='tight')


def draw_boxplot_MARL(problem_size="10-5", save=False):
    # 1. 데이터 불러오기
    df = pd.read_excel('../output/test/results_scaled_by_rand+rand.xlsx',
                       sheet_name=problem_size, engine="openpyxl", index_col=0).iloc[:-1]

    # 2. 열 정의
    # groups = [
    #     ["SPT+SETT"],
    #     ["RL+SETT", "MWKR+RL (comm.)", "RL+RL (comm.)"],
    #     ["CTCE", "DTDE (comm.)", "DTDE with pt (comm.)", "CTDE with CL (comm.)", "CTDE with EP (comm.)"]
    # ]
    groups = [
        ["SPT+SETT"],
        ["RL+SETT", "MWKR+RL (comm.)", "RL+RL (comm.)"],
        ["DTDE with pt (comm.)", "DTDE (comm.)", "CTDE with CL (comm.)", "CTDE with EP (comm.)", "CTCE"]
    ]

    mapping = {
        "SPT+SETT": "SPT+SETT",
        "RL+SETT": "RL+SETT",
        "MWKR+RL (comm.)": "MWKR+RL",
        "RL+RL (comm.)": "RL+RL",
        "CTCE": "CTCE",
        "DTDE (comm.)": "DTDE",
        "DTDE with pt (comm.)": "DIA-DTDE",
        "CTDE with CL (comm.)": "CTDE\nwith CL",
        "CTDE with EP (comm.)": "CTDE\nwith EP"
    }

    colors = {
        "SPT+SETT": "#440154",  # Deep Purple
        "RL+SETT": "#3B528B",  # Indigo Blue
        "MWKR+RL": "#21908C",  # Teal Green
        "RL+RL": "#5DC863",  # Soft Green
        "CTCE": "#FDE725",  # Golden Yellow
        "DTDE": "#F9844A",  # Warm Orange
        "DIA-DTDE": "#D43E4F",  # Coral Red
        "CTDE\nwith CL": "#728EA3",  # Slate Blue
        "CTDE\nwith EP": "#A6AD00",  # Olive Green
    }

    # 3. 데이터, 라벨, 색상 수집
    data, labels, box_colors = [], [], []
    for group in groups:
        for col in group:
            data.append(df[col].dropna().values)
            labels.append(mapping[col])
            box_colors.append(colors[mapping[col]])

    # 4. x축 위치 지정 (그룹 간 간격 포함)
    fig_width = 12
    fig_height = 8
    box_width = 0.5
    # 비율 기반 spacing 설정
    group_spacing_ratio = 0.04  # fig_width 대비 그룹 간 간격 비율
    intra_group_spacing_ratio = 0.02  # fig_width 대비 그룹 내 박스 간 간격 비율

    # 전체 박스 개수와 그룹 간 개수 계산
    num_groups = len(groups)
    boxes_per_group = [len(g) for g in groups]

    # fig_width 기준 간격 계산
    total_group_spacing = group_spacing_ratio * fig_width * (num_groups - 1)
    total_intra_spacing = sum((n - 1) * intra_group_spacing_ratio * fig_width for n in boxes_per_group)
    total_box_width = sum(len(g) * box_width for g in groups)

    # 실제로 사용할 수 있는 총 공간
    used_width = total_group_spacing + total_intra_spacing + total_box_width

    # 시작 위치를 가운데 정렬하기 위한 offset
    start_pos = (fig_width - used_width) / 2

    # 위치 계산
    positions = []
    base = start_pos
    for group in groups:
        for i in range(len(group)):
            positions.append(base)
            base += box_width
            if i != len(group) - 1:
                base += intra_group_spacing_ratio * fig_width  # 그룹 내 간격
        base += group_spacing_ratio * fig_width  # 그룹 간 간격 (다음 그룹으로 이동)

    # 6. boxplot 그리기
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=box_width,
        notch=True,
        patch_artist=True  # 색상 채우기 허용
    )

    # 7. 색상 입히기
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)
        patch.set_linewidth(0.8)

    # 9. x축 설정
    group_boundaries = []
    count = 0
    for i, group in enumerate(groups[:-1]):
        count += len(group)
        boundary = (positions[count - 1] + positions[count]) / 2
        group_boundaries.append(boundary)

    for boundary in group_boundaries:
        ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.7)

    # ax.set_xlabel("Algorithms", fontsize=20)
    ax.set_ylabel("Performance Measure ($\sigma$)", fontsize=20)
    # ax.set_title(problem_size, fontsize=25)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='center', fontsize=20)

    ax.set_ylim(-10, 40)
    yticklabels = ax.get_yticklabels()
    ax.set_yticklabels(yticklabels, fontsize=20)

    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    if save:
        file_dir = '../output/test/figures/MARL/'
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        fig.savefig(file_dir + 'box_plot(%s).png' % problem_size, dpi=300, bbox_inches='tight')


def draw_bar_plot(problem_size="10-5", tag="FJSP", save=False):
    title = {
        "10-5": "(a) 5X10",
        "15-5": "(b) 5X15",
        "15-10": "(c) 10X15",
        "20-10": "(d) 10X20",
        "20-15": "(e) 15X20",
        "25-15": "(f) 15X25"
    }

    df = pd.read_excel('../output/test/results_scaled_by_rand+rand.xlsx',
                       sheet_name=problem_size, engine="openpyxl", index_col=0).iloc[:-1]

    if tag == "FJSP":
        columns = ["RL", "SPT", "MOR", "MWKR"]
        df["RL"] = (df["RL+SETT"] + df["RL+TDD"] + df["RL+TDT"]) / 3
        df["SPT"] = (df["SPT+SETT"] + df["SPT+TDD"] + df["SPT+TDT"]) / 3
        df["MOR"] = (df["MOR+SETT"] + df["MOR+TDD"] + df["MOR+TDT"]) / 3
        df["MWKR"] = (df["MWKR+SETT"] + df["MWKR+TDD"] + df["MWKR+TDT"]) / 3
    else:
        columns = ["RL", "SETT", "TDD", "TDT"]
        df["RL"] = (df["SPT+RL"] + df["MOR+RL"] + df["MWKR+RL"]) / 3
        df["SETT"] = (df["SPT+SETT"] + df["MOR+SETT"] + df["MWKR+SETT"]) / 3
        df["TDD"] = (df["SPT+TDD"] + df["MOR+TDD"] + df["MWKR+TDD"]) / 3
        df["TDT"] = (df["SPT+TDT"] + df["MOR+TDT"] + df["MWKR+TDT"]) / 3

    data = [np.mean(df[col].values) for col in columns]

    positions = []
    base = 0
    for i in range(len(columns)):
        positions.append(base)
        base = positions[-1] + 1.5

    colors = {
        "RL": "#440154",
        "SPT": "#3B528B",
        "MOR": "#21908C",
        "MWKR": "#5DC863",
        "SETT": "#35B779",
        "TDD": "#FDE725",
        "TDT": "#29798E"
    }

    # 막대 그리기
    # 자동 위치 계산
    x = np.arange(len(columns))

    # 그래프
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(x, data, color=[colors.get(col, "#333333") for col in columns], edgecolor='black')

    # 레이블 표시
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{height:.2f}',
            ha='center',
            va='bottom',
            fontsize=15
        )

    # 축 설정
    ax.set_title(title[problem_size], fontsize=25)
    # ax.set_xlabel("Algorithms", fontsize=20)
    ax.set_ylabel("Performance Measure ($\sigma$)", fontsize=20)

    ax.set_xticks(x)
    ax.set_xticklabels(columns, fontsize=20)

    if tag == "FJSP":
        ax.set_ylim(0, 25)
        yticklabels = ax.get_yticklabels()
        ax.set_yticklabels(yticklabels, fontsize=20)
    else:
        ax.set_ylim(0, 25)
        yticklabels = ax.get_yticklabels()
        ax.set_yticklabels(yticklabels, fontsize=20)

    ax.tick_params(axis='y', labelsize=20)
    ax.grid(True, linestyle='--', axis='y', alpha=0.5)

    plt.tight_layout()
    plt.show()

    # 저장
    if save:
        file_dir = '../output/test/figures/SARL/%s/' % tag
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        fig.savefig(file_dir + 'bar plot(%s).png' % problem_size, dpi=300, bbox_inches='tight')


def draw_pie_chart(problem_size="10-5", save=False):
    title = {
        "10-5": "(a) 5X10",
        "15-5": "(b) 5X15",
        "15-10": "(c) 10X15",
        "20-10": "(d) 10X20",
        "20-15": "(e) 15X20",
        "25-15": "(f) 15X25"
    }

    # 1. 데이터 불러오기
    df = pd.read_excel('../output/test/results_scaled_by_rand+rand.xlsx',
                       sheet_name=problem_size, engine="openpyxl", index_col=0).iloc[:-1]

    # algorithms = ["SPT+SETT", "RL+SETT", "MWKR+RL (comm.)", "RL+RL (comm.)", "CTCE",
    #               "DTDE (comm.)", "DTDE with pt (comm.)", "CTDE with CL (comm.)", "CTDE with EP (comm.)"]
    algorithms = ["DTDE with pt (comm.)", "DTDE (comm.)", "CTDE with CL (comm.)", "CTDE with EP (comm.)", "CTCE",
                  "RL+RL (comm.)", "RL+SETT", "MWKR+RL (comm.)", "MWKR+SETT"]
    df_selected = df[algorithms]

    mapping = {
        "MWKR+SETT": "MWKR+SETT",
        "RL+SETT": "RL+SETT",
        "MWKR+RL (comm.)": "MWKR+RL",
        "RL+RL (comm.)": "RL+RL",
        "CTCE": "CTCE",
        "DTDE (comm.)": "DTDE",
        "DTDE with pt (comm.)": "DIA-DTDE",
        "CTDE with CL (comm.)": "CTDE with CL",
        "CTDE with EP (comm.)": "CTDE with EP"
    }

    colors = {
        "MWKR+SETT": "#440154",  # Deep Purple
        "RL+SETT": "#3B528B",  # Indigo Blue
        "MWKR+RL": "#21908C",  # Teal Green
        "RL+RL": "#5DC863",  # Soft Green
        "CTCE": "#FDE725",  # Golden Yellow
        "DTDE": "#F9844A",  # Warm Orange
        "DIA-DTDE": "#D43E4F",  # Coral Red
        "CTDE with CL": "#728EA3",  # Slate Blue
        "CTDE with EP": "#A6AD00",  # Olive Green
    }

    max_columns = df_selected.idxmax(axis=1)
    max_counts = max_columns.value_counts()

    # 2. 알고리즘 이름 매핑 및 색상 준비
    mapped_counts = {}
    chart_colors = []

    for original_name, count in max_counts.items():
        mapped_name = mapping[original_name]
        mapped_counts[mapped_name] = count
        chart_colors.append(colors[mapped_name])

    # 3. Pie chart 그리기
    fig_width = 7
    fig_height = 5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    max_idx = max(mapped_counts.values())
    max_key = [k for k, v in mapped_counts.items() if v == max_idx][0]
    max_index = list(mapped_counts.keys()).index(max_key)

    base_rgb = mcolors.to_rgb(colors[max_key])
    h, l, s = colorsys.rgb_to_hls(*base_rgb)
    highlight_rgb = colorsys.hls_to_rgb(h, min(1, l + 0.3), min(1, s + 0.2))  # 더 밝게
    highlight_color = highlight_rgb

    explode = [0.02 if i == max_index else 0 for i in range(len(mapped_counts))]

    def func_autopct(pct):
        # 원하는 최소 비율 설정 (예: 5.0% 미만은 텍스트 숨기기)
        min_pct_threshold = 3.0

        if pct > min_pct_threshold:
            return f'{pct:.1f}%'  # 5% 이상이면 포맷된 문자열 반환
        else:
            return ''  # 5% 미만이면 빈 문자열 반환 (텍스트 숨김)

    wedges, texts, autotexts = ax.pie(
        mapped_counts.values(),
        colors=chart_colors,
        autopct=func_autopct,
        startangle=90,
        counterclock=False,
        explode=explode,
        pctdistance=0.7,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1},
        textprops={'fontsize': 20, 'color': 'black'}
    )

    wedges[max_index].set_edgecolor(highlight_color)
    wedges[max_index].set_linewidth(4)

    # 텍스트 테두리 효과 추가
    for autotext in autotexts:
        autotext.set_fontsize(20)
        autotext.set_color('white')
        autotext.set_path_effects([
            patheffects.withStroke(linewidth=2, foreground='black')
        ])

    # 제목 스타일
    ax.set_title(title[problem_size], fontsize=25)

    # 범례 위치 및 디자인
    used_algorithms = set(mapped_counts.keys())

    legend_labels = [
        mapping[algorithm]
        for algorithm in algorithms
    ]

    legend_colors = [
        colors[label] if label in used_algorithms else 'lightgray'
        for label in legend_labels
    ]

    legend_handles = [
        Patch(facecolor=color, edgecolor='white', label=label)
        for label, color in zip(legend_labels, legend_colors)
    ]

    # 범례 설정
    ax.legend(
        handles=legend_handles,
        loc='center left',
        bbox_to_anchor=(0.9, 0.5),
        frameon=False,
        fontsize=15
    )

    # 원형 유지 및 여백 조정
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    if save:
        file_dir = '../output/test/figures/MARL/'
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        fig.savefig(file_dir + 'pie chart(%s).png' % problem_size, dpi=300, bbox_inches='tight')


def draw_paired_bar_plot(save=False):
    problem_sizes = ["10-5", "15-5", "15-10", "20-10", "20-15", "25-15"]
    # algorithms_comm = ["RL+RL (comm.)", "DTDE (comm.)", "DTDE with pt (comm.)", "CTDE with CL (comm.)", "CTDE with EP (comm.)"]
    # algorithms_wo_comm = ["RL+RL", "DTDE", "DTDE with pt", "CTDE with CL", "CTDE with EP"]
    algorithms_comm = ["DTDE with pt (comm.)", "DTDE (comm.)", "CTDE with CL (comm.)", "CTDE with EP (comm.)", "RL+RL (comm.)"]
    algorithms_wo_comm = ["DTDE with pt", "DTDE", "CTDE with CL", "CTDE with EP", "RL+RL"]

    df_avg = pd.DataFrame(index=problem_sizes, columns=algorithms_comm + algorithms_wo_comm)
    for i, size in enumerate(problem_sizes):
        df = pd.read_excel('../output/test/results_scaled_by_rand+rand.xlsx',
                           sheet_name=size, engine="openpyxl", index_col=0).iloc[:-1]
        df_selected = df[algorithms_comm + algorithms_wo_comm]
        df_avg.iloc[i] = df_selected.mean()

    df_summary = df_avg.mean()
    df_comm = df_summary[algorithms_comm]
    df_wo_comm = df_summary[algorithms_wo_comm]

    # 색상 팔레트
    colors = {
        "comm": '#E74C3C',  # 빨강 계열
        "wo_comm": '#696969'  # 초록 계열
    }

    # 그래프 생성
    fig, ax = plt.subplots(figsize=(12, 8))

    # 알고리즘 이름에서 " (comm.)" 제거한 버전 (x축 레이블용)
    mapping = {
        "RL+RL (comm.)": "RL+RL",
        "DTDE (comm.)": "DTDE",
        "DTDE with pt (comm.)": "DIA-DTDE",
        "CTDE with CL (comm.)": "CTDE with CL",
        "CTDE with EP (comm.)": "CTDE with EP"
    }

    clean_algorithm_names = [mapping[name] for name in algorithms_comm]
    # clean_algorithm_names = [name.replace(" (comm.)", "") for name in algorithms_comm]

    # 막대 위치 설정
    x = np.arange(len(clean_algorithm_names))
    width = 0.35

    # 막대 그래프 그리기
    bars1 = ax.bar(x - width / 2, df_wo_comm.values, width,
                   label='Without Communication', color=colors['wo_comm'],
                   alpha=0.8, edgecolor='white', linewidth=1.5)

    bars2 = ax.bar(x + width / 2, df_comm.values, width,
                   label='Communication', color=colors['comm'],
                   alpha=0.8, edgecolor='white', linewidth=1.5)

    def add_value_labels(bars, values):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{value:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center',
                        va='bottom',
                        fontsize=15)

    add_value_labels(bars1, df_wo_comm.values)
    add_value_labels(bars2, df_comm.values)

    # 그래프 꾸미기
    # ax.set_xlabel("Algorithms", fontsize=20)
    ax.set_ylabel("Performance Measure ($\sigma$)", fontsize=20)

    ax.set_xticks(x)
    ax.set_xticklabels(clean_algorithm_names, ha='center', fontsize=20)

    ax.set_ylim(0, 35)
    yticklabels = ax.get_yticklabels()
    ax.set_yticklabels(yticklabels, fontsize=20)

    # 범례
    ax.legend(
        loc='upper right',
        frameon=False,
        fontsize=15
    )

    # 격자 표시
    ax.grid(True, axis='y', alpha=0.5, linestyle='--')

    # 레이아웃 조정
    plt.tight_layout()
    plt.show()

    if save:
        file_dir = '../output/test/figures/MARL/'
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        fig.savefig(file_dir + 'paired bar chart.png', dpi=300, bbox_inches='tight')


def draw_line_graph(save=False):
    problem_sizes = ["10-5", "15-5", "15-10", "20-10", "20-15", "25-15"]
    # algorithms = ["SPT+SETT", "RL+SETT", "MWKR+RL (comm.)", "RL+RL (comm.)", "CTCE",
    #               "DTDE (comm.)", "DTDE with pt (comm.)", "CTDE with CL (comm.)", "CTDE with EP (comm.)"]
    algorithms = ["DTDE with pt (comm.)", "DTDE (comm.)", "CTDE with CL (comm.)", "CTDE with EP (comm.)", "CTCE",
                  "RL+RL (comm.)", "RL+SETT", "MWKR+RL (comm.)", "MWKR+SETT"]

    df_list = []
    for i, size in enumerate(problem_sizes):
        df = pd.read_excel('../output/test/results_computing_time.xlsx',
                           sheet_name=size, engine="openpyxl", index_col=0)
        df_selected = df[["num_operations"] + algorithms].reset_index(drop=True)
        df_list.append(df_selected)
    df_total = pd.concat(df_list, axis=0)

    df_group = df_total.groupby(by="num_operations").mean()

    mapping = {
        "MWKR+SETT": "MWKR+SETT",
        "RL+SETT": "RL+SETT",
        "MWKR+RL (comm.)": "MWKR+RL",
        "RL+RL (comm.)": "RL+RL",
        "CTCE": "CTCE",
        "DTDE (comm.)": "DTDE",
        "DTDE with pt (comm.)": "DIA-DTDE",
        "CTDE with CL (comm.)": "CTDE with CL",
        "CTDE with EP (comm.)": "CTDE with EP"
    }

    colors = {
        "MWKR+SETT": "#440154",  # Deep Purple
        "RL+SETT": "#3B528B",  # Indigo Blue
        "MWKR+RL": "#21908C",  # Teal Green
        "RL+RL": "#5DC863",  # Soft Green
        "CTCE": "#FDE725",  # Golden Yellow
        "DTDE": "#F9844A",  # Warm Orange
        "DIA-DTDE": "#D43E4F",  # Coral Red
        "CTDE with CL": "#728EA3",  # Slate Blue
        "CTDE with EP": "#A6AD00",  # Olive Green
    }

    # 그래프 생성
    fig, ax = plt.subplots(figsize=(12, 8))

    # 실제 num_operations 값을 x축으로 사용
    x_values = df_group.index.values

    # 각 알고리즘별로 선 그래프 그리기
    for algorithm in algorithms:
        mapped_name = mapping[algorithm]
        y_values = df_group[algorithm].values

        ax.plot(x_values, y_values,
                label=mapped_name,
                color=colors[mapped_name],
                linewidth=2,
                markersize=6)

    # x축 라벨 설정 (45부터 130까지 5단위로 동일 간격 표시)
    desired_ticks = list(range(40, 136, 5))  # 45, 50, 55, ..., 130
    ax.set_xticks(desired_ticks)
    ax.set_xticklabels(desired_ticks, fontsize=20)
    ax.set_xlim(40, 135)

    # y축
    ax.set_ylim(0, 25)
    yticklabels = ax.get_yticklabels()
    ax.set_yticklabels(yticklabels, fontsize=20)

    # 축 라벨 및 제목 설정
    ax.set_xlabel('Number of Operations', fontsize=20)
    ax.set_ylabel('Computing Time (s)', fontsize=20)

    # 범례 설정
    ax.legend(loc='upper center',  # 기준 위치 (anchor 기준에서 위쪽 가운데)
              bbox_to_anchor=(0.5, -0.15),  # (x, y) → x는 가운데(0.5), y는 아래쪽 바깥 (-0.15)
              ncol=5,  # 범례 항목을 가로로 배열할 때 열 수 지정 (선택)
              frameon=False,
              fontsize=15)

    # ax.legend(loc='upper left',  # 위치를 '왼쪽 상단'으로 변경
    #           ncol=5,  # 열 개수를 2개로 조절 (1도 가능)
    #           frameon=False,
    #           fontsize=15)

    # 격자 추가-
    ax.grid(True, linestyle='--', alpha=0.7)

    # 레이아웃 조정
    plt.tight_layout()
    plt.show()

    if save:
        file_dir = '../output/test/figures/MARL/'
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        fig.savefig(file_dir + 'line plot.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    problem_sizes = ["10-5", "15-5", "15-10", "20-10", "20-15", "25-15"]
    for size in problem_sizes:
        draw_pie_chart(problem_size=size, save=True)
        draw_bar_plot(problem_size=size, tag="CT", save=True)
        draw_bar_plot(problem_size=size, tag="FJSP", save=True)

    draw_boxplot_SARL(problem_size="20-10", tag="CT", save=True)
    draw_boxplot_SARL(problem_size="20-10", tag="FJSP", save=True)
    draw_boxplot_MARL(problem_size="20-10", save=True)
    draw_scatter_plot(save=True)
    draw_paired_bar_plot(save=True)
    draw_line_graph(save=True)
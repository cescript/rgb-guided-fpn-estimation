import os
import matplotlib.pyplot as plt
from collections import defaultdict

# read the data and return algorithm_names vs scores
def read_scores(filename, sort_result=True, ignore_safta=False):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f]
    
    # get the metric names from the first line
    metric_names = lines[0][1:].split(',')[1:]
    metric_names = [m.strip() for m in metric_names]
    
    # extract algorithm names and scores
    raw_data = []
    for i in range(0, len(lines), 2):
        name = lines[i][1:].split(',')[0].strip()
        
        # ignore safta and use SAFTA-RGB as SAFTA
        if ignore_safta:
            if name == "SAFTA":
                continue
            name = name.replace("SAFTA-RGB", "SAFTA")

        values = [float(v.strip()) for v in lines[i + 1].split(',')]
        raw_data.append((name, values))
    
    # sort by first metric in descending order
    if sort_result:
        raw_data.sort(key=lambda x: x[1][0], reverse=True)
    
    algorithm_names = [name for name, _ in raw_data]
    scores = {m: [] for m in metric_names}
    for _, values in raw_data:
        for j, m in enumerate(metric_names):
            scores[m].append(values[j])
    
    return algorithm_names, scores

# save figures for selected metric
def save_figures(algorithm_names, scores, selected_metric, title, rotation_angle=30):
    # plot each metric
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(algorithm_names, scores[selected_metric], color='#7fcdbb')
    ax.set_title(title)
    ax.set_ylabel(f'{selected_metric.upper()}')
    ax.set_xticks(range(len(algorithm_names)))
    ax.set_xticklabels(algorithm_names, rotation=rotation_angle, ha='center', fontfamily='monospace')
    
    # Add value labels on top of bars
    for bar, value in zip(bars, scores[selected_metric]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height * 0.9, f"{value:.2f}",
                ha='center', va='top', fontsize=12, color='#f7fcb9')
    
    fig.tight_layout()
    figure_name = os.path.join("output", f"{title.replace(' ', '_').lower()}_{selected_metric}.png")
    fig.savefig(figure_name, dpi=300)

# for ablation study to look better
def format_algorithm_label(code):
    parts = code.split('_')
    names = {"rgb": "RGB", "irn": "IRN", "fpn": "FPN"}
    symbols = {"1": "✓", "0": "✗"}

    lines = []
    for part in parts:
        lines.append(f"{names[part[:-1]]}: {symbols[part[-1]]}")
    return "\n".join(lines)

def print_formatted(algorithm_names_and_k, scores, selected_metric, title):
    # split algorithm names and k's
    grouped = defaultdict(list)
    for idx, item in enumerate(algorithm_names_and_k):
        # get algorithm name and k
        aname, k = item.rsplit("-", 1)
        grouped[aname].append({"k": int(k), "score": scores[selected_metric][idx]})
    
    # now print thr results
    alg_names = sorted(grouped.keys())
    print(f"{title}")
    for aname in alg_names:
        print(f"{aname}: ", end=' & ')
        sorted_data = sorted(grouped[aname], key=lambda d: d["k"])
        for item in sorted_data:
            print(f"{item['score']: .2f}", end=' & ')
        print("")
    
# draw figures for all tests
if __name__ == '__main__':
    selected_metric = "psnr"
    
    # get the results for FPN
    title = "Model Performance for FPN"
    algorithm_names, scores = read_scores("output/model_comparison/scores.txt", ignore_safta=True)
    save_figures(algorithm_names, scores, selected_metric, title)

    # get the results for HFN
    title = "Model Performance for HFN"
    algorithm_names, scores = read_scores("output/model_comparison_hfn/scores.txt", ignore_safta=True)
    save_figures(algorithm_names, scores, selected_metric, title)

    # get the results for FPN
    title = "Ablation Performance on FPN"
    algorithm_names, scores = read_scores("output/ablation_study/scores.txt")
    algorithm_names = [format_algorithm_label(n) for n in algorithm_names]
    save_figures(algorithm_names, scores, selected_metric, title, 0)

    # get the results for HFN
    title = "Ablation Performance on HFN"
    algorithm_names, scores = read_scores("output/ablation_study_hfn/scores.txt")
    algorithm_names = [format_algorithm_label(n) for n in algorithm_names]
    save_figures(algorithm_names, scores, selected_metric, title, 0)

    # get the results for FPN
    title = "OLS vs FPN Estimator Performance for FPN"
    algorithm_names, scores = read_scores("output/ols_comparison/scores.txt", ignore_safta=True)
    save_figures(algorithm_names, scores, selected_metric, title)

    # get the results for HFN
    title = "OLS vs FPN Estimator Performance for HFN"
    algorithm_names, scores = read_scores("output/ols_comparison_hfn/scores.txt", ignore_safta=True)
    save_figures(algorithm_names, scores, selected_metric, title)
    
    # get the results for FPN vs K
    title = "Effect of Sequence Length on FPN"
    algorithm_names, scores = read_scores("output/aggregation_comparison/scores.txt", sort_result=False, ignore_safta=True)
    print_formatted(algorithm_names, scores, selected_metric, title)

    # get the results for HFN
    title = "Effect of Sequence Length on HFN"
    algorithm_names, scores = read_scores("output/aggregation_comparison_hfn/scores.txt", sort_result=False, ignore_safta=True)
    print_formatted(algorithm_names, scores, selected_metric, title)
    
    

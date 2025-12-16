import os
import matplotlib.pyplot as plt
from collections import defaultdict

# read the data and return algorithm_names vs scores
def read_scores(logfile, append_k=0):
    # get each line from file content
    content = open(logfile, "r")
    lines = [line.strip() for line in content]
    
    # get the metric names from the first line
    metric_names = lines[0][1:].split(',')[1:]
    metric_names = [m.strip() for m in metric_names]

    # extract algorithm names and scores
    raw_data = []
    for i in range(0, len(lines), 2):
        name = lines[i][1:].split(',')[0].strip()
        # make SAFTA, SAFTA-RGB conversion
        if name == "SAFTA":
            name = "SAFTA-NONE"

        # append K into algorithm name
        if append_k > 0:
            name = f"{name}K{append_k}"

        # push name and values
        values = [float(v.strip()) for v in lines[i + 1].split(',')]
        raw_data.append((name, values))

    algorithm_names = [name for name, _ in raw_data]
    scores = {m: [] for m in metric_names}
    for _, values in raw_data:
        for j, m in enumerate(metric_names):
            scores[m].append(values[j])

    return algorithm_names, scores

def write_score_table_txt(filename, algorithm_names, scores, metrics=("psnr", "ssim", "gmsd")):
    with open(filename, "w") as f:
        # header of the table
        f.write(",".join(["metric"] + algorithm_names) + "\n")
        # rows are the scores
        for m in metrics:
            values = scores[m]
            row = [m] + [f"{v:.3f}" for v in values]
            f.write(",".join(row) + "\n")

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
    figure_path = os.path.join("output", "figures")
    filename = f"{title.replace(' ', '_').lower()}_{selected_metric}"

    # create output directory for the images and logs
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    # dump image and log
    write_score_table_txt(os.path.join(figure_path, f"{filename}.txt"), algorithm_names, scores)
    fig.savefig(os.path.join(figure_path, f"{filename}.png"), dpi=300)

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
            print(f"{item['score']: .3f}", end=' & ')
        print("")
    
# draw figures for all tests
if __name__ == '__main__':

    ################# DO COMPARISON ANALYSIS ON M3FD #################
    # get the results for FPN
    title = "Model Performance for FPN on M3FD"
    logpath = "output/comparison/m3fd_config_K_12_fpn/scores.txt"
    if os.path.isfile(logpath):
        algorithm_names, scores = read_scores(logpath)
        save_figures(algorithm_names, scores, "psnr", title)

    # get the results for HFN
    title = "Model Performance for HFN on M3FD"
    logpath = "output/comparison/m3fd_config_K_12_hfn/scores.txt"
    if os.path.isfile(logpath):
        algorithm_names, scores = read_scores(logpath)
        save_figures(algorithm_names, scores, "psnr", title)

    ################# DO COMPARISON ANALYSIS ON MSRS #################
    # get the results for FPN
    title = "Model Performance for FPN on MSRS"
    logpath = "output/comparison/msrs_config_K_12_fpn/scores.txt"
    if os.path.isfile(logpath):
        algorithm_names, scores = read_scores(logpath)
        save_figures(algorithm_names, scores, "psnr", title)

    # get the results for HFN
    title = "Model Performance for HFN  on MSRS"
    logpath = "output/comparison/msrs_config_K_12_hfn/scores.txt"
    if os.path.isfile(logpath):
        algorithm_names, scores = read_scores(logpath)
        save_figures(algorithm_names, scores, "psnr", title)

    ################# DO ABLATION ANALYSIS #################
    # get the results for FPN
    title = "Ablation Performance on FPN"
    logpath = "output/ablation/m3fd_config_K_12_fpn/scores.txt"
    if os.path.isfile(logpath):
        algorithm_names, scores = read_scores(logpath)
        save_figures(algorithm_names, scores, "psnr", title, 0)

    # get the results for HFN
    title = "Ablation Performance on HFN"
    logpath = "output/ablation/m3fd_config_K_12_hfn/scores.txt"
    if os.path.isfile(logpath):
        algorithm_names, scores = read_scores(logpath)
        save_figures(algorithm_names, scores, "psnr", title, 0)

    
    

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
            name = "SAFTA-NIL"
        if name == "EMPTY":
            name = "NOISY"

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

# return tble for comparison
def get_metric_table(logpath):
    header = ["metric"]
    rows = {}
    if os.path.isfile(logpath):
        alg_names, scores = read_scores(logpath)
        header.extend(alg_names)
        for score in scores.keys():
            rows.setdefault(score, []).extend(scores[score])
    return header, rows

# return table for k effect
def get_k_table(logpaths):
    rows = {}
    for logpath in logpaths:
        if os.path.isfile(logpath):
            alg_names, alg_scores = read_scores(logpath)
            for idx, alg_name in enumerate(alg_names):
                rows.setdefault(alg_name, []).append(alg_scores["psnr"][idx])
    return rows

def write_score_table_txt(filename, header, rows):
    with open(filename, "w") as f:
        # header of the table
        f.write(",".join(header) + "\n")
        # rows are the scores
        for key in rows.keys():
            if key == "ssim" or key == "gmsd":
                row = [key] + [f"{v:.3f}" for v in rows[key]]
            else:
                row = [key] + [f"{v:.2f}" for v in rows[key]]
            f.write(",".join(row) + "\n")

# save figures for selected metric
def save_figures(figure_path, headers, rows, selected_key, title, rotation_angle=30):
    # plot each metric
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(headers[1:], rows[selected_key], color='#7fcdbb')
    ax.set_title(title)
    ax.set_ylabel(f'{selected_key.upper()}')
    ax.set_xticks(range(len(headers[1:])))
    ax.set_xticklabels(headers[1:], rotation=rotation_angle, ha='center', fontfamily='monospace')
    
    # Add value labels on top of bars
    for bar, value in zip(bars, rows[selected_key]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height * 0.9, f"{value:.2f}",
                ha='center', va='top', fontsize=12, color='#f7fcb9')
    
    fig.tight_layout()
    filename = f"{title.replace(' ', '_').lower()}_{selected_key}"
    fig.savefig(os.path.join(figure_path, f"{filename}.png"), dpi=300)
    plt.close(fig)

def write_scores_and_figures(headers, rows, title, rotation_angle=30):
    # create output directory for the images and logs
    figure_path = os.path.join("output", "figures")
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    # save figures for each key
    for key in rows:
        save_figures(figure_path, headers, rows, key, title, rotation_angle)

    # save table
    filename = os.path.join(figure_path, f"{title.replace(' ', '_').lower()}.txt")
    write_score_table_txt(filename, headers, rows)

# draw figures for all tests
if __name__ == '__main__':

    ################# DO COMPARISON ANALYSIS ON M3FD #################
    # get the results for FPN
    title = "Model Performance for FPN on M3FD"
    logpath = "output/comparison/m3fd_config_K_12_fpn/scores.txt"
    header, rows = get_metric_table(logpath)
    write_scores_and_figures(header, rows, title, 0)

    # get the results for HFN
    title = "Model Performance for HFN on M3FD"
    logpath = "output/comparison/m3fd_config_K_12_hfn/scores.txt"
    header, rows = get_metric_table(logpath)
    write_scores_and_figures(header, rows, title, 0)

    ################# DO COMPARISON ANALYSIS ON MSRS #################
    # get the results for FPN
    title = "Model Performance for FPN on MSRS"
    logpath = "output/comparison/msrs_config_K_12_fpn/scores.txt"
    header, rows = get_metric_table(logpath)
    write_scores_and_figures(header, rows, title, 0)

    # get the results for HFN
    title = "Model Performance for HFN  on MSRS"
    logpath = "output/comparison/msrs_config_K_12_hfn/scores.txt"
    header, rows = get_metric_table(logpath)
    write_scores_and_figures(header, rows, title, 0)

    ################# DO ABLATION ANALYSIS #################
    # get the results for FPN
    title = "Ablation Performance on FPN"
    logpath = "output/ablation/m3fd_config_K_12_fpn/scores.txt"
    header, rows = get_metric_table(logpath)
    write_scores_and_figures(header, rows, title, 0)

    # get the results for HFN
    title = "Ablation Performance on HFN"
    logpath = "output/ablation/m3fd_config_K_12_hfn/scores.txt"
    header, rows = get_metric_table(logpath)
    write_scores_and_figures(header, rows, title, 0)

    ################# DO K EFFECT ANALYSIS #################
    title = "Effect of K on FPN"
    logpaths = []
    header = ["algorithm"]
    for k in range(2, 22, 2):
        logpaths.append(f"output/k_effect/m3fd_config_K_{k}_fpn/scores.txt")
        header.append(f"K={k}")
    rows = get_k_table(logpaths)
    write_scores_and_figures(header, rows, title, 0)

    # effect on hfn
    title = "Effect of K on HFN"
    logpaths = []
    header = ["algorithm"]
    for k in range(2, 22, 2):
        logpaths.append(f"output/k_effect/m3fd_config_K_{k}_hfn/scores.txt")
        header.append(f"K={k}")
    rows = get_k_table(logpaths)
    write_scores_and_figures(header, rows, title, 0)

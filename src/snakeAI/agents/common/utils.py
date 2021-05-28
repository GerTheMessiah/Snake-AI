import sys
from os import listdir
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import linregress


def plot_learning_curve(x, apples, scores, figure_file):
    plt.title("Learning Results")
    plt.xlabel("Iterations of ten games")
    plt.ylabel("Apple_mean / Score_mean")
    plt.plot(x, apples, color='red')
    plt.plot(x, scores, color='green')
    m, b, _, _, _ = linregress(x, apples)
    m2, b2, _, _, _ = linregress(x, scores)
    plt.plot(x, [m * y + b for y in x], color='blue')
    plt.plot(x, [m2 * y + b2 for y in x], color='purple')
    red_patch = mpatches.Patch(color='red', label=f'Apple_mean: {round(mean(apples), 3)}')
    green_patch = mpatches.Patch(color='green', label=f'Score_mean: {round(mean(scores), 3)}')
    blue_patch = mpatches.Patch(color='blue', label=f'Apple_reg m: {round(m, 3)}, b: {round(b, 3)}')
    purple_patch = mpatches.Patch(color='purple', label=f'Score_reg m: {round(m2, 3)}, b: {round(b2, 3)}')
    plt.legend(handles=[red_patch, green_patch, blue_patch, purple_patch], loc=1)
    plt.savefig(figure_file)


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=50):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\r\n')
    sys.stdout.flush()


def file_path(dir: str, new_save: bool, file_name: str = "model"):
    MODEL_DIR_PATH = str(Path(__file__).parent.parent.parent.parent) + fr"\resources\{dir}"
    try:
        MODEL_ID = max([int(item[6]) for item in listdir(MODEL_DIR_PATH)])
    except Exception:
        print("Loading model failed")
        return rf"{MODEL_DIR_PATH}\{file_name}_0"
    return rf"{MODEL_DIR_PATH}\{file_name}_{MODEL_ID + 1 if new_save else MODEL_ID}"

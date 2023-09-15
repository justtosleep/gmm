import argparse
import pandas as pd
import matplotlib.pyplot as plt


# read command line arguments
parser = argparse.ArgumentParser(description="GaussianMixture Clustering")

# add positional arguments
parser.add_argument("dataset", type=str, help="name of dataset")

# add optional arguments
parser.add_argument("--method", type=str, default="", help="name of dataset processing method")

args = parser.parse_args()
domain = args.dataset
method = args.method

name = domain+"_"+method if method != "" else domain
kmean = name+"_kmean"
metrics_dir = "./metrics/"+name+"/measure"
kmean_metrics_dir = "./metrics/"+kmean+"/measure/"
input_dir = metrics_dir + "/"
inprob_dir = metrics_dir + "_prob/"

colors = ["blue", "black", "purple", "green", "red", "cyan", "magenta", "yellow", "pink", "brown"]
col_names = ["mse", "mae"] #
markers = ["o", "v", "^", "<", ">", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D", "d", "|", "_"]
csv_files = []

csv_files.append(input_dir+"full.csv")
# csv_files.append(inprob_dir+"full.csv")
csv_files.append(kmean_metrics_dir+"kmean.csv")

width = 12
height = 4
plt.figure(figsize=(width, height))
def col_plot(n, m, i, x, y, label, color, xlabel='k', ylabel=''):
    ax = plt.subplot(n, m, i, title=ylabel)
    ax.plot(x, y, label=label, color=color)
    ax.grid(True)
    ax.legend()
    fontsize = 14
    ax.set_xlabel(xlabel, fontsize=fontsize)
    # ax.set_ylabel(ylabel, fontsize=fontsize)
    plt.sca(ax)

#read csv files and plot
# labels = ["measure", "measure(prob)"]
labels = ["gmm", "kmean"]
for i, csv_file in enumerate(csv_files):
    print(f"Reading {csv_file}...")
    try:
        df = pd.read_csv(csv_file)
        xlabel = 'n_components'
        x = df[xlabel]
        # print(f"Plotting {x}")
        label = labels[i]
        for j, col_name in enumerate(col_names):
            if col_name in df.columns:
                print(f"Plotting {label} {col_name}...")
                y = df[col_name]
                col_plot(1, 2, j+1, x, y, label, colors[i], xlabel, col_name)
    except:
        print(f"Error reading {csv_file}!")

output_path = "./vil_graph/"+name+".png"
plt.savefig(output_path)

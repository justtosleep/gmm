import pandas as pd
import matplotlib.pyplot as plt


domain = "UCLAdult"
famd = domain+"_famd"
sparse = domain+"_sparse"
norm105 = domain+"_norm105"
mix = domain+"_mix"
kmean = norm105+"_kmean"
kfreq = domain+"_kfreq"
input_dir = "./metrics/{}/measure/full.csv"

colors = ["red", "blue", "green", "black", "purple", "orange", "gray", "pink", "brown"]
col_names = ["mse", "mae"] #
markers = ["o", "v", "^", "<", ">", "s", "p", "P", "*", "h"]
csv_files = []

csv_files.append(input_dir.format(famd))
csv_files.append(input_dir.format(sparse))
csv_files.append(input_dir.format(norm105))
csv_files.append(input_dir.format(mix))
csv_files.append("./metrics/{}/measure/kmean.csv".format(kmean))
csv_files.append("./metrics/{}/measure/kfreq.csv".format(kfreq))
# csv_files.append(input_dir+mix+"/full.csv")


width = 12
height = 4
plt.figure(figsize=(width, height))
def col_plot(n, m, i, x, y, label, color, xlabel='k', ylabel=''):
    ax = plt.subplot(n, m, i, title=ylabel)
    ax.plot(x, y, label=label, color=color)
    ax.grid(True)
    ax.legend()
    ax.set_xlabel(xlabel)
    # ax.set_ylabel(ylabel)
    plt.sca(ax)

# read csv files and plot
labels = ["GMM(famd)", "GMM(sparse)", "GMM(norm105)", "GMM(mix)", "KMean(norm105)", "KFreq(sparse)"]
print("Start reading data")
for i, csv_file in enumerate(csv_files):
    print(f"Reading {csv_file}...")
    try:
        df = pd.read_csv(csv_file)
        xlabel = 'n_components'
        x = df[xlabel]
        label = labels[i]
        for j, col_name in enumerate(col_names):
            if col_name in df.columns:
                print(f"Plotting {label} {col_name}...")
                y = df[col_name]
                col_plot(1, 2, j+1, x, y, label, colors[i], xlabel, col_name)
    except:
        print(f"Error reading {csv_file}!")

output_path = "./vil_graph/"+domain+".png"
plt.savefig(output_path)


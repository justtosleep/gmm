import numpy as np

def read_bin_data(n, d, fname):
    fname = "../dataset/" + fname
    try:
        with open(fname, "rb") as fp:
            data = np.fromfile(fp, dtype=np.float32, count=n * d)
            data = data.reshape((n, d))
            return data, 0  # Successful read, return data and status code 0
    except FileNotFoundError:
        print(f"Could not open {fname}")
        return None, 1  # Failed to open file, return None and status code 1

# Example usage:
datasets = {
    # "Gist":{
    #     "n": 982694,
    #     "d": 960,
    #     "fname": "Gist/Gist.ds"
    # },
    # "Trevi":{
    #     "n": 100900,
    #     "d": 4096,
    #     "fname": "Trevi/Trevi.ds"
    # },
    # "Tiny":{
    #     "n": 1000000,
    #     "d": 384,
    #     "fname": "Tiny/Tiny1M.ds"
    # },
    "Cifar":{	
        "n": 50000,
        "d": 512,
        "fname": "Cifar/Cifar.ds"
    },
    "LabelMe":{
        "n": 181093,
        "d": 512,
        "fname": "LabelMe/LabelMe.ds"
    },
}

for dataset in datasets.keys():
    print(f"Reading {dataset} dataset...")
    data, status = read_bin_data(datasets[dataset]["n"], datasets[dataset]["d"], datasets[dataset]["fname"])
    if status == 0:
        print(f"Successfully read {dataset} dataset.")
        headers = [f"{i}" for i in range(1, datasets[dataset]["d"] + 1)]
        data_path = datasets[dataset]["fname"].replace('.ds', '.data')
        np.savetxt(f"../dataset/{data_path}", data, delimiter=",", header=",".join(headers), comments="")
    else:
        print(f"Error occurred while reading {dataset} dataset.")


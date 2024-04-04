from matplotlib import pyplot as plt
import numpy as np
import csv
import os
reward_scale = 1
#multiplot each csv of the directory. each csv has a Step header and Value Header
directory = "./DDPG"
# get files from OS
files = os.listdir(directory)
# filter out non-csv files
files = [f for f in files if f.endswith(".csv")]
# sort filenames 
files.sort()
# create subplots for each file, stacked vertically with shared axis
fig, axs = plt.subplots(len(files), sharex=True, sharey=False)
fig.supxlabel("Steps")
fig.tight_layout()
# iterate over each file
for i, file in enumerate(files):
    # open file
    with open(f"{directory}/{file}") as f:
        # read csv
        reader = csv.DictReader(f)
        # get headers
        headers = reader.fieldnames
        # get data
        data = list(reader)
        # get x (Step) and y (Value)
        x = [int(row["Step"]) for row in data]
        y = [float(row["Value"]) for row in data]
        if "reward" in file:
            y = [v*reward_scale for v in y]
        # plot data
        axs[i].plot(x, y)
        axs[i].legend([file.removesuffix(".csv")])
        # set title
        # axs[i].set_title(file.removesuffix(".csv"))
        
        # stop scientific notatition in X scale
        axs[i].get_xaxis().get_major_formatter().set_scientific(False)
# set common x and y labels

plt.show()

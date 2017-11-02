USAGE_MESSAGE = """
Usage:
For generatating all plots (with different settings like log scale) and save them, call:
$ python plotting_tools.py all path/to/input_filename.csv plot_title
For displaying one plot (not saving), call:
$ python plotting_tools.py path/to/input_filename.csv optional_plot_title
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Directory for automatically saving plots
OUTPUT_PATH = "data/plots"

# Names of labels displayed on the legend box of plots
DIST_LABEL1 = "signal"
DIST_LABEL2 = "background"

def check_dir_exists(path):
    """
    Automatically create folders if it doesn't exist.
    Prevent errors during output.
    """
    import os
    path = os.path.join(os.path.dirname(__file__), path)
    if not os.path.exists(path):
        os.makedirs(path)

def format_out_filename(xlabel, normed, xlog, ylog, nbin):
    """
    Format filename of plots in following manner:
        e.g. Absolute Isolation_orig_xlog_ylog_100bin.png
    """
    namebuffer = [xlabel]
    if normed:
        namebuffer.append("norm")
    else:
        namebuffer.append("orig")
    if xlog:
        namebuffer.append("xlog")
    else:
        namebuffer.append("xlin")
    if ylog:
        namebuffer.append("ylog")
    else:
        namebuffer.append("ylin")
    namebuffer.append("{}bin".format(nbin))
    return "_".join(namebuffer)+".png"

def plot_distribution(data, xlabel="", title="Signal/Background Distribution", nbin=100, normed=False, xlog=False, ylog=False, save=False, noshow=False):

    if xlog:
        # Accomodating for 0 values if x is in log scale
        data[:,0] += 1

    # Format of separated_data : [signal_data, background_data]
    separated_data = [data[data[:,1]==each,0] for each in [1,0]]
    minval, maxval = np.min(data[:,0]), np.max(data[:,0])

    # Create linear or logged bins for histogram
    if xlog:
        bins = np.logspace(np.log10(minval), np.log10(maxval), nbin)
    else:
        bins = np.linspace(minval, maxval, nbin)

    # Plotting histograms for two distributions
    fig, ax = plt.subplots()
    for each, label in zip(separated_data, ["signal", "background"]):
        plt.hist(each, bins, alpha=0.5, normed=normed, label=label)

    # Configuring log scale on x-axis
    if xlog:
        plt.xscale("log")
        title += "(logged x)"
        # Decreasing tick labels by a factor of 10
        xlim = np.log10(ax.get_xlim())
        xlim[0], xlim[1] = np.ceil(xlim[0]), np.floor(xlim[1])+1
        tick_pos, real_val = 10**np.arange(*xlim), 10**np.arange(*(xlim-1))
        # Change label "0.1" to "0"
        for i in range(len(real_val)):
            if real_val[i] == 0.1:
                real_val[i] = 0
                break
        # Replace tick labels
        plt.xticks(tick_pos, real_val)
    # Configuring log scale on y-axis
    if ylog:
        plt.yscale("log")
        title += "(logged y)"
    # Change labels if histogram is normalized
    if normed:
        title += "(normed y)"
        plt.ylabel("Proportion")
    else:
        plt.ylabel("Frequency")

    plt.legend()
    plt.xlabel(xlabel)
    plt.title(title)

    # Whether to display or save figures
    if not noshow:
        plt.show()

    if save:
        check_dir_exists(OUTPUT_PATH)
        path = OUTPUT_PATH+"/"+format_out_filename(xlabel, normed, xlog, ylog, nbin)
        plt.savefig(path, dpi=200)

def generate_all_plots(data, xlabel, nbin=100):
    """
    Iterate through all possible combinations of `normed`, `xlog` and `ylog`
    settings. Generate 2^3 plots. Save to OUTPUT_PATH. Not showing pop-up plots.
    """
    from itertools import product
    for normed, xlog, ylog in product([True, False], repeat=3):
        plot_distribution(data, xlabel=xlabel,
            normed=normed, xlog=xlog, ylog=ylog, nbin=nbin,
            save=True, noshow=True)

if __name__=="__main__":
    import sys

    # Fast operations from commandline
    if len(sys.argv)>1:
        if sys.argv[1] == "all":
            try:
                filename=sys.argv[2]
                xlabel = sys.argv[3]
            except:
                raise ValueError("Not enough arguments. See Usage below.\n{}".format(USAGE_MESSAGE))
            data = np.loadtxt(filename)
            generate_all_plots(data, xlabel)
        else:
            try:
                filename=sys.argv[1]
                xlabel = sys.argv[2]
            except:
                raise ValueError("Not enough arguments. See Usage below.\n{}".format(USAGE_MESSAGE))
                print USAGE_MESSAGE
            data = np.loadtxt(filename)
            plot_distribution(data, xlabel=xlabel, normed=True, xlog=False, ylog=False, nbin=100)
    else:
        pass
        # Make one plot with log scale on y-axis.
        #plot_distribution(data, xlabel="Absolute Isolation", normed=False, xlog=False, ylog=True, nbin=100)

        # Meta function that automatically generates 8 plots and saves them.
        #generate_all_plots(data, "Absolute Isolation")
        #generate_all_plots(data, "Relative Isolation")

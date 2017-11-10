import numpy as np
from preprocess import check_dir_exists

SESSION_TASKS = ["dl"]

INPUT_PATH = "data"
INPUT_FILES = {
"abs":"abs_iso_with_labels.csv",
"rel":"rel_iso_with_labels.csv",
"dl": "deeplearning_with_labels.csv"
}
OUTPUT_PATH = "data"
OUTPUT_FILES = {
"abs":"abs_iso_roc_data.csv",
"rel":"rel_iso_roc_data.csv",
"dl": "dl_roc_data.csv"
}

def read_labeled_data(filepath):
    try:
        data = np.loadtxt(filepath)
    except ValueError:
        data = np.loadtxt(filepath, delimiter=",")
    return data

def calculate_roc(data, ndots=1000, reverse=False):
    """
    DATA FORMAT:
    Column 1: isolation values/classification probabilities
    Column 2: labels, either 1 or 0
        1: signal
        0: background
    """

    signal_data = data[:,0][data[:,1]==1]# Take values when label=1
    backgr_data = data[:,0][data[:,1]==0]# Take values when label=0

    minval, maxval = np.min(data[:,0]), np.max(data[:,0])

    total_signal = len(signal_data)
    total_backgr = len(backgr_data)

    signal_data = np.sort(signal_data)
    backgr_data = np.sort(backgr_data)

    signal_count = np.empty(ndots)
    backgr_count = np.empty(ndots)

    ps = 0
    pb = 0
    i = 0
    cutoff = np.linspace(minval, maxval, ndots)
    if reverse:
        signal_data = signal_data[::-1]
        backgr_data = backgr_data[::-1]
        cutoff = cutoff[::-1]
        while i < ndots:
            while (ps < total_signal) and (signal_data[ps] > cutoff[i]):
                ps += 1
            while (pb < total_backgr) and (backgr_data[pb] > cutoff[i]):
                pb += 1
            signal_count[i] = ps
            backgr_count[i] = pb
            i += 1
    else:
        while i < ndots:
            while (ps < total_signal) and (signal_data[ps] < cutoff[i]):
                ps += 1
            while (pb < total_backgr) and (backgr_data[pb] < cutoff[i]):
                pb += 1
            signal_count[i] = ps
            backgr_count[i] = pb
            i += 1

    signal_eff = signal_count/float(total_signal)
    backgr_eff = backgr_count/float(total_backgr)

    return backgr_eff, signal_eff, cutoff

def concat(roc):
    shape = (len(roc[0]), 3)
    combined = np.empty(shape)
    for i in range(3):
        combined[:,i] = roc[i]
    return combined

def export_csv(data, filename):
    check_dir_exists(OUTPUT_PATH)
    np.savetxt(OUTPUT_PATH+"/"+filename, data, delimiter=",", fmt="%.10g")
    print ("Saved as {}!".format(OUTPUT_PATH+"/"+filename))

def main():
    for each in SESSION_TASKS:
        data = read_labeled_data(INPUT_PATH+"/"+INPUT_FILES[each])
        print "Read {} data!".format(each)
        if each =="dl":
            roc = calculate_roc(data, reverse=True)
        else:
            roc = calculate_roc(data)
        print "Calculated roc for {}!".format(each)
        combined = concat(roc)
        export_csv(combined, OUTPUT_FILES[each])


if __name__ == "__main__":
    main()

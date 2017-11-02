import numpy as np
from preprocess import read_txt
OUTPUT_PATH = "data"
OUTPUT_ABS_FILE = "abs_iso_with_labels.csv"
OUTPUT_REL_FILE = "rel_iso_with_labels.csv"

def check_dir_exists(path):
    """
    Automatically create folders if it doesn't exist.
    Prevent errors during output.
    """
    import os
    path = os.path.join(os.path.dirname(__file__), path)
    if not os.path.exists(path):
        os.makedirs(path)

def calc_abs_iso(px, py):
    """
    Numpy matrix operations to reduce run time.
    """
    # Transverse momentum Pt = sqrt(PX^2+PY^2)
    pt = np.sqrt(np.square(px)+np.square(py))
    pt_sum = np.sum(pt, axis=1)
    print "Calculated absolute isolation!"
    return pt_sum

def calc_rel_iso(px_muon, py_muon, abs_iso):
    """
    Divide absolute isolation by transverse momentum of muon to get
    relative isolation.
    """
    # Transverse momentum Pt = sqrt(PX^2+PY^2)
    pt_muon = np.sqrt(np.square(px_muon)+np.square(py_muon))
    rel_iso = abs_iso / pt_muon
    print "Calculated relative isolation!"
    return rel_iso


def concat(iso):
    len0, len1 = len(iso[0]), len(iso[1])
    shape = (len0+len1, 2)
    combined = np.empty(shape)
    combined[:len0,0] = iso[0]
    combined[:len0,1] = 1 # signal
    combined[len0:,0] = iso[1]
    combined[len0:,1] = 0 # background
    return combined

def export_csv(data, filename):
    check_dir_exists(OUTPUT_PATH)
    np.savetxt(OUTPUT_PATH+"/"+filename, data, delimiter=" ")
    print ("Saved as {}!".format(OUTPUT_PATH+"/"+filename))

def generate_pf_iden(zjets_file=None, qcd_file=None):
    #Source files
    if zjets_file is None:
        zjets_file='data/zjets_iso_mu_20_large.txt'#'ttbar_R_1.0_pt_500_600_m_130_200.txt'##'ttbar_R_1.0_pt_500_600_m_130_200.txt'#'ttbar_no_matching_pt500-600_m150-190_add.txt' # 'ttbar_no_matching_pt500-600_m150-190.txt'
    if qcd_file is None:
        qcd_file='data/qcd_iso_mu_20_large.txt'#'qcd_R_1.0_pt_500_600_m_130_200.txt'#'qcd_R_1.0_pt_500_600_m_130_200.txt'#'dijets_pt_500-600_m150-190_np_add.txt' #  'dijets_pt_500-600_m150-190_np.txt'

    # Column index for Px and Py data
    xcol = 0
    ycol = 1

    # Read data
    files = [zjets_file, qcd_file]
    returns = [read_txt(each) for each in files]
    muon_data = [each[0] for each in returns]
    comp_data = [each[1] for each in returns]

    ##############################
    # Calculate absolute isolation; shape=[1-D array, 1-D array]
    abs_iso = [calc_abs_iso(each[:,:,xcol], each[:,:,ycol]) for each in comp_data]

    # Convert format to single table for plotting purpose
    combined = concat(abs_iso)

    # Export
    export_csv(combined, OUTPUT_ABS_FILE)

    ##############################
    # Calculate relative isolation
    rel_iso = [calc_rel_iso(muon[:,xcol], muon[:,ycol], each_abs)
        for muon, each_abs in zip(muon_data, abs_iso)]

    # Convert format to single table for plotting purpose
    combined = concat(rel_iso)

    # Export
    export_csv(combined, OUTPUT_REL_FILE)


if __name__ == "__main__":
    generate_pf_iden()

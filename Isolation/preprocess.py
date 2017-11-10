import numpy as np

masscuts=[0,9999]#[120,220]#[120,220]#[150,190]#[130,200]#[0,1e5]  #mass cuts

def check_dir_exists(path):
    """
    Automatically create folders if it doesn't exist.
    Prevent errors during output.
    """
    import os
    path = os.path.join(os.path.dirname(__file__), path)
    if not os.path.exists(path):
        os.makedirs(path)

def get_shape(data):
    nset = 0
    maxn = 0
    for row in data:
        if row[0] != " ":
            nset += 1
            temp = int(row)
            if temp > maxn: maxn = temp
    print "Shape = ({}, {})".format(nset, maxn)
    return nset, maxn

def read_txt(filename, shape=None, masscuts=None):
    # Function that reads a txt file and output the
    # leading jet along with its components and mass

    with open(filename, "r") as f:
        data = f.readlines()

    if shape:
        nset, maxn = shape
    else:
        nset, maxn = get_shape(data)

    leading = np.empty((nset, 4))
    components = np.zeros((nset, maxn, 4))
    irow = 0
    iset = 0
    while irow <len(data):
        # Read number of components
        ncomp = int(data[irow])
        # Read leading
        irow += 1
        leading[iset] = np.fromstring(data[irow], sep=" ")
        # Read components
        irow += 1
        strbuffer = data[irow:irow+ncomp]
        string = " ".join(strbuffer)
        temp = np.fromstring(string, sep=" ")
        temp.shape = (ncomp, 4)
        components[iset,:ncomp,:] = temp
        # Next set
        irow += ncomp
        iset += 1

    mass = np.sqrt(leading[:,3]**2-leading[:,0]**2-leading[:,1]**2-leading[:,2]**2)
    if masscuts:
        mask=np.bitwise_and(mass>=masscuts[0],mass<=masscuts[1])
        leading=leading[mask,:]
        components=components[mask,:,:]
        mass = mass[mask]
    #print leading.shape, components.shape, mass.shape
    print "Successfully read {}!".format(filename)
    return leading, components, mass

if __name__ == "__main__":
    #Source files
    top_file='data/zjets_iso_mu_20_large.txt'#'ttbar_R_1.0_pt_500_600_m_130_200.txt'##'ttbar_R_1.0_pt_500_600_m_130_200.txt'#'ttbar_no_matching_pt500-600_m150-190_add.txt' # 'ttbar_no_matching_pt500-600_m150-190.txt'
    dijet_file='data/qcd_iso_mu_20_large.txt'#'qcd_R_1.0_pt_500_600_m_130_200.txt'#'qcd_R_1.0_pt_500_600_m_130_200.txt'#'dijets_pt_500-600_m150-190_np_add.txt' #  'dijets_pt_500-600_m150-190_np.txt'

    read_txt(top_file, shape=(302731, 46))
    read_txt(dijet_file, shape=(16586, 127))

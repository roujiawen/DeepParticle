import matplotlib.pyplot as plt
import csv
import numpy as np

"""
cms=csv.reader(open('CMS.csv','r'))
comb=csv.reader(open('CMS_comb.csv','r'))

eff_cms=[]
mis_cms=[]
eff_comb=[]
mis_comb=[]
    
for row in cms:
    eff_cms.append(row[0])
    mis_cms.append(row[1])
eff_cms=np.array(eff_cms)
mis_cms=np.array(mis_cms)
order=eff_cms.argsort()
eff_cms=eff_cms[order]
mis_cms=mis_cms[order]
    
for row in comb:
    eff_comb.append(row[0])
    mis_comb.append(row[1])
eff_comb=np.array(eff_comb)
mis_comb=np.array(mis_comb)
order=eff_comb.argsort()
eff_comb=eff_comb[order]
mis_comb=mis_comb[order]
"""

def plot(output_valid,valid_data_out,threshold_prob=0):

    res_plot=10000
    tag_eff_vec=np.zeros(res_plot)
    mistag_vec=np.zeros(res_plot)
    treshold_vec=np.linspace(threshold_prob-0.05,1.0,res_plot)
  
    print "output", len(output_valid)
    print output_valid

    print "valid", len(valid_data_out)  
    print valid_data_out 
    
    print "sum data 1", sum(1-valid_data_out.astype(float))
    print "sum data 2", sum(valid_data_out.astype(float))
 
    for j in range(res_plot):
        output_valid_binary=(output_valid>treshold_vec[res_plot-1-j]).astype(float)
        #print output_valid_binary
        tag_eff_vec[j]=np.sum((output_valid_binary*valid_data_out).astype(float))/sum(valid_data_out.astype(float))
        mistag_vec[j]=np.sum(output_valid_binary*(1-valid_data_out))/sum(1-valid_data_out.astype(float))
        
    fig, ax = plt.subplots(figsize=(10,8))
    
    ax.plot(tag_eff_vec,mistag_vec,'b',linewidth=3)
    #ax.plot(eff_cms,mis_cms,'r',linewidth=3)
    #ax.plot(eff_comb,mis_comb,'g',linewidth=3)
    
    ax.legend(['NNs + Boosted trees Tagger','CMS Top Tagger',r'CMS+$\tau_3/\tau_2$+subjet b-tag'],loc='best',prop={'size':18})
    ax.set_yscale('log')
    xmin, xmax = plt.xlim()
    plt.xlim((xmin,0.6))
    
    ax.set_title('Mistag Rate vs Efficiency',fontsize=30)
    [i.set_linewidth(1) for i in ax.spines.itervalues()]
    plt.setp(ax.yaxis.get_ticklines(), 'markersize', 5)
    plt.setp(ax.xaxis.get_ticklines(), 'markersize', 5)
    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_xlabel(r'Top Tag Efficiency', fontsize=30)
    ax.set_ylabel(r'Mistage Rate', fontsize=30);

    fig.tight_layout()
    plt.show()

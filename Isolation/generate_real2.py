import csv
import numpy as np
from math import *
import matplotlib.pyplot as plt
from numpy import linalg as LA

res=20 # resolution of the cell, total size = res^2
treshold = 0.000000001 # treshold for droping cell which are almost never used
masscuts=[0,9999]#[120,220]#[120,220]#[150,190]#[130,200]#[0,1e5]  #mass cuts

def reader_jets(csv_object): # Function that reads the jets file object and output the
                             # leading jet along with its components and mass
    leading=[]
    components=[]
    components_temp=[]
    flag=0
    justone=0
   
    init = True
    for row in csv_object:
        row=np.array(filter(None,row)).astype(float)
        if len(row)==1:
            flag=1
            if not init:
                components_temp=np.array(components_temp)
                components.append(components_temp)            
            components_temp=[]
            init = False 
        elif flag==1:
            leading.append(row)
            flag=0
        elif len(row)==4:
            components_temp.append(row)
    components_temp=np.array(components_temp)
    components.append(components_temp)
 
    leading=np.array(leading)
    components=np.array(components)    

    mass = np.sqrt(leading[0::,3]**2-leading[0::,0]**2-leading[0::,1]**2-leading[0::,2]**2)
    mask=np.bitwise_and(mass>=masscuts[0],mass<=masscuts[1])
    leading=leading[mask,0::]
    components=components[mask]
    mass = mass[mask]
    
    return leading, components, mass


def get_histo(lead, comp, resolution,rotation_type=0):  # function that gets a histogram based on the events
    
    phi, eta, weights_jet = get_angles(lead,comp)
    
    if rotation_type!=0:
        etaprime, phiprime = get_axis(phi, eta, weights_jet,rotation_type)
        oldeta=eta
        oldphi=phi        
        eta = oldphi*phiprime+oldeta*etaprime
        phi = oldphi*etaprime-oldeta*phiprime
    
    H, xedges, yedges = np.histogram2d(phi,eta,resolution,[[-pi/6.0, pi/6.0], [-pi/6.0, pi/6.0]],weights=weights_jet)
    
    return H

def get_angles(lead, comp): # function that gets the angles out of the events
    
    nbparticles=len(comp)
    
    phi=np.zeros((nbparticles))
    eta=np.zeros((nbparticles))
    weights_jet=np.zeros((nbparticles))    
    
    for j in range(nbparticles):
    
        phi[j] = np.arctan2(comp[j,1],comp[j,0]) - np.arctan2(lead[1],lead[0])
        eta[j] = 0.5*np.log((comp[j,3] + comp[j,2])*(lead[3] - lead[2])/((comp[j,3] - comp[j,2])*(lead[3] + lead[2])))
        weights_jet[j]=comp[j,3]
        
    return phi, eta, weights_jet

def get_axis(phi, eta, weights_jet,rotation_type=1): # function that gets the principal axis used for the rotation
    
    if rotation_type==1:
        
        etaprime = np.sum(eta*weights_jet/(np.sqrt(eta**2+phi**2)))
        phiprime = np.sum(phi*weights_jet/(np.sqrt(eta**2+phi**2)))
        
        norm=1.0/np.sqrt(etaprime**2+phiprime**2)
        etaprime=norm*etaprime
        phiprime=norm*phiprime
        
    elif rotation_type==2:
        
        Inertia=np.zeros((2,2))
        Inertia[0,0]=np.sum(eta**2*weights_jet)
        Inertia[1,1]=np.sum(phi**2*weights_jet)
        Inertia[0,1]=np.sum(phi*eta*weights_jet)
        Inertia[0,1]=Inertia[1,0]
        
        w,v = LA.eig(Inertia)
        index_max = np.argmax(np.absolute(w))
        
        etaprime=v[0,index_max]*np.sign(w[index_max])
        phiprime=v[1,index_max]*np.sign(w[index_max])
        
    return etaprime, phiprime

def feature_mask(feature_matrix, tresh = 0.00): # function which returns a vector of output to preserve if they are above the treshold

    average = np.mean(feature_matrix)
    clean_mask = []
    
    for j in range(len(feature_matrix[0,0::])):
        average_j = np.mean(feature_matrix[0::,j])
        if average_j>=tresh*average:
            clean_mask.append(j)
            
    print 'Number of dropped cell:', len(feature_matrix[0,0::])-len(clean_mask)
    return clean_mask    
    

#Source files
top_file='zjets_iso_mu_20_large.txt'#'ttbar_R_1.0_pt_500_600_m_130_200.txt'##'ttbar_R_1.0_pt_500_600_m_130_200.txt'#'ttbar_no_matching_pt500-600_m150-190_add.txt' # 'ttbar_no_matching_pt500-600_m150-190.txt'
dijet_file='qcd_iso_mu_20_large.txt'#'qcd_R_1.0_pt_500_600_m_130_200.txt'#'qcd_R_1.0_pt_500_600_m_130_200.txt'#'dijets_pt_500-600_m150-190_np_add.txt' #  'dijets_pt_500-600_m150-190_np.txt'

top_file_obj=open(top_file,'r')
dijet_file_obj=open(dijet_file,'r')

top_csv=csv.reader(top_file_obj,delimiter=' ')
dijet_csv=csv.reader(dijet_file_obj,delimiter=' ')

leading_top, components_top, mass_top = reader_jets(top_csv)
leading_dijet, components_dijet, mass_dijet = reader_jets(dijet_csv)

print 'number of zjets jets', np.shape(leading_top)[0]
print 'number of qcd muons', np.shape(leading_dijet)[0]

### Get the Histogram and feature vectors
H_top=[]
H_top_feature=[]
for j in range(np.shape(leading_top)[0]):
    H_top.append(get_histo(leading_top[j], components_top[j], res,rotation_type=0))
    H_top_feature.append(np.reshape(H_top[j],(res**2)))
H_top_feature=np.array(H_top_feature)

H_dijet=[]
H_dijet_feature=[]
for j in range(np.shape(leading_dijet)[0]):
    H_dijet.append(get_histo(leading_dijet[j], components_dijet[j], res,rotation_type=0))
    H_dijet_feature.append(np.reshape(H_dijet[j],(res**2)))
H_dijet_feature=np.array(H_dijet_feature)

#Drop values which are almost always zero
#mask_feature=feature_mask(np.vstack((H_top_feature,H_dijet_feature)),treshold)
#H_top_feature=H_top_feature[0::,mask_feature]
#H_dijet_feature=H_dijet_feature[0::,mask_feature]

### Save  the feature vector
np.save('H_zjets_feature_mu_20_res_20_large_test', H_top_feature)
np.save('H_qcd_feature_mu_20_res_20_large_test', H_dijet_feature)

np.save('H_zjets_mu_20_res_20_large_full', mass_top)
np.save('H_qcd_mu_20_res_20_large_full', mass_dijet)

### Plot a random sample of each
fig = plt.figure()
ax = fig.add_subplot(131)
ax.set_title('zjets')
im = plt.imshow(H_top[int(np.random.uniform(0,len(H_top)))], interpolation='nearest', origin='low',
                extent=[-pi/2, pi/2, -pi/2, pi/2])
ax = fig.add_subplot(133)
ax.set_title('qcd')
im = plt.imshow(H_dijet[int(np.random.uniform(0,len(H_dijet)))], interpolation='nearest', origin='low',
                extent=[-pi/2, pi/2, -pi/2, pi/2])
plt.show()

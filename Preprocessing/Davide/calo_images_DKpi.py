
# coding: utf-8

# In[1]:


import root_numpy as rn
import ROOT as r
import numpy as np
import pandas as pd


# In[2]:


import matplotlib.pyplot as plt


# In[7]:


file='DKpi_cells_inout'
file_name='HCAL/'+file+'.root'
tree_name='mytuple/DecayTree'


# In[8]:


f = r.TFile(file_name)
t = f.Get(tree_name)


# In[ ]:


lenT = len([1 for event in t])


# In[6]:


j=295954


# In[7]:


N=j
batch_size=25000
n_batches= N//batch_size


# In[18]:


particle = 'piplus'
cal_zone = 'inner'
variable = 'X'

cellsX_inner_dict={}

for j in range(n_batches):
    cellsX_inner_dict[j]=rn.root2array(
        filenames=file_name,
        treename=tree_name,
        branches=particle+'_L0Calo_HCAL_Cells'+variable+'_'+cal_zone,
        start=j*batch_size,
        stop=(j+1)*batch_size,
    )

    cellsX_inner_dict[j]=np.array([cellsX_inner_dict[j][i] for i in range(batch_size)])
    np.save('npy/'+file+'/'+variable+'_'+cal_zone+'/batch_'+str(j), cellsX_inner_dict[j])

if N % batch_size != 0:

    cellsX_inner_dict[j+1]=rn.root2array(
        filenames=file_name,
        treename=tree_name,
        branches=particle+'_L0Calo_HCAL_Cells'+variable+'_'+cal_zone,
        start=n_batches*batch_size,
        stop=N,
    )
    cellsX_inner_dict[j+1]=np.array([cellsX_inner_dict[j+1][i] for i in range((N % batch_size))])
    np.save('npy/'+file+'/'+variable+'_'+cal_zone+'/batch_'+str(j+1), cellsX_inner_dict[j+1])

variable = 'Y'

cellsY_inner_dict ={}

for j in range(n_batches):
    cellsY_inner_dict[j]=rn.root2array(
        filenames=file_name,
        treename=tree_name,
        branches=particle+'_L0Calo_HCAL_Cells'+variable+'_'+cal_zone,
        start=j*batch_size,
        stop=(j+1)*batch_size,
    )
    cellsY_inner_dict[j]=np.array([cellsY_inner_dict[j][i] for i in range(batch_size)])
    np.save('npy/'+file+'/'+variable+'_'+cal_zone+'/batch_'+str(j), cellsY_inner_dict[j])

if N % batch_size != 0:

    cellsY_inner_dict[j+1]=rn.root2array(
        filenames=file_name,
        treename=tree_name,
        branches=particle+'_L0Calo_HCAL_Cells'+variable+'_'+cal_zone,
        start=n_batches*batch_size,
        stop=N,
    )
    cellsY_inner_dict[j+1]=np.array([cellsY_inner_dict[j+1][i] for i in range((N % batch_size))])
    np.save('npy/'+file+'/'+variable+'_'+cal_zone+'/batch_'+str(j+1), cellsY_inner_dict[j+1])


# In[ ]:


particle = 'piplus'
cal_zone = 'outer'
variable = 'X'

cellsX_outer_dict={}

for j in range(n_batches):
    cellsX_outer_dict[j]=rn.root2array(
        filenames=file_name,
        treename=tree_name,
        branches=particle+'_L0Calo_HCAL_Cells'+variable+'_'+cal_zone,
        start=j*batch_size,
        stop=(j+1)*batch_size,
    )

    cellsX_outer_dict[j]=np.array([cellsX_outer_dict[j][i] for i in range(batch_size)])
    np.save('npy/'+file+'/'+variable+'_'+cal_zone+'/batch_'+str(j), cellsX_outer_dict[j])

if N % batch_size != 0:

    cellsX_outer_dict[j+1]=rn.root2array(
        filenames=file_name,
        treename=tree_name,
        branches=particle+'_L0Calo_HCAL_Cells'+variable+'_'+cal_zone,
        start=n_batches*batch_size,
        stop=N,
    )
    cellsX_outer_dict[j+1]=np.array([cellsX_outer_dict[j+1][i] for i in range((N % batch_size))])
    np.save('npy/'+file+'/'+variable+'_'+cal_zone+'/batch_'+str(j+1), cellsX_outer_dict[j+1])

variable = 'Y'

cellsY_outer_dict ={}

for j in range(n_batches):
    cellsY_outer_dict[j]=rn.root2array(
        filenames=file_name,
        treename=tree_name,
        branches=particle+'_L0Calo_HCAL_Cells'+variable+'_'+cal_zone,
        start=j*batch_size,
        stop=(j+1)*batch_size,
    )
    cellsY_outer_dict[j]=np.array([cellsY_outer_dict[j][i] for i in range(batch_size)])
    np.save('npy/'+file+'/'+variable+'_'+cal_zone+'/batch_'+str(j), cellsY_outer_dict[j])

if N % batch_size != 0:

    cellsY_outer_dict[j+1]=rn.root2array(
        filenames=file_name,
        treename=tree_name,
        branches=particle+'_L0Calo_HCAL_Cells'+variable+'_'+cal_zone,
        start=n_batches*batch_size,
        stop=N,
    )
    cellsY_outer_dict[j+1]=np.array([cellsY_outer_dict[j+1][i] for i in range((N % batch_size))])
    np.save('npy/'+file+'/'+variable+'_'+cal_zone+'/batch_'+str(j+1), cellsY_outer_dict[j+1])


# In[8]:


particle = 'piplus'
variable = 'ET'

cal_zone = 'inner'

cellsET_inner_dict={}

for j in range(n_batches):
    cellsET_inner_dict[j]=rn.root2array(
        filenames=file_name,
        treename=tree_name,
        branches=particle+'_L0Calo_HCAL_Cells'+variable+'_'+cal_zone,
        start=j*batch_size,
        stop=(j+1)*batch_size,
    )

    cellsET_inner_dict[j]=np.array([cellsET_inner_dict[j][i] for i in range(batch_size)])
    np.save('npy/'+file+'/'+particle+'/'+variable+'_'+cal_zone+'/batch_'+str(j), cellsET_inner_dict[j])

if N % batch_size != 0:

    cellsET_inner_dict[j+1]=rn.root2array(
        filenames=file_name,
        treename=tree_name,
        branches=particle+'_L0Calo_HCAL_Cells'+variable+'_'+cal_zone,
        start=n_batches*batch_size,
        stop=N,
    )
    cellsET_inner_dict[j+1]=np.array([cellsET_inner_dict[j+1][i] for i in range((N % batch_size))])
    np.save('npy/'+file+'/'+particle+'/'+variable+'_'+cal_zone+'/batch_'+str(j+1), cellsET_inner_dict[j+1])



# In[9]:


particle = 'piplus'
variable = 'ET'

cal_zone='outer'

cellsET_outer_dict ={}

for j in range(n_batches):
    cellsET_outer_dict[j]=rn.root2array(
        filenames=file_name,
        treename=tree_name,
        branches=particle+'_L0Calo_HCAL_Cells'+variable+'_'+cal_zone,
        start=j*batch_size,
        stop=(j+1)*batch_size,
    )
    cellsET_outer_dict[j]=np.array([cellsET_outer_dict[j][i] for i in range(batch_size)])
    np.save('npy/'+file+'/'+particle+'/'+variable+'_'+cal_zone+'/batch_'+str(j),  cellsET_outer_dict[j])

if N % batch_size != 0:

    cellsET_outer_dict[j+1]=rn.root2array(
        filenames=file_name,
        treename=tree_name,
        branches=particle+'_L0Calo_HCAL_Cells'+variable+'_'+cal_zone,
        start=n_batches*batch_size,
        stop=N,
    )
    cellsET_outer_dict[j+1]=np.array([cellsET_outer_dict[j+1][i] for i in range((N % batch_size))])
    np.save('npy/'+file+'/'+particle+'/'+variable+'_'+cal_zone+'/batch_'+str(j+1),  cellsET_outer_dict[j+1])


# In[10]:


particle = 'Kminus'
variable = 'ET'

cal_zone = 'inner'

cellsET_inner_dict={}

for j in range(n_batches):
    cellsET_inner_dict[j]=rn.root2array(
        filenames=file_name,
        treename=tree_name,
        branches=particle+'_L0Calo_HCAL_Cells'+variable+'_'+cal_zone,
        start=j*batch_size,
        stop=(j+1)*batch_size,
    )

    cellsET_inner_dict[j]=np.array([cellsET_inner_dict[j][i] for i in range(batch_size)])
    np.save('npy/'+file+'/'+particle+'/'+variable+'_'+cal_zone+'/batch_'+str(j), cellsET_inner_dict[j])

if N % batch_size != 0:

    cellsET_inner_dict[j+1]=rn.root2array(
        filenames=file_name,
        treename=tree_name,
        branches=particle+'_L0Calo_HCAL_Cells'+variable+'_'+cal_zone,
        start=n_batches*batch_size,
        stop=N,
    )
    cellsET_inner_dict[j+1]=np.array([cellsET_inner_dict[j+1][i] for i in range((N % batch_size))])
    np.save('npy/'+file+'/'+particle+'/'+variable+'_'+cal_zone+'/batch_'+str(j+1), cellsET_inner_dict[j+1])



# In[11]:


particle = 'Kminus'
variable = 'ET'

cal_zone='outer'

cellsET_outer_dict ={}

for j in range(n_batches):
    cellsET_outer_dict[j]=rn.root2array(
        filenames=file_name,
        treename=tree_name,
        branches=particle+'_L0Calo_HCAL_Cells'+variable+'_'+cal_zone,
        start=j*batch_size,
        stop=(j+1)*batch_size,
    )
    cellsET_outer_dict[j]=np.array([cellsET_outer_dict[j][i] for i in range(batch_size)])
    np.save('npy/'+file+'/'+particle+'/'+variable+'_'+cal_zone+'/batch_'+str(j),  cellsET_outer_dict[j])

if N % batch_size != 0:

    cellsET_outer_dict[j+1]=rn.root2array(
        filenames=file_name,
        treename=tree_name,
        branches=particle+'_L0Calo_HCAL_Cells'+variable+'_'+cal_zone,
        start=n_batches*batch_size,
        stop=N,
    )
    cellsET_outer_dict[j+1]=np.array([cellsET_outer_dict[j+1][i] for i in range((N % batch_size))])
    np.save('npy/'+file+'/'+particle+'/'+variable+'_'+cal_zone+'/batch_'+str(j+1),  cellsET_outer_dict[j+1])


# In[30]:


batch=0
n=100


# In[31]:


plt.imshow(cellsET_inner_dict[batch][n],cmap='gray')


# In[32]:


plt.imshow(cellsET_outer_dict[batch][n],cmap='gray')


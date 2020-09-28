
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import ROOT as r


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


f = r.TFile('HCAL/piplus_cells_inout.root')


# In[4]:


t = f.Get("mytuple/DecayTree")


# In[5]:


r.TBrowser()


# In[6]:


t.SetBranchStatus("*",0)


# In[7]:


t.SetBranchStatus("piplus_L0Calo_HCAL_xProjection",1)
t.SetBranchStatus("piplus_L0Calo_HCAL_yProjection",1)
t.SetBranchStatus("piplus_L0Calo_HCAL_realET",1)
t.SetBranchStatus("piplus_L0Calo_HCAL_region", 1)


# In[8]:


#Build the MCtruth image


# In[9]:


x_projections=[]
y_projections=[]
real_ET=[]
region=[]


# In[10]:


for k, event in enumerate(t):
    
        region.append(t.piplus_L0Calo_HCAL_region)
        x_projections.append(t.piplus_L0Calo_HCAL_xProjection)
        y_projections.append(t.piplus_L0Calo_HCAL_yProjection)
        real_ET.append(t.piplus_L0Calo_HCAL_realET)


# In[11]:


true_events ={'true_x':np.array(x_projections),'true_y':np.array(y_projections),'true_ET':np.array(real_ET),'region':np.array(region)}


# In[12]:


df = pd.DataFrame.from_dict(true_events)
df.to_csv('csv/piplus_cells_inout/MCtrue_piplus.csv', index=False)


# In[13]:


true_events['region'].shape


# In[14]:


true_events['region'][np.where(true_events['region']<0)].shape


# In[15]:


true_events['true_x'][0],true_events['true_y'][0],true_events['region'][0]


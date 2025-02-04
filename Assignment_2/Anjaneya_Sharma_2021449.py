#!/usr/bin/env python
# coding: utf-8

# In[82]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.io import loadmat
import seaborn as sns
from scipy.integrate import trapz


# ***Part 2 A***

# In[ ]:


# *****************************************************************************************************


# In[106]:


data = scipy.io.loadmat('Assignment2_2A_NDM_2024.mat')['NDM_Assignment2']
n_participants, n_conditions = data.shape ######## check

#setting up a dictionary for reaction times in the beginning
rtimeDict = {0: [], 1: []}     


# In[107]:


# **********************************************************************************************


# In[108]:


for condition in range(2):
    for participant in range(n_participants):
        participant_data = data[participant, condition]
        trial_rts = []       
        
        for trial in participant_data:
            threshold_indices = np.where(trial >= 600)[0]
            if len(threshold_indices) > 300:
                rt = threshold_indices[0]  #first time threshold crssoing
            else:
                rt = np.nan  # If threshold is never reached, which doesn't happen here
                #added this just in case
            trial_rts.append(rt)
        
        rtimeDict[condition].append(trial_rts)

mean_rt_cond1 = np.nanmean(rtimeDict[0], axis=1)
mean_rt_cond2 = np.nanmean(rtimeDict[1], axis=1)


# In[109]:


#i have checked the normality of data and with the help of p values(using shapiro wilk test), determining which test to perform
_, p_val_1 = stats.shapiro(mean_rt_cond1)
_, p_val_2 = stats.shapiro(mean_rt_cond2)


if p_val_1 > 0.05 and p_val_2 > 0.05:
    statistic, p_value = stats.ttest_ind(mean_rt_cond1, mean_rt_cond2, nan_policy='omit')
    test_name = "Independent t-test"
else:
    statistic, p_value = stats.mannwhitneyu(mean_rt_cond1, mean_rt_cond2)
    test_name = "Mann-Whitney U test"




# In[110]:


# ***************************************************************************


# In[111]:


# Print statistical results      
print("\nResults and metrics :")
print(f"{test_name} test was used")
print(f"Test statistic: {statistic}")
print(f"p-value: {p_value}")
print(f"\nCondition 1 (High Coherence): Mean RT = {np.nanmean(mean_rt_cond1):.1f} ms")
print(f"\nCondition 2 (Low Coherence): Mean RT = {np.nanmean(mean_rt_cond2):.1f} ms")


# In[112]:


# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), facecolor='white')
bins = np.linspace(0, 1000, 21)
condition_names = ['High Coherence', 'Low Coherence']

# Basic plot for each condition
for condition, (ax, name) in enumerate(zip([ax1, ax2], condition_names)):
    participant_means = np.nanmean(rtimeDict[condition], axis=1)
    mean_rt = np.nanmean(participant_means)
    
    ax.hist(participant_means, bins=bins, color='green', edgecolor='black')
    ax.axvline(mean_rt, color='red', linestyle='--', linewidth=1.5, label=f'Mean RT: {mean_rt:.1f} ms')
    
    ax.set_xlabel('Reaction Time (ms)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'{name} (Mean RT = {mean_rt:.1f} ms)', fontsize=14)
    ax.legend(fontsize=10)

plt.suptitle('Distribution of reaction times in Random Dot Motion Task', fontsize=20, color='blue')


plt.show()


# In[113]:


# **********************************************************************************************


# In[97]:


mean_rt_cond1


# In[98]:


mean_rt_cond2


# ***Part 2 B***

# In[83]:


data_xl = pd.read_excel('Assignment2-2B-NDM-2024.xlsx', index_col = False)

file_path = 'Assignment2-2B-NDM-2024.xlsx'
xls = pd.ExcelFile(file_path)

auc_dict = {}


# In[87]:


for cnt, sheet_name in enumerate(xls.sheet_names, start=1):
    #skipping the first row containing HA and FT
    df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=1)
    
    condition_aucs = []
    
    # i have seperated the HT and FA columns from the dataframe to make it simpler for me to 
    # perform the test

    if cnt == 1: # i have also used cnt to correct the first sheet column offset problem

        ht_cols = df.iloc[:, 1::3]
        fa_cols = df.iloc[:, 2::3]
    else:
        
        ht_cols = df.iloc[:, ::3]
        fa_cols = df.iloc[:, 1::3]
    
    # converting columns from object data typee to float64
    ht_cols = ht_cols.apply(pd.to_numeric, errors='coerce')
    fa_cols = fa_cols.apply(pd.to_numeric, errors='coerce')
    

    fa_cols.columns = ht_cols.columns    #in order to ensure no problems occur while performing operations on them
    

    for participant in range(ht_cols.shape[1]):
        ht = ht_cols.iloc[:, participant].values
        fa = fa_cols.iloc[:, participant].values


        sorted_indices = np.argsort(fa)
        fa_sorted = fa[sorted_indices]
        ht_sorted = ht[sorted_indices]

        auc_value = trapz(ht_sorted, fa_sorted)
        condition_aucs.append(auc_value)

    auc_dict[sheet_name] = condition_aucs  

auc_df = pd.DataFrame(auc_dict, columns=['Cond1', 'Cond2', 'Cond3', 'Cond4'])


    


# In[88]:


print(auc_df)
for i in range(4):
    print(f"Mean AUC for Cond{i+1}: {auc_df.iloc[:, i].mean()}")


# In[89]:


plt.figure(figsize=(12, 6))
auc_df.boxplot()
plt.title('Box and Whisker Plot of AUC Values for Each Experimental Condition')
plt.xlabel('Experimental Conditions')
plt.ylabel('AUC (Accuracy)')
plt.grid(False)
plt.show()

friedman_stat, p_value = stats.friedmanchisquare(
    auc_df['Cond1'], auc_df['Cond2'], auc_df['Cond3'], auc_df['Cond4']
)

print(f"Friedman's Chi-square statistic: {friedman_stat}")
print(f"p-value: {p_value}")


# In[ ]:





# In[ ]:





# In[ ]:





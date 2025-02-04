#!/usr/bin/env python
# coding: utf-8

# In[64]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import shapiro, mannwhitneyu, wilcoxon, ttest_ind, ttest_rel
from math import sqrt


# ## Reading the data from the xlsx files

# In[65]:


choice = pd.read_excel('choice.xlsx', sheet_name=None, header = None)
win = pd.read_excel('win.xlsx', sheet_name=None, header = None)
loss = pd.read_excel('loss.xlsx', sheet_name=None, header = None)


# In[66]:


len(choice['group2'])


# # Question 2 (A)

# In[66]:





# In[67]:


# Function to calculate proportion of switches after a gain/loss trial
def calculate_switch_proportions(choice, win, loss):
    participants = choice.shape[0]
    trials = choice.shape[1]
    switch_after_gain = np.zeros(participants)
    switch_after_loss = np.zeros(participants)

    for i in range(participants):
        switches_gain = 0
        switches_loss = 0
        gain_trials = 0
        loss_trials = 0

        for t in range(trials - 1):
            if win.iloc[i, t] > abs(loss.iloc[i, t]):  # Gain trial (since loss values are <= 0)
                gain_trials += 1
                if choice.iloc[i, t] != choice.iloc[i, t + 1]:
                    switches_gain += 1
            else:  # Loss trial
                loss_trials += 1
                if choice.iloc[i, t] != choice.iloc[i, t + 1]:
                    switches_loss += 1

        switch_after_gain[i] = switches_gain / (gain_trials)
        switch_after_loss[i] = switches_loss / (loss_trials)

    return switch_after_gain, switch_after_loss


# In[68]:


switch_gain_group1, switch_loss_group1 = calculate_switch_proportions(choice['group1'], win['group1'], loss['group1'])
switch_gain_group2, switch_loss_group2 = calculate_switch_proportions(choice['group2'], win['group2'], loss['group2'])


# In[69]:


mean_gain_group1 = np.mean(switch_gain_group1)
mean_loss_group1 = np.mean(switch_loss_group1)
mean_gain_group2 = np.mean(switch_gain_group2)
mean_loss_group2 = np.mean(switch_loss_group2)


# In[70]:


sem_gain_group1 = stats.sem(switch_gain_group1)
sem_loss_group1 = stats.sem(switch_loss_group1)
sem_gain_group2 = stats.sem(switch_gain_group2)
sem_loss_group2 = stats.sem(switch_loss_group2)


# In[71]:


# Plotting the bar plots with error bars
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
# Group 1 plot

axes[0].bar(['Gain', 'Loss'], [mean_gain_group1, mean_loss_group1], yerr=[sem_gain_group1, sem_loss_group1], capsize=5, color=['blue', 'red'])
axes[0].set_title('Group 1')
axes[0].set_xlabel('Trial Type')
axes[0].set_ylabel('Mean Proportion of Switched Responses')


# Group 2 plot
axes[1].bar(['Gain', 'Loss'], [mean_gain_group2, mean_loss_group2], yerr=[sem_gain_group2, sem_loss_group2], capsize=5, color=['blue', 'red'])
axes[1].set_title('Group 2')
axes[1].set_xlabel('Trial Type')


plt.suptitle('Mean Proportion of Switched Responses After Gain and Loss Trials')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# ### Checking additional stats and see if there is a need for correction

# In[72]:


# Diagnostic check for variability and outliers
def diagnostic_check(data, group_name):
    print(f"Diagnostic check for {group_name}:")
    print(f"Mean: {np.mean(data):.3f}, Std Dev: {np.std(data):.3f}, Min: {np.min(data):.3f}, Max: {np.max(data):.3f}")
    print(f"Number of unique values: {len(np.unique(data))}")
    if len(np.unique(data)) < 5:
        print("Problem somewhere, variablity low")
    else:
        print("No problem detected")


# In[73]:


# Run diagnostic checks
for group_data, group_name in zip([switch_gain_group1, switch_loss_group1, switch_gain_group2, switch_loss_group2],
                                  ["Group 1 Gain", "Group 1 Loss", "Group 2 Gain", "Group 2 Loss"]):
    diagnostic_check(group_data, group_name)


# In[73]:





# ### Running statistical tests and finding out the test metrics and p values

# In[74]:


# Statistical tests
def perform_stat_tests(switch_gain_group1, switch_loss_group1, switch_gain_group2, switch_loss_group2):
    # Normality check using Shapiro-Wilk test
    p_gain_group1 = shapiro(switch_gain_group1).pvalue
    p_loss_group1 = shapiro(switch_loss_group1).pvalue
    p_gain_group2 = shapiro(switch_gain_group2).pvalue
    p_loss_group2 = shapiro(switch_loss_group2).pvalue

    print("Shapiro-Wilk test for normality:")
    print(f"Group 1 Gain trials: p-value = {p_gain_group1}")
    print(f"Group 1 Loss trials: p-value = {p_loss_group1}")
    print(f"Group 2 Gain trials: p-value = {p_gain_group2}")
    print(f"Group 2 Loss trials: p-value = {p_loss_group2}\n")

    # Determine whether to use parametric or non-parametric tests based on normality results
    def use_parametric_test(p_values):

        return all(p > 0.05 for p in p_values)

    # Between groups comparison
    if use_parametric_test([p_gain_group1, p_gain_group2]):
        # Parametric test (t-test) for gain trials between groups
        t_gain, p_gain = ttest_ind(switch_gain_group1, switch_gain_group2)
        print(f"Gain trials between groups (t-test): t-statistic = {t_gain}, p-value = {p_gain}")
    else:
        # Non-parametric test (Mann-Whitney U) for gain trials between groups
        u_gain, p_gain = mannwhitneyu(switch_gain_group1, switch_gain_group2, alternative='two-sided')
        print(f"Gain trials between groups (Mann-Whitney U): U-statistic = {u_gain}, p-value = {p_gain}")

    if use_parametric_test([p_loss_group1, p_loss_group2]):
        # Parametric test (t-test) for loss trials between groups
        t_loss, p_loss = ttest_ind(switch_loss_group1, switch_loss_group2)
        print(f"Loss trials between groups (t-test): t-statistic = {t_loss}, p-value = {p_loss}\n")
    else:
        # Non-parametric test (Mann-Whitney U) for loss trials between groups
        u_loss, p_loss = mannwhitneyu(switch_loss_group1, switch_loss_group2, alternative='two-sided')
        print(f"Loss trials between groups (Mann-Whitney U): U-statistic = {u_loss}, p-value = {p_loss}\n")

    # Within group comparisons (Gain vs Loss)
    if use_parametric_test([p_gain_group1, p_loss_group1]):
        # Parametric test (paired t-test) for Group 1
        t_group1, p_group1 = ttest_rel(switch_gain_group1, switch_loss_group1)
        print(f"Group 1 (Gain vs Loss, t-test): t-statistic = {t_group1}, p-value = {p_group1}")
    else:
        # Non-parametric test (Wilcoxon) for Group 1
        w_group1, p_group1 = wilcoxon(switch_gain_group1, switch_loss_group1)
        print(f"Group 1 (Gain vs Loss, Wilcoxon): W-statistic = {w_group1}, p-value = {p_group1}")

    if use_parametric_test([p_gain_group2, p_loss_group2]):
        # Parametric test (paired t-test) for Group 2
        t_group2, p_group2 = ttest_rel(switch_gain_group2, switch_loss_group2)
        print(f"Group 2 (Gain vs Loss, t-test): t-statistic = {t_group2}, p-value = {p_group2}")
    else:
        # Non-parametric test (Wilcoxon) for Group 2
        w_group2, p_group2 = wilcoxon(switch_gain_group2, switch_loss_group2)
        print(f"Group 2 (Gain vs Loss, Wilcoxon): W-statistic = {w_group2}, p-value = {p_group2}")

# Perform statistical tests
perform_stat_tests(switch_gain_group1, switch_loss_group1, switch_gain_group2, switch_loss_group2)


# In[74]:





# In[75]:


# Function to calculate deck choice proportions before and after loss trials
def calculate_deck_proportions(choice, loss,win):
    participants = choice.shape[0]
    trials = choice.shape[1]
    deck_choices_before_loss = {1: 0, 2: 0, 3: 0, 4: 0}
    deck_choices_after_loss = {1: 0, 2: 0, 3: 0, 4: 0}

    for i in range(participants):
        for t in range(1, trials):
            if win.iloc[i,t-1] + loss.iloc[i, t - 1] < 0 and choice.iloc[i,t] != choice.iloc[i,t-1]:  # Loss trial occurred
                deck_choices_after_loss[choice.iloc[i, t]] += 1
                deck_choices_before_loss[choice.iloc[i, t - 1]] += 1

                # print("participant: ",i+1)
                # print("trial :", t)
                # print("choice after loss : ", choice.iloc[i,t])
                # print("choice before loss : ", choice.iloc[i,t-1])


    total_before = sum(deck_choices_before_loss.values())
    total_after = sum(deck_choices_after_loss.values())

    proportions_before = {deck: count / total_before for deck, count in deck_choices_before_loss.items()}
    proportions_after = {deck: count / total_after for deck, count in deck_choices_after_loss.items()}

    return proportions_before, proportions_after

# Calculate deck choice proportions for both groups

proportions_before_group1, proportions_after_group1 = calculate_deck_proportions(choice['group1'], loss['group1'], win['group2'])
proportions_before_group2, proportions_after_group2 = calculate_deck_proportions(choice['group2'], loss['group2'], win['group2'])


# In[76]:


# Print the results in a formatted way
def print_deck_analysis(proportions_before, proportions_after, group_name):
    print(f"{group_name} deck rankings (before loss):")
    for rank, (deck, proportion) in enumerate(sorted(proportions_before.items(), key=lambda x: x[1], reverse=True), 1):
        print(f"Rank {rank}: Deck {deck} (proportion: {round(proportion,3)})")
    print()
    print(f"{group_name} deck rankings (after loss):")
    for rank, (deck, proportion) in enumerate(sorted(proportions_after.items(), key=lambda x: x[1], reverse=True), 1):
        print(f"Rank {rank}: Deck {deck} (proportion: {round(proportion,3)})")
    print()

# Print deck choice analysis for both groups
print("Analysis of deck choices BEFORE and AFTER loss trials:")
print_deck_analysis(proportions_before_group1, proportions_after_group1, "Group 1")
print_deck_analysis(proportions_before_group2, proportions_after_group2, "Group 2")


# In[76]:





# In[76]:





# In[77]:


# Function to determine the deck chosen before switching after a loss trial
def calculate_deck_before_switch(choice, loss, win):
    participants = choice.shape[0]
    trials = choice.shape[1]
    deck_before_switch = {1: 0, 2: 0, 3: 0, 4: 0}
    total_switches = 0

    for i in range(participants):
        for t in range(trials - 1):
            if win.iloc[i,t] + loss.iloc[i, t] < 0 and choice.iloc[i, t] != choice.iloc[i, t + 1]:  # Loss trial and switch occurred
                deck_before_switch[choice.iloc[i, t]] += 1
                total_switches += 1

    proportions_before_switch = {deck: count / total_switches for deck, count in deck_before_switch.items()}
    return proportions_before_switch

# Calculate deck choice proportions before switching after loss trials for both groups
proportions_before_switch_group1 = calculate_deck_before_switch(choice['group1'], loss['group1'], win['group1'])
proportions_before_switch_group2 = calculate_deck_before_switch(choice['group2'], loss['group2'], win['group2'])


# In[78]:


# Plotting the bar plots for deck choice before switching after loss trials
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Group 1 plot
axes[0].bar(['Deck 1', 'Deck 2', 'Deck 3', 'Deck 4'], list(proportions_before_switch_group1.values()), capsize=5, color=['blue', 'green', 'orange', 'purple'])
axes[0].set_title('Group 1')
axes[0].set_xlabel('Deck')
axes[0].set_ylabel('Mean Proportion Before Switching After Loss')

# Group 2 plot
axes[1].bar(['Deck 1', 'Deck 2', 'Deck 3', 'Deck 4'], list(proportions_before_switch_group2.values()), capsize=5, color=['blue', 'green', 'orange', 'purple'])
axes[1].set_title('Group 2')
axes[1].set_xlabel('Deck')

plt.suptitle('Mean Proportion of Deck Choice Before Switching After Loss Trials')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Ranking decks in decreasing order based on their mean proportions for each group
ranking_group1 = sorted(zip(['Deck 1', 'Deck 2', 'Deck 3', 'Deck 4'], proportions_before_switch_group1.values()), key=lambda x: x[1], reverse=True)
ranking_group2 = sorted(zip(['Deck 1', 'Deck 2', 'Deck 3', 'Deck 4'], proportions_before_switch_group2.values()), key=lambda x: x[1], reverse=True)


# In[79]:


print("Ranking of decks for Group 1 (based on mean proportions before switching after loss):")
for rank, (deck, proportion) in enumerate(ranking_group1, start=1):
    print(f"Rank {rank}: {deck} (Proportion: {proportion:.3f})")

print("\nRanking of decks for Group 2 (based on mean proportions before switching after loss):")
for rank, (deck, proportion) in enumerate(ranking_group2, start=1):
    print(f"Rank {rank}: {deck} (Proportion: {proportion:.3f})")


# In[80]:


# Function to determine the deck chosen after switching after a loss trial
def calculate_deck_after_switch(choice, loss, win):
    participants = choice.shape[0]
    trials = choice.shape[1]
    deck_after_switch = {1: 0, 2: 0, 3: 0, 4: 0}
    total_switches = 0

    for i in range(participants):
        for t in range(trials - 1):
            if win.iloc[i,t] + loss.iloc[i, t] < 0 and choice.iloc[i, t] != choice.iloc[i, t + 1]:  # Loss trial and switch occurred
                deck_after_switch[choice.iloc[i, t + 1]] += 1
                total_switches += 1

    proportions_after_switch = {deck: count / total_switches for deck, count in deck_after_switch.items()}
    return proportions_after_switch

# Calculate deck choice proportions after switching after loss trials for both groups
proportions_after_switch_group1 = calculate_deck_after_switch(choice['group1'], loss['group1'], win['group1'])
proportions_after_switch_group2 = calculate_deck_after_switch(choice['group2'], loss['group2'], win['group1'])


# In[81]:


# Plotting the bar plots for deck choice after switching after loss trials
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Group 1 plot
axes[0].bar(['Deck 1', 'Deck 2', 'Deck 3', 'Deck 4'], list(proportions_after_switch_group1.values()), capsize=5, color=['blue', 'green', 'orange', 'purple'])
axes[0].set_title('Group 1')
axes[0].set_xlabel('Deck')
axes[0].set_ylabel('Mean Proportion After Switching After Loss')

# Group 2 plot
axes[1].bar(['Deck 1', 'Deck 2', 'Deck 3', 'Deck 4'], list(proportions_after_switch_group2.values()), capsize=5, color=['blue', 'green', 'orange', 'purple'])
axes[1].set_title('Group 2')
axes[1].set_xlabel('Deck')

plt.suptitle('Mean Proportion of Deck Choice After Switching After Loss Trials')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()



# In[82]:


# Ranking decks in decreasing order based on their mean proportions for each group
ranking_after_switch_group1 = sorted(zip(['Deck 1', 'Deck 2', 'Deck 3', 'Deck 4'], proportions_after_switch_group1.values()), key=lambda x: x[1], reverse=True)
ranking_after_switch_group2 = sorted(zip(['Deck 1', 'Deck 2', 'Deck 3', 'Deck 4'], proportions_after_switch_group2.values()), key=lambda x: x[1], reverse=True)

print("Ranking of decks for Group 1 (based on mean proportions after switching after loss):")
for rank, (deck, proportion) in enumerate(ranking_after_switch_group1, start=1):
    print(f"Rank {rank}: {deck} (Proportion: {proportion:.3f})")

print("\nRanking of decks for Group 2 (based on mean proportions after switching after loss):")
for rank, (deck, proportion) in enumerate(ranking_after_switch_group2, start=1):
    print(f"Rank {rank}: {deck} (Proportion: {proportion:.3f})")


# In[82]:





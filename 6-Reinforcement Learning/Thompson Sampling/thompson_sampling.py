# Thompson Sampling

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB
import random
N = 10000
d = 10
ads_selected = []
Number_of_rewards_ones = [0] * d
Number_of_rewards_zeroes = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_data = random.betavariate(Number_of_rewards_ones[i] +1 , Number_of_rewards_zeroes[i] +1 )
        if random_data > max_random:
            max_random = random_data
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        Number_of_rewards_ones[ad] =  Number_of_rewards_ones[ad] + 1
    else:
        Number_of_rewards_zeroes[ad] = Number_of_rewards_zeroes[ad] + 1
    total_reward = total_reward + reward

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

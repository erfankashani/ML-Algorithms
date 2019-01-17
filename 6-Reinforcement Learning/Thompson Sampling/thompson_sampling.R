# Thompson Sampling

#import the dataset
dataset = read.csv('Ads_CTR_optimisation.csv')

#implement the UCB algorithem
N = 10000
d = 10
ads_selected = integer(0)
Number_of_rewards_1 = integer(d)
Number_of_rewards_0 = integer(d)
total_reward = 0
for (n in 1:N){
  Ad = 0
  max_reward = 0
  for (i in 1:d){
    random_data = rbeta(n = 1,
                        shape1 = Number_of_rewards_1[i] +1 ,
                        shape2 = Number_of_rewards_0[i] +1)
    if(random_data > max_reward){
      max_reward = random_data
      Ad = i
    }
  }
  ads_selected = append(ads_selected, Ad)
  reward = dataset[n , Ad]
  if (reward ==1){
    Number_of_rewards_0[Ad] = Number_of_rewards_0[Ad] +1
  }else {
    Number_of_rewards_1[Ad] = Number_of_rewards_1[Ad] +1
  }
  total_reward = total_reward + reward
}


#visualise the information
hist(ads_selected,
     col = 'red',
     main = 'Histogram of ads selections',
     xlab = 'Ads',
     ylab = 'Number of times each ad was selected')

# Upper Confidence Bound

#import the dataset
dataset = read.csv('Ads_CTR_optimisation.csv')

#implement the UCB algorithem
N = 10000
d = 10
ads_selected = integer(0)
Number_of_selections = integer(d)
sum_of_rewards = integer(d)
total_reward = 0
for (n in 1:N){
  Ad = 0
  max_upper_bound = 0
  for (i in 1:d){

    if(Number_of_selections[i] > 0){
      average_reward =(sum_of_rewards[i]/Number_of_selections[i])
      delta_i = sqrt(3/2 * log(n) / Number_of_selections[i])
      Upper_bound = average_reward + delta_i
    } else {
      Upper_bound = 1e400

    }
    if(Upper_bound > max_upper_bound){
      max_upper_bound = Upper_bound
      Ad = i
    }
  }
  ads_selected = append(ads_selected, Ad)
  Number_of_selections[Ad] = Number_of_selections[Ad] + 1
  reward = dataset[n , Ad]
  sum_of_rewards[Ad] = sum_of_rewards[Ad] + reward
  total_reward = total_reward + reward
}


#visualise the information
hist(ads_selected,
     col = 'red',
     main = 'Histogram of ads selections',
     xlab = 'Ads',
     ylab = 'Number of times each ad was selected')

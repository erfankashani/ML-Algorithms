# Natural Language Processing

#import the dataset
dataset_original = read.delim('Restaurant_Reviews.tsv' , quote = '', stringsAsFactors = FALSE )

#cleaning the text
#install.packages('tm')
#install.packages('SnowballC')
library(tm)
library(snowballC)


corpus = VCorpus(VectorSource(dataset_original$Review))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords())
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)

#create the bag of words // spark matrix
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999)
dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_original$Liked

#import the classification algorithem


#Encoding the target feature as factor
dataset$Liked = factor(dataset$Liked, levels = c(0,1))



# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked , SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting the Random Forest tree Classification Model to the dataset and prediction
#install.packages('randomForest')
library(randomForest)

classifier = randomForest(x = training_set[-692],
                          y = training_set$Liked ,
                          ntree = 10)


#preducting on testing set /// here we had to type the type also becasue the withou it we would only have a series of probabilities
y_pred = predict(classifier, newdata =test_set[-692], type ='class')


#making confusion matrix
cm = table(test_set[, 692], y_pred)

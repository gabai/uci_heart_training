# Documentation not clear about reprocessed outcome variable. From 0-1 on initial data description to 0-4 on reprocessed data.
# Still not successful 

library(keras)

#Get data
#Hungarian Data
data <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data"),
                  na.strings="?", header=F)
#Cleveland Data
#data <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"),
#                 na.strings="?", header=F)

data<- data[, 1:10, 14]
data<- data[complete.cases(data), ]
print(data[,10])

#Convert data to matrix
data <- as.matrix(data)
str(data)
#dimnames(data) <- NULL


data[, 1:9] <- normalize(data[, 1:9])
data[, 10] <- as.numeric(data[, 10])
summary(data)

#Set random seed
set.seed(7)

#Split data 70/30 for training/testing
ind <- sample(2, nrow(data), replace = T, prob = c(0.7, 0.3))
training <- data[ind==1, 1:9]
test <- data[ind==2, 1:9]
trainingtarget <- data[ind==1,10]
testtarget <- data[ind==2, 10]

#Outcome Training
print(trainingtarget)
print(testtarget)

#Make numeric as categorical matrix for model
trainlabel <- to_categorical(trainingtarget)
testlabel <- to_categorical(testtarget)
print(trainlabel)

# Model
model <- keras_model_sequential()

model %>% 
  layer_dense(units=25, activation = 'relu', input_shape = c(9)) %>%
  layer_dense(units = 1, activation = 'softmax')

summary(model)

#Compile
model %>%
  compile(loss = "binary_crossentropy",
          optimizer = 'adam',
          metrics = 'accuracy')

# Fit Model
history <- model %>%
  fit(training, 
      trainlabel,
      epoch = 200,
      batch_size = 32,
      validation_split = 0.2)

plot(history)

#Evaluate model
model %>%
  evaluate(test,
           testlabel)

#Predictions
prob <- model %>%
  predict_proba(test)

pred <- model %>%
  predict_classes(test)

table(Predicted = pred, Actual = testtarget)
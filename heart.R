library(keras)

data <- read.csv("~/Downloads/heart-attack-prediction/data.csv", na.strings="?")

data <- as.matrix(data)
str(data)
dimnames(data) <- NULL

data[, 1:13] <- normalize(data[, 1:13])
data[, 14] <- as.numeric(data[, 14])
summary(data)

set.seed(7)
ind <- sample(2, nrow(data), replace = T, prob = c(0.7, 0.3))
training <- data[ind==1, 1:13]
test <- data[ind==2, 1:13]
trainingtarget <- data[ind==1,14]
testtarget <- data[ind==2, 14]

print(trainingtarget)

trainlabel <- to_categorical(trainingtarget)
testlabel <- to_categorical(testtarget)
print(trainlabel)

# Model
model <- keras_model_sequential()

model %>% 
  layer_dense(units=8, activation = 'relu', input_shape = c(13)) %>%
  layer_dense(units = 2, activation = 'softmax')

summary(model)

#Compile
model %>%
  compile(loss = "categorical_crossentropy",
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
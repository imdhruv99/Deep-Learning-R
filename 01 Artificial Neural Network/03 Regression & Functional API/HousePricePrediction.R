# importing keras
library(keras)

# Loading Dataset
df <- dataset_boston_housing()

# Train Test Split
c(train_data, train_labels) %<-% df$train
c(test_data, test_labels) %<-% df$test

# Normalizing the Data
train_data <- scale(train_data)

# Use means and Standard Deviation from training set to normalize the test data
col_means_train <- attr(train_data, 'scaled:center')
col_stdDev_train <- attr(train_data, 'scaled:scale')
test_data <- scale(test_data, center = col_means_train, scale = col_stdDev_train)

### Functional API

# input layer
inputs <- layer_input(shape = dim(train_data)[2])

# output layer structure of neural network
predictions <- inputs %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(units = 1)  

# create and compile model
model <- keras_model(inputs = inputs, outputs = predictions)

model %>% compile(
  optimizer = 'rmsprop', loss = 'mse', metrics = list('mean_absolute_error')
)

model %>% fit(train_data, train_labels, epoch=30, batch_size=100)

# Test Performance
score <- model %>% evaluate(test_data, test_labels)
score

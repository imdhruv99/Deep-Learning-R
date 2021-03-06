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

##### Saving and Restoring Models #####

model %>% save_model_hdf5("my_model.h5")

new_model <- load_model_hdf5("my_model.h5")

model %>% summary()
# or you can write summary(model)
new_model %>% summary()

checkpoint_dir <- "checkpoints"

dir.create(checkpoint_dir, showWarnings = FALSE)

filepath <- file.path(checkpoint_dir, "Epoch-{epoch:02d}.hdf5")

# Create checkpoint callback
cp_callback <- callback_model_checkpoint(filepath = filepath)

rm(model)
k_clear_session()

model_callback <- keras_model(inputs = inputs, outputs = predictions)
model_callback %>% compile(optimizer = 'rmsprop',loss = 'mse',
                           metrics = list("mean_absolute_error"))

model_callback %>% fit(train_data, train_labels, epochs = 30,
                       callbacks = list(cp_callback))

list.files(checkpoint_dir)

tenth_model <- load_model_hdf5(file.path(checkpoint_dir, "Epoch-10.hdf5"))

summary(tenth_model)

##### Only saving the best model

callbacks_best <- callback_model_checkpoint(filepath = "best_model.h5", monitor = "val_loss", 
                                            save_best_only = TRUE)

rm(model_callback)
k_clear_session()

model_cb_best <- keras_model(inputs = inputs, outputs = predictions)
model_cb_best %>% compile(optimizer = 'rmsprop',loss = 'mse',
                          metrics = list("mean_absolute_error"))

model_cb_best %>% fit(train_data, train_labels, epochs = 30, 
                      validation_data=list(test_data,test_labels),
                      callbacks = list(callbacks_best))

best_model <- load_model_hdf5("best_model.h5")

### Stopping the processing when we find the best model

callbacks_list <- list(
  callback_early_stopping(monitor = "val_loss",patience = 3),
  callback_model_checkpoint(filepath = "best_model_early_stopping.h5", monitor = "val_loss", save_best_only = TRUE)
)

rm(model_cb_best)
k_clear_session()

model_cb_early <- keras_model(inputs = inputs, outputs = predictions)
model_cb_early %>% compile(optimizer = 'rmsprop',loss = 'mse',
                           metrics = list("mean_absolute_error"))

model_cb_early %>% fit(train_data, train_labels, epochs = 100, 
                       validation_data=list(test_data,test_labels),
                       callbacks = callbacks_list)

best_model_early_stopping <- load_model_hdf5("best_model_early_stopping.h5")

k_clear_session()

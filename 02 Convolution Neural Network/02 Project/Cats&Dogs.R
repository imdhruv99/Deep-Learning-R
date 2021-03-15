# importing keras
library(keras)

### Train, Test, Validation Directories
trainDir <- file.path('02 Project/datasets/train')
testDir <- file.path('02 Project/datasets/test')
valDir <- file.path('02 Project/datasets/validation')

### using image_data_generator to read images from directories
### This func help to generate artificial data as well
trainDataGen <- image_data_generator(rescale = 1/255)
valDataGen <- image_data_generator(rescale = 1/255)

### Infinite loop for flowing images in batches of 20
trainGenerator <- flow_images_from_directory(
  trainDir,
  trainDataGen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = 'binary'
)

valGenerator <- flow_images_from_directory(
  valDir,
  valDataGen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = 'binary'
)

### Model Architecture
model <- keras_model_sequential() %>%
  
  ### 4 convolution layer with max pooling layers
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  # flatten layer
  layer_flatten() %>%
  
  # Dense layer
  layer_dense(units = 512, activation = 'relu') %>%
  
  # output dense layer
  layer_dense(units = 1, activation = 'sigmoid')

model


### Compiling the Model
model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_rmsprop(lr = 0.004),
  metrics = c('acc')
)
 
### Fitting the model
history <- model %>%fit_generator(
  trainGenerator,
  steps_per_epoch = 100,
  epochs = 20,
  validation_data = valGenerator,
  validation_steps = 50
)

model %>% save_model_hdf5("cats&Dogs.h5")

plot(history)
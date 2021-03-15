# importing keras
library(keras)

### Train, Test, Validation Directories
trainDir <- file.path('02 Project/datasets/train')
testDir <- file.path('02 Project/datasets/test')
valDir <- file.path('02 Project/datasets/validation')

####### Model Using Data Augmentation

### Data Augmentation
dataGen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)

TestDataGen <- image_data_generator(rescale = 1/255)

Train_Generator <- flow_images_from_directory(
  trainDir,
  dataGen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = 'binary'
)

Val_Generator <- flow_images_from_directory(
  valDir,
  TestDataGen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = 'binary'
)

### Model Architecture
DataAugModel <- keras_model_sequential() %>%
  
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

DataAugModel

### Compiling the Model
DataAugModel %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_rmsprop(lr = 0.004),
  metrics = c('acc')
)

### Fitting the model
DataAugHistory <- DataAugModel %>%fit_generator(
  Train_Generator,
  steps_per_epoch = 100,
  epochs = 20,
  validation_data = Val_Generator,
  validation_steps = 50
)

DataAugModel %>% save_model_hdf5("Augmentedcats&Dogs.h5")

plot(DataAugHistory)
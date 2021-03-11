### Importing Library

library(keras)

### Importing Dataset
df <- dataset_fashion_mnist()

### Test Train Split
c(train_images, train_labels) %<-% df$train
c(test_images, test_labels) %<-% df$test

### Classes
class_names = c('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Shirt', 'Sneaker', 'Bag', 'Ankel boot')

### Data Preprocessing
train_images <- train_images / 255
test_images <- test_images / 255

val_indices <- 1:5000
val_images <- train_images[val_indices,,]
part_train_images <- train_images[-val_indices,,]
val_labels <- train_labels[val_indices]
part_train_labels<- train_labels[5001:60000]

str(part_train_images)
part_train_images <- array_reshape(part_train_images, c(55000, 28, 28, 1))
val_images <- array_reshape(val_images, c(5000, 28, 28, 1))
test_images <- array_reshape(test_images, c(10000, 28, 28, 1))

### Define model architecture
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",input_shape = c(28, 28,1))

model <- model %>%
  layer_flatten() %>%
  layer_dense(units = 300, activation = "relu") %>%
  layer_dense(units = 100, activation = 'relu') %>% 
  layer_dense(units = 10, activation = "softmax")

model

### Configuring the Model
model %>% compile(
  optimizer = 'sgd', 
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)

### Training the Model
model %>% fit(part_train_images, part_train_labels, epochs = 10, batch_size=64, validation_data=list(val_images,val_labels))

### Test Performance
CNN_score <- model %>% evaluate(test_images, test_labels)
CNN_score

### Predicting on Test set
class_pred <- model %>% predict_classes(test_images)
class_pred[1:20]
class_names[class_pred[1:20]+1]
class_names[test_labels[1:20]+1]
plot(as.raster(test_images[1, , , ]), max = 255)

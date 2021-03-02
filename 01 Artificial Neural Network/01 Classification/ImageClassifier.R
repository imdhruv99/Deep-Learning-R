# importing keras
library(keras)

# importing dataset
df <- dataset_fashion_mnist()

# Test Train Split
c(train_images, train_labels) %<-% df$train
c(test_images, test_labels) %<-% df$test


# Exploring the Data structure
dim(train_images)
str(train_images)

# plotting the image
fobject <- train_images[25,,]
plot(as.raster(fobject, max = 255))

# classes available in dataset
class_name = c('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Shirt', 'Sneaker', 'Bag', 'Ankel boot')

class_name[train_labels[25]+1]

# Normalization
train_images <- train_images / 255
test_images <- test_images / 255

# splitting data for validation
# validation set is used for hyperparameters
val_indices <- 1:5000
val_images <- train_images[val_indices,,]
part_train_images <- train_images[-val_indices,,]
val_labels <- train_labels[val_indices]
part_train_labels <- train_labels[-val_indices]

# Flattening the dataset
# Using Sequential API

# initialization
model <- keras_model_sequential()

# structure of model
model %>%
  layer_flatten(input_shape = c(28, 28)) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

# compiling the model
model %>% compile(
  optimizer = 'sgd', loss = 'sparse_categorical_crossentropy', metrics = c('accuracy')
)

# Training the model
model %>% fit(part_train_images, part_train_labels, epochs = 30, batch_size=100, validation_data=list(val_images,val_labels))

# Test Performance

score <- model %>% evaluate(test_images, test_labels)

# Predicting on Test set

predictions <- model %>% predict(test_images)
predictions[1, ]
which.max(predictions[1, ])
# class_names[which.max(predictions[1, ])]
plot(as.raster(test_images[1, , ], max = 1))

class_pred <- model %>% predict_classes(test_images)
class_pred[1:20]
#---------#---------#---------#---------#---------#---------#---------#---------
#install.packages('keras')
rm(list=ls())
library(keras)

mnist <- dataset_mnist()
train.images <- mnist$train$x
train.labels <- mnist$train$y
test.images  <- mnist$test$x
test.labels  <- mnist$test$y

str(train.images)
dim(train.images) # 60000 28 28 (60,0000 28x28 images)
str(train.labels)
dim(train.labels) # 60000 

par(mfrow=c(3, 3))
par(mar=c(0, 0, 2, 0))
for (i in 1:9) {
  image(train.images[i, , ], 
        main=train.labels[i], 
        xaxt='n', 
        yaxt='n', 
        col=grey(0:12 / 12))	
}


network <- keras_model_sequential() %>%
  layer_dense(units=512, activation='relu', input_shape=c(28 * 28)) %>%
  layer_dense(units=10, activation='softmax')
  
network
summary(network) # same as previous

network %>% compile(optimizer='rmsprop',
                    loss='categorical_crossentropy', 
                    metrics=c('accuracy'))
                    
train.images <- array_reshape(train.images, c(60000, 28*28)) # 60000 x 784
train.images <- train.images / 255 # on [0, 1]
test.images <- array_reshape(test.images, c(10000, 28*28))   # 10000 x 784
test.images <- test.images / 255 

train.labels <- to_categorical(train.labels) # one-hot encodings
test.labels  <- to_categorical(test.labels)
train.labels[1, ]

network %>% fit(train.images, train.labels, epochs=5, batch_size=128)

metrics <- network %>% evaluate(test.images, test.labels)
metrics

network %>% predict_classes(test.images[1:10, ])

x <- array(rep(0, 2 * 3 * 2), dim=c(2, 3, 2))
str(x)
dim(x) # 3D tensor



# 2.3 Tensor Operations



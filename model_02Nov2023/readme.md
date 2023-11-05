# The first model

## Model specification

**Data**
- [cifar100](https://knowyourdata-tfds.withgoogle.com/#tab=STATS&dataset=cifar100) with corase labels (20 classes)
- No image augmentation 

**Model**
```python
model = keras.Sequential([
    keras.layers.Conv2D(8, (3,3), input_shape=(32,32,3), activation='relu',), 
    keras.layers.Dropout(0.2), 

    keras.layers.Conv2D(16, (3,3), activation='relu'), 
    keras.layers.MaxPool2D((2,2)),
    keras.layers.Dropout(0.2), 

    keras.layers.Conv2D(32, (3,3), activation='relu'), 
    keras.layers.MaxPool2D((2,2)),
    
    keras.layers.Flatten(),

    keras.layers.Dense(500, activation='relu'), 
    keras.layers.Dropout(0.2), 

    keras.layers.Dense(20, activation='softmax'), 
])
```

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 30, 30, 8)         224       
                                                                 
 dropout (Dropout)           (None, 30, 30, 8)         0         
                                                                 
 conv2d_1 (Conv2D)           (None, 28, 28, 16)        1168      
                                                                 
 max_pooling2d (MaxPooling2  (None, 14, 14, 16)        0         
 D)                                                              
                                                                 
 dropout_1 (Dropout)         (None, 14, 14, 16)        0         
                                                                 
 conv2d_2 (Conv2D)           (None, 12, 12, 32)        4640      
                                                                 
 max_pooling2d_1 (MaxPoolin  (None, 6, 6, 32)          0         
 g2D)                                                            
                                                                 
 flatten (Flatten)           (None, 1152)              0         
                                                                 
 dense (Dense)               (None, 500)               576500    
                                                                 
 dropout_2 (Dropout)         (None, 500)               0         
                                                                 
 dense_1 (Dense)             (None, 20)                10020     
                                                                 
=================================================================
Total params: 592552 (2.26 MB)
Trainable params: 592552 (2.26 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```

**Training**  
Batch size = 16
```python
model.compile(
    optimizer='adam', 
    loss=keras.losses.SparseCategoricalCrossentropy(), 
    metrics=['accuracy']
)
```


```
Epoch 50/50
3125/3125 [==============================] - 11s 4ms/step - loss: 0.6368 - accuracy: 0.7908 - val_loss: 2.6305 - val_accuracy: 0.4431

```


## Notes
- Overfitting 
- There are dropout layers but they aren't obvious in the filters
- orientation and colour selectivties were developed even though it's overfitting
    - idea: can make tools to access feature selectivities
    - Can learn more about input optimisation 
- 8 filters aren't enough for the first layer for both orientations and colours
- I want to know in what ways overfitting happens. Is it possible to be visualised? 
    - perhaps do PCA weights or outputs of the dense layer
- of those 500 units in the dense layer, 140 of those aren't active in all the testing data
    - how about the training data? 

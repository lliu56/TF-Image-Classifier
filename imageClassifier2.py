
import matplotlib
import numpy as np
import tensorflow as tf           
from tensorflow import keras        
from clothingList import c_list
# printing stuff 
import matplotlib.pyplot as plt


# first thing for ML projects is to import data
fashion_mnist = keras.datasets.fashion_mnist            # pre-labelled data from keras under the .dataset module 

#pull data from datasets and assign it to a variable (img, labels)
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data ()   # loads 60k imgs for training and the remaining 10k for testing

## showing the data using matplotlib 

# print(train_labels[0])      # int 
# print (train_images[0])     # list numpy arrays



## Defining neural net structure
# the .Sequential neural network is a sequence (in a row) of columns of nodes (neurons) where
# nodes of one colum only connects to the next and cant hop over
# we need to difine the layers (tip: code input and ouput layers first and then experiment w the hidden layer(s)):
# 1st: (28,20) matrix input for img but flattend input a column array...each node has 1 px (784)
# 2nd: hidden dense layer (128)... can play w the amount of nodes in the hidden layer
# 3rd: ouputlayer of 10 nodes (10 item labels) that probabilisticly tells what the pixel should be when it is that ite
# and .Dense means that each node is connected to every other node in the previous column
# overall structure: each connection between nodes are the weights, between each layers, the imput node value will times the weights and return a diff value
# and stores it in the next node (matrix multiplications in numpy)
# subsequently it ass through an activation fxn to specify some thresold (filtering mechanism) --> look it up 
model = keras.Sequential ([keras.layers.Flatten(input_shape=(28,28)), 
keras.layers.Dense (units= 128,activation = tf.nn.relu ),               # look up relu: new optimization technique: getting rid of negative numbers to increase computational speed
keras.layers.Dense(units=10,activation = tf.nn.softmax)])               # units is the number of nodes... softmax: picks the one with the greatest probability (returining the greatest as 1 and others as 0)


## compile our model --> take the keras built model and prepare it to do stuff 

# specifying the loss (cost) fxn and optimizer, loss tells how wrong the weights are and optimizers makes changes to the weights to minimize the loss
model.compile (optimizer = tf.optimizers.Adam(learning_rate =0.00075), loss ='sparse_categorical_crossentropy')

## train model on all 60k images...returns a trained model## 
model.fit (train_images,train_labels,epochs= 5)         # sepecifies how many times you wanna go through the whole neural network fxn with epoch

## test the model on all 10k images##
test_loss = model.evaluate (test_images,test_labels)

#show it w matplotlib

## use trained model to make predictions ## 
def get_prediction(test_im_index=int):
    predictions = model.predict(test_images)
    # print (type(predictions))
    # print (type(predictions[test_im_index]))
    predicted_index =  np.argmax (predictions[test_im_index])
    labels = test_labels[test_im_index] 
    if predicted_index == labels:
        print (f'The model correctly predicted the image as a {c_list[predicted_index]}')
    else: 
        print (f''' The model was incorrect :( 
                    The test image was a {c_list[labels]},
                    but the prediton was {c_list[predicted_index]}''')
    plt.imshow (test_images[test_im_index],cmap ='gray',vmin=0, vmax=225)       # making sure the color map (cmap) is gray scale
    plt.show()
   



get_prediction(258)


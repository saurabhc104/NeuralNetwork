#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 07:36:45 2017

@author: saurabh
"""

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import os

path = "/home/saurabh/Python_Practice/DigitRecognizer"
os.chdir(path)

train_df = pd.read_csv("train.csv")
df_train_output = train_df.label
df_train_input  = train_df.iloc[:,1:]

n_nodes_hl1 = 500
n_output = 10
batch_size = 1000

x = tf.placeholder("float",[None,784])
y = tf.placeholder("float",[None,n_output])




def neural_network_model(data):
    
    hl1 = {'weights' : tf.Variable(tf.random_normal([784,n_nodes_hl1])) , 
           'biases'  : tf.Variable(tf.random_normal([n_nodes_hl1]))}
    output_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl1,n_output])),
                    'biases' : tf.random_normal([n_output])}
    
    l1 = tf.add( (tf.matmul(data,hl1['weights'])), hl1['biases'] )
    l1 = tf.nn.relu(l1)
    
    output = tf.add(tf.matmul(l1,output_layer['weights']) , output_layer['biases'] )
    
    return output

def train_neural_network(data):
    prediction = neural_network_model(data)
    softmax_output = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)
    cost = tf.reduce_mean(softmax_output)
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    n_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epochs in range(n_epochs):
            epochs_loss = 0
            #batch_x, batch_y =  tf.train.batch([df_train_input,df_train_output],batch_size)
           for i in range(len(df_train_input)):
               _, c =  sess.run([optimizer,cost], feed_dict = { x:df_train_input[i], y:df_train_output[i] })
               epochs_loss += c
            
            
           print("Epoch",epochs, "completed out of", n_epochs,"loss: ", epochs_loss)
            
        correct = tf.equal( tf.argmax(prediction,1), tf.argmax(y,1))
        
        

train_neural_network(x)

        
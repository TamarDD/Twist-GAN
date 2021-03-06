# -*- coding: utf-8 -*-

import tensorflow.keras as k
from keras.layers import Input, Dense, Reshape, Flatten, Dropout,Concatenate,Lambda
from keras.layers import BatchNormalization, Activation
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf



class TwistGAN():
  def __init__(self, input_size, black_box, noise_shape=100):

    self.input_size = input_size
    self.noise_shape = noise_shape
    self.bb = black_box
    
    optimizer = Adam(0.0002, 0.5)


    self.discriminator = self._buildDiscriminator()
    self.discriminator.compile(loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy'])

    self.generator = self._buildGenerator()
    c = Input(shape=(1,))
    noise = Input(shape=(self.noise_shape))
    fake_sample = self.generator((noise,c))

    self.discriminator.trainable = False
    check = self.discriminator((fake_sample,c))

    self.gan = Model([noise,c],check)
    self.gan.compile(loss='binary_crossentropy', optimizer=optimizer)


  def _buildDiscriminator(self):
    
    
    input_sample = Input(shape=(self.input_size,), name='input_sample')
    input_c = Input(shape=(1,), name="c")
    #input_y = Input(shape=(1,), name="y")
    model = Sequential()
    concatenate_layer = Concatenate(axis=1)([input_sample, input_c])
    dense1 = Dense(512)(concatenate_layer)
    active_1 = LeakyReLU(alpha=0.2)(dense1)
    dense2 = Dense(256)(active_1)
    active_2 = LeakyReLU(alpha=0.2)(dense2)
    output_dense = Dense(1, activation='sigmoid')(active_2)

    discriminator = Model(inputs=[input_sample, input_c], outputs=output_dense)
    
    

    
    return discriminator


  def _buildGenerator(self):

    input_sample = Input(shape=(self.noise_shape,))
    input_c = Input(shape=(1,), name="c")
    model = Sequential()
    concatenate_layer = Concatenate(axis=1)([input_sample, input_c])
    dense1 = Dense(256)(concatenate_layer)
    active_1 = LeakyReLU(alpha=0.2)(dense1)
    bt1 = BatchNormalization(momentum=0.8)(active_1)
    dense2 = Dense(512)(bt1)
    active_2 = LeakyReLU(alpha=0.2)(dense2)
    bt2 = BatchNormalization(momentum=0.8)(active_2)
    dense3 = Dense(1024)(bt2)
    active_3 = LeakyReLU(alpha=0.2)(dense3)
    bt3 = BatchNormalization(momentum=0.8)(active_3)
    output_dense = Dense(self.input_size)(bt3)

    generator =  Model([input_sample,input_c],outputs= output_dense)

    return generator


  def evalute_model(self, number_of_sampels):
    noise = np.random.normal(0, 1, (number_of_sampels, self.noise_shape))
    c = np.random.rand(number_of_sampels,1)
    fake_sampels = self.generator([noise,c], training=False)
    pred = tf.round(self.discriminator([fake_sampels,c], training=False))
    real_pred = np.array(fake_sampels)[np.where(pred == 1)[0]]
    fake_pred = np.array(fake_sampels)[np.where(pred == 0)[0]]
    print(f"Predicted as real : {len(real_pred)}")
    print(f"Predicted as fake : {len(fake_pred)}")
    return real_pred, fake_pred

  def train(self,x_train,epochs):

    batch_size = 128
    c = np.random.rand(batch_size,1)

    y_real = np.ones((batch_size, 1))
    y_fake = np.zeros((batch_size, 1))

    loss_g = []
    epoches_g = []

    loss_d = []
    epoches_d = []

    for epoch in range(epochs):

      idx = np.random.randint(0, x_train.shape[0], batch_size)


      real_sampels = x_train.iloc[idx]
      rf_y = self.bb.predict(real_sampels) 

      noise = np.random.normal(0, 1, (batch_size, self.noise_shape))

      fake_samples = self.generator.predict([noise,c])

      d_loss_fake = self.discriminator.train_on_batch((fake_samples,c), y_fake)
      d_loss_real = self.discriminator.train_on_batch((fake_samples,rf_y), y_real)
      
      d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

      # TRAIN GENERTOR

      noise = np.random.normal(0, 1, (batch_size, self.noise_shape))
      g_loss = self.gan.train_on_batch((noise,c), y_real)
      

      loss_g.append(g_loss)
      loss_d.append(d_loss[0])
      epoches_g.append(epoch)
      epoches_d.append(epoch)

      if epoch % 500 == 0:

        print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))


    plt.plot(epoches_g,loss_g)
    plt.plot(epoches_g,loss_d)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()

from keras.layers import Dense, Input, Conv2D, Flatten, Reshape, Conv2DTranspose, BatchNormalization, Dropout
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.datasets import cifar100
from keras import Sequential
import tensorflow_addons as tfa

import cv2
import numpy as np
import matplotlib.pyplot as plt

def rgb2gray(rgb):
  return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

def load_datasets():
  print("Loading train dataset...")
  print("Loading test dataset...")
  (y_train, _), (y_test, _) = cifar100.load_data()
  print("Preparing train dataset...")
  y_train = y_train.astype('float32') / 255
  x_train = np.expand_dims([rgb2gray(x) for x in y_train], axis=3)
  print("Preparing test dataset...")
  y_test = y_test.astype('float32') / 255
  x_test = np.expand_dims([rgb2gray(x) for x in y_test], axis=3)
  print("Finished.")
  return (x_train, y_train), (x_test, y_test)

class Encoder(Model):
  def __init__(self, layers_filters, use_normalization=False, use_dropout=False):
    super().__init__(name='encoder')
    def layers():
      for filters in layers_filters:
        yield Conv2D(filters, kernel_size=3, padding='same', activation='relu')
        if use_normalization:
          yield BatchNormalization()
        if use_dropout:
          yield Dropout(use_dropout)
    self.net = Sequential(list(layers()))

  def call(self, inputs): return self.net(inputs)

class Decoder(Model):
  def __init__(self, layers_filters):
    super().__init__(name='decoder')
    def layers():
      for filters in reversed(layers_filters):
        yield Conv2DTranspose(filters, kernel_size=3, padding='same', activation='relu')
      yield Conv2DTranspose(3, kernel_size=3, padding='same', activation='sigmoid')

    self.net = Sequential(list(layers()))

  def call(self, inputs): return self.net(inputs)

class Autoencoder(Model):
  def __init__(self, layers_filters, use_normalization=False, use_dropout=False):
    super().__init__(name='autoencoder')
    def layers():
      yield Encoder(layers_filters, use_normalization, use_dropout)
      yield Decoder(layers_filters)
    self.net = Sequential(list(layers()))
  def call(self, inputs): return self.net(inputs)

def train_model(model, weight_decay=0, savename=None):
  callbacks = [
    ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=3, min_lr=0.5e-6),
    ModelCheckpoint(filepath=f'models/{savename}.ckpt', monitor='val_loss', save_best_only=True)
  ]

  optimizer = tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=weight_decay)

  model.compile(loss='mse', optimizer=optimizer)
  return model.fit(x_train, y_train, validation_data=(x_test, y_test),
                   epochs=20, batch_size=16, callbacks=callbacks)

def plot_histories(*pairs, parameters):
  figure, axes = plt.subplots(ncols=len(parameters), figsize=(10, 3))
  (models, histories) = list(zip(*pairs))

  labels = [x.name for x in models]
  for axis, parameter in zip(axes.reshape(-1), parameters):
    for history in histories:
      axis.plot(history.history[parameter])
    axis.set_ylabel(parameter)
    axis.set_xlabel('epoch')
    axis.legend(labels, loc='upper right')
  plt.show()

if __name__ == '__main__':
  (x_train, y_train), (x_test, y_test) = load_datasets()

  layers = [32, 64, 128]
  autoencoder = Autoencoder(layers, use_dropout=0.00)
  autoencoder.build(input_shape=(None, None, None, 1))
  history = train_model(autoencoder, savename='dropout-0.00.{epoch:03d}')
  plot_histories((autoencoder, history), parameters=['val_loss', 'loss', 'lr'])

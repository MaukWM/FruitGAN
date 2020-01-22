from __future__ import print_function, division

import os
import pickle
import time

from PIL import Image
# from keras.datasets import mnist
from keras import models
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, MaxPooling2D, Conv2DTranspose
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys

import numpy as np

class GAN():
    def __init__(self):
        self.img_rows = 48
        self.img_cols = 48
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 64
        self.flat_dim = self.img_rows * self.img_cols * self.channels

        optimizer = Adam(0.0001, 0.5)

        # Build and compile the discriminator
        # self.discriminator = self.build_discriminator()
        # self.discriminator.compile(loss='binary_crossentropy',
        #     optimizer=optimizer,
        #     metrics=['accuracy'])
        #
        # # Build the generator
        # self.generator = self.build_generator()

        # Create encoder and decoder model
        input_img = Input(shape=(self.flat_dim,))

        encoded = Dense(128, activation='relu')(input_img)
        encoded = Dense(self.latent_dim, activation='relu')(encoded)

        decoded = Dense(self.latent_dim, activation='relu')(encoded)
        decoded = Dense(256, activation='relu')(decoded)
        decoded = Dense(self.flat_dim, activation='sigmoid')(decoded)

        # Map input to reconstruction
        self.autoencoder = Model(input_img, decoded)

        self.encoder = Model(input_img, encoded)
        self.encoder.summary()

        encoded_input = Input(shape=(self.latent_dim,))

        decoder_layer = self.autoencoder.layers[-1]

        self.decoder = Model(encoded_input, decoder_layer(encoded_input))
        self.decoder.summary()

        self.autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy')
        self.autoencoder.summary()

        # Initialize noise for generated pictures
        self.sample_noise = np.random.normal(0, 1, (5 * 5, self.latent_dim))  # 5 * 5 = r * c

    # def build_encoder(self):
    #     model = Sequential()
    #
    #     model.add()
    #
    #     model.summary()
    #
    #     return model
    #
    # def build_decoder(self):
    #     model = Sequential()
    #
    #     model.add()
    #
    #     model.add(Reshape(self.img_shape))
    #
    #     return model

    def build_generator(self):

        model = Sequential()

        model.add(Dense(12*12*256, use_bias=False, input_shape=(self.latent_dim,)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Reshape((12, 12, 256)))

        model.add(Conv2DTranspose(128, (7, 7), strides=(1, 1), use_bias=False, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), use_bias=False, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', activation='sigmoid'))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(64, 5, input_shape=self.img_shape, padding='same', strides=2))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, 5, input_shape=self.img_shape, padding='same', strides=2))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, 5, padding='same', strides=2))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))
        model.add(Flatten())

        model.add(Dense(32))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def load_images(self, path="images/preprocessed/48x48/oranges/"):
        result = np.zeros(shape=(len(os.listdir(path)), self.img_rows * self.img_cols * self.channels))
        idx = 0
        for file in os.listdir(path):
            img = Image.open(os.path.join(path, file))
            img = img.convert("RGB")
            img = np.array(img).flatten()

            result[idx] = img

            idx += 1
        return result

    def train(self, epochs, batch_size=32, sample_interval=50, save_interval=1500):

        loss = []

        # # Load the dataset
        # (X_train, _), (_, _) = mnist.load_data()

        # Load the images
        X_train = self.load_images()

        # Normalize
        X_train = X_train / 255

        history = self.autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=True,
                                       validation_data=(X_train, X_train))

        plt.plot(history.history['loss'])

        self.sample_images(X_train)

    def sample_images(self, X_train):
        encoded_imgs = self.encoder.predict(X_train)
        decoded_imgs = self.decoder.predict(encoded_imgs)

        n = 10  # how many digits we will display
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(X_train[i].reshape(self.img_shape))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(decoded_imgs[i].reshape(self.img_shape))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

        plt.close()


if __name__ == '__main__':
    gan = GAN()
    # gan.train(epochs=40, batch_size=32, sample_interval=20, save_interval=4)
    # gan.generator.load_weights("saved_models/1578953900-generator.h5")
    # gan.discriminator.load_weights("saved_models/1578953900-discriminator.h5")
    gan.train(epochs=10, batch_size=32, sample_interval=500, save_interval=5000)
    # gan.combined.load_weights("saved_models/1578953512-combined.h5")
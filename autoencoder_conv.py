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

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        # Map input to reconstruction

        img = Input(shape=self.img_shape)
        z = self.encoder(img)
        output = self.decoder(z)

        self.autoencoder = Model(img, output)

        self.autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy')
        self.autoencoder.summary()

        # Initialize noise for generated pictures
        self.sample_noise = np.random.normal(0, 1, (5 * 5, self.latent_dim))  # 5 * 5 = r * c

    def build_encoder(self):
        model = Sequential()

        # model.add(Dense(128, activation='relu', input_shape=self.img_shape))
        # model.add(Flatten())
        #
        # model.add(Dense(self.latent_dim, activation='relu'))

        model.add(Conv2D(32, 5, input_shape=self.img_shape, padding='same', strides=2))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, 5, input_shape=self.img_shape, padding='same', strides=2))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, 5, padding='same', strides=2))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))
        model.add(Flatten())

        model.add(Dense(self.latent_dim, activation='sigmoid'))
        model.add(LeakyReLU(alpha=0.2))

        model.summary()

        # img = Input(shape=self.img_shape)
        # encoded = model(img)

        return model

    def build_decoder(self):
        model = Sequential()

        # model.add(Dense(128, activation='relu', input_shape=(self.latent_dim,)))
        # model.add(Dense(self.flat_dim, activation='sigmoid'))
        #
        # # print(self.flat_dim, self.img_shape[0] * self.img_shape[1] * self.img_shape[2])
        #
        # model.add(Reshape(self.img_shape))
        # model.summary()

        model.add(Dense(12 * 12 * 128, use_bias=False, input_shape=(self.latent_dim,)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Reshape((12, 12, 128)))

        model.add(Conv2DTranspose(64, (7, 7), strides=(1, 1), use_bias=False, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(32, (5, 5), strides=(2, 2), use_bias=False, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', activation='sigmoid'))

        model.summary()
        # noise = Input(shape=(self.latent_dim,))
        # decoded = model(noise)

        return model

    def load_images(self, path="images/preprocessed/48x48/oranges/"):
        result = np.zeros(shape=(len(os.listdir(path)), self.img_rows, self.img_cols, self.channels))
        idx = 0
        for file in os.listdir(path):
            img = Image.open(os.path.join(path, file))
            img = img.convert("RGB")
            img = np.array(img)

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

        try:
            for i in range(int(epochs/sample_interval)):
                print("True Epoch: " + str(i * sample_interval))
                history = self.autoencoder.fit(X_train, X_train, epochs=sample_interval, batch_size=batch_size, shuffle=True,
                                               validation_data=(X_train, X_train))

                self.sample_images(X_train, i * epochs, noise=False)
                self.sample_images(X_train, i * epochs)

                loss.append(history.history['loss'])
        except KeyboardInterrupt:
            pass

        plt.plot(loss)

    def sample_images(self, X_train, epoch, noise=True):
        if noise:
            r, c = 5, 5
            gen_imgs = self.decoder.predict(self.sample_noise, batch_size=5 * 5)

            # Rescale images 0 - 1
            gen_imgs = 0.5 * gen_imgs + 0.5

            fig, axs = plt.subplots(r, c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                    axs[i, j].axis('off')
                    cnt += 1
            fig.savefig("images/generated-%d.png" % epoch)
            plt.close()
        else:
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
                plt.savefig("images/reconstructed%d.png" % epoch)

        plt.close()


if __name__ == '__main__':
    gan = GAN()
    # gan.train(epochs=40, batch_size=32, sample_interval=20, save_interval=4)
    # gan.generator.load_weights("saved_models/1578953900-generator.h5")
    # gan.discriminator.load_weights("saved_models/1578953900-discriminator.h5")
    gan.train(epochs=15000, batch_size=32, sample_interval=50, save_interval=5000)
    # gan.combined.load_weights("saved_models/1578953512-combined.h5")
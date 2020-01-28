from __future__ import print_function, division

import os
import pickle
import time

from PIL import Image
# from keras.datasets import mnist
from keras import models
from keras.backend import binary_crossentropy
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, MaxPooling2D, Conv2DTranspose, Lambda, K
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.losses import mse
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
        self.batch_size = 32

        # VAE model = encoder + decoder
        # build encoder model
        self.inputs = Input(shape=(self.flat_dim,), name='encoder_input')
        # self.inputs = Reshape(target_shape=(self.flat_dim,), name='encoder_input_reshape')(self.raw_inputs)
        x = Dense(128, activation='relu')(self.inputs)
        self.z_mean = Dense(self.latent_dim, name='z_mean')(x)
        self.z_log_var = Dense(self.latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([self.z_mean, self.z_log_var])

        # instantiate encoder model
        self.encoder = Model(self.inputs, [self.z_mean, self.z_log_var, z], name='encoder')
        self.encoder.summary()
        # plot_model(self.encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

        # build decoder model
        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        y = Dense(128, activation='relu')(latent_inputs)
        self.outputs = Dense(self.flat_dim, activation='sigmoid')(y)
        # self.outputs = Reshape(target_shape=self.img_shape)(self.outputs)

        # instantiate decoder model
        self.decoder = Model(latent_inputs, self.outputs, name='decoder')
        self.decoder.summary()
        # plot_model(self.decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

        # instantiate VAE model
        self.outputs = self.decoder(self.encoder(self.inputs)[2])
        self.vae = Model(self.inputs, self.outputs, name='vae_mlp')

        self.sample_noise = np.random.normal(0, 1, (5 * 5, self.latent_dim))  # 5 * 5 = r * c

    # reparameterization trick
    # instead of sampling from Q(z|X), sample epsilon = N(0,I)
    # z = z_mean + sqrt(var) * epsilon
    def sampling(self, args):
        """Reparameterization trick by sampling from an isotropic unit Gaussian.
        # Arguments
            args (tensor): mean and log of variance of Q(z|X)
        # Returns
            z (tensor): sampled latent vector
        """

        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

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

    def plot_results(self,
                     models,
                     data,
                     batch_size=128,
                     model_name="vae_fruit"):
        """Plots labels and MNIST digits as a function of the 2D latent vector
        # Arguments
            models (tuple): encoder and decoder models
            data (tuple): test data and label
            batch_size (int): prediction batch size
            model_name (string): which model is using this function
        """

        encoder, decoder = models
        x_test, y_test = data
        # os.makedirs(model_name, exist_ok=True)

        # filename = os.path.join(model_name, "vae_mean.png")
        # # display a 2D plot of the digit classes in the latent space
        # z_mean, _, _ = encoder.predict(x_test,
        #                                batch_size=batch_size)
        # plt.figure(figsize=(12, 10))
        # plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test[0:128])
        # plt.colorbar()
        # plt.xlabel("z[0]")
        # plt.ylabel("z[1]")
        # plt.savefig(filename)
        # plt.show()

        # filename = os.path.join(model_name, "digits_over_latent.png")
        # display a 30x30 2D manifold of digits
        n = 5
        digit_size = self.img_rows
        figure = np.zeros((digit_size * n, digit_size * n))
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x = np.linspace(-4, 4, n)
        grid_y = np.linspace(-4, 4, n)[::-1]

        # for i, yi in enumerate(grid_y):
        #     for j, xi in enumerate(grid_x):
        #         z_sample = np.array([[xi, yi]])
        #         x_decoded = decoder.predict(z_sample)
        #         digit = x_decoded[0].reshape(digit_size, digit_size)
        #         figure[i * digit_size: (i + 1) * digit_size,
        #         j * digit_size: (j + 1) * digit_size] = digit

        plt.figure(figsize=(10, 10))
        start_range = digit_size // 2
        end_range = (n - 1) * digit_size + start_range + 1
        pixel_range = np.arange(start_range, end_range, digit_size)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.imshow(figure, cmap='Greys_r')
        plt.savefig(filename)
        plt.show()

    def train(self, epochs, batch_size=32, sample_interval=50, save_interval=1500):
        # # Load the dataset
        # (X_train, _), (_, _) = mnist.load_data()

        loss = []

        # Load the images
        X_train = self.load_images()

        # image_size = X_train.shape[1]
        # original_dim = image_size * image_size

        # Normalize
        X_train = X_train / 255

        # Reshape
        X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))

        models = (self.encoder, self.decoder)
        data = (X_train, X_train)

        # VAE loss = mse_loss or xent_loss + kl_loss
        reconstruction_loss = mse(self.inputs, self.outputs)
        reconstruction_loss *= self.img_rows * self.img_cols
        kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        print(reconstruction_loss.shape, kl_loss.shape)
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer='adam')
        self.vae.summary()
        # plot_model(self.vae,
        #            to_file='vae_mlp.png',
        #            show_shapes=True)

        try:
            for i in range(1, int(epochs/sample_interval)):
                print("True Epoch: " + str(i * sample_interval))
                # train the autoencoder
                history = self.vae.fit(X_train,
                                       shuffle=True,
                                       epochs=int(i * sample_interval),
                                       batch_size=batch_size,
                                       validation_data=(X_train, None))  # TODO: make test
                self.vae.save_weights('vae_mlp_fruit.h5')

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

            gen_imgs = np.stack(gen_imgs).reshape((5 * 5, self.img_rows, self.img_cols, self.channels))

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
            decoded_imgs = self.decoder.predict(encoded_imgs[2])

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
    gan.train(epochs=500, batch_size=gan.batch_size, sample_interval=50, save_interval=5000)
    # gan.combined.load_weights("saved_models/1578953512-combined.h5")
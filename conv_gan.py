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
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 64

        optimizer = Adam(0.0001, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

        # Initialize noise for generated pictures
        self.sample_noise = np.random.normal(0, 1, (5 * 5, self.latent_dim))  # 5 * 5 = r * c

    def build_generator(self):

        model = Sequential()

        model.add(Dense(8*8*256, use_bias=False, input_shape=(self.latent_dim,)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Reshape((8, 8, 256)))

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

        model.add(Conv2D(128, 5, padding='same', strides=2))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))
        model.add(Flatten())

        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def load_images(self, path="images/preprocessed/32x32/apples/"):
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

        D_acc = []
        D_fake_ratio = []
        D_loss = []
        G_loss = []

        # # Load the dataset
        # (X_train, _), (_, _) = mnist.load_data()

        # Load the images
        X_train = self.load_images()

        # Normalize
        X_train = X_train / 255

        # # Rescale -1 to 1
        # X_train = X_train / 127.5 - 1.
        # X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        try:
            for epoch in range(epochs):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]

                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                D_loss.append(d_loss[0])

                # Calculate amount rated as true/false by discriminator
                real_preds = self.discriminator.predict(imgs)
                fake_preds = self.discriminator.predict(gen_imgs)

                real_preds_norm = [0 if pred < 0.5 else 1 for pred in real_preds]
                fake_preds_norm = [0 if pred < 0.5 else 1 for pred in fake_preds]

                tot_real_preds = real_preds_norm.count(1) + fake_preds_norm.count(1)
                tot_fake_preds = real_preds_norm.count(0) + fake_preds_norm.count(0)

                D_fake_ratio.append(tot_fake_preds / (tot_fake_preds + tot_real_preds))

                # ---------------------
                #  Train Generator
                # ---------------------

                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Train the generator (to have the discriminator label samples as valid)
                g_loss = self.combined.train_on_batch(noise, valid)

                G_loss.append(g_loss)

                # Plot the progress
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

                D_acc.append(d_loss[1])

                # If at save interval => save generated image samples
                if epoch % sample_interval == 0:
                    self.sample_images(epoch)

                if epoch % save_interval == 0:
                    self.discriminator.trainable = True
                    self.combined.save_weights("saved_models/%d-combined.h5" % time.time())
                    self.discriminator.save_weights("saved_models/%d-discriminator.h5" % time.time())
                    self.generator.save_weights("saved_models/%d-generator.h5" % time.time())
                    self.discriminator.trainable = False

        except KeyboardInterrupt:
            pass

        # print(D_acc)
        # print(D_fake_ratio)
        # print(D_loss)
        # print(G_loss)

        plt.plot(D_acc, label='Discriminator Accuracy')
        plt.legend()
        plt.title(label='Discriminator Accuracy')
        plt.savefig("images/plots/%d-disc_acc.png" % time.time())
        plt.show()

        plt.clf()
        plt.plot(D_loss, label='Discriminator Loss')
        plt.plot(G_loss, label='Generator Loss')
        plt.legend()
        plt.title(label='Disc-Gen Loss')
        plt.savefig("images/plots/%d-disc-gen_loss.png" % time.time())
        plt.show()

        plt.clf()
        plt.plot(D_fake_ratio, label='Fake guesses')
        plt.legend()
        plt.title(label='Discriminator Ratio Fake Guesses')
        plt.savefig("images/plots/%d-disc_fake_guesses.png" % time.time())
        plt.show()

        history = {"D_acc": D_acc,
                   "D_fake_ratio": D_fake_ratio,
                   "D_loss": D_loss,
                   "G_loss": G_loss}

        history_file = open("histories/%d-history.pkl" % time.time(), "wb")

        pickle.dump(history, history_file)

    def sample_images(self, epoch):
        r, c = 5, 5
        gen_imgs = self.generator.predict(self.sample_noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    gan = GAN()
    # gan.train(epochs=40, batch_size=32, sample_interval=20, save_interval=4)
    # gan.generator.load_weights("saved_models/1578953900-generator.h5")
    # gan.discriminator.load_weights("saved_models/1578953900-discriminator.h5")
    gan.train(epochs=50000, batch_size=32, sample_interval=20, save_interval=5000)
    # gan.combined.load_weights("saved_models/1578953512-combined.h5")
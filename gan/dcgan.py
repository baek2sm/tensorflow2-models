from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, ZeroPadding2D, UpSampling2D
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU, Activation, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt


class DCGAN():
    def __init__(self):
        self.img_shape = (28, 28, 1)  # Target image shape, values of width and height should be multiples of 4.
        self.latent_dim = (100, )  # Generator input shape
        self.generator_initial_size = (self.img_shape[0]//4, self.img_shape[1]//4, 128)  # There are 2UpSampling layers.

        # Build discriminator and generator
        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()

        # Compile discriminator
        self.discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=1e-5), metrics=['accuracy'])
        self.discriminator.trainable = False

        # Compile combined model
        combined_model_input = Input(shape=self.latent_dim)
        combined_model_output = self.discriminator(self.generator(combined_model_input))
        self.combined_model = Model(combined_model_input, combined_model_output)
        self.combined_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=1e-5), metrics=['accuracy'])

    def build_generator(self):
        g_input = Input(shape=self.latent_dim)

        x = Dense(units=np.prod(self.generator_initial_size), activation='relu')(g_input)
        x = Reshape(target_shape=self.generator_initial_size)(x)
        x = UpSampling2D()(x)

        x = Conv2D(filters=128, kernel_size=3, padding='same')(x)
        x = BatchNormalization(momentum=0.99)(x)
        x = Activation('relu')(x)
        x = UpSampling2D()(x)

        x = Conv2D(filters=64, kernel_size=3, padding='same')(x)
        x = BatchNormalization(momentum=0.99)(x)
        x = Activation('relu')(x)

        x = Conv2D(self.img_shape[2], kernel_size=3, padding='same')(x)
        g_output = Activation('tanh')(x)

        return Model(g_input, g_output)

    def build_discriminator(self):
        d_input = Input(shape=self.img_shape)
        x = Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(d_input)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(rate=0.25)(x)

        x = Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(x)
        x = BatchNormalization(momentum=0.99)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(rate=0.25)(x)

        x = Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(x)
        x = BatchNormalization(momentum=0.99)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(rate=0.25)(x)

        x = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(x)
        x = BatchNormalization(momentum=0.99)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(rate=0.25)(x)

        x = Flatten()(x)
        d_output = Dense(units=1, activation='sigmoid')(x)

        return Model(d_input, d_output)

    def train_discriminator(self, X_train, batch_size):
        # Create labels
        valid_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        # Train with real data
        batch_index = np.random.randint(0, X_train.shape[0], batch_size)
        X_train_batch = X_train[batch_index]
        self.discriminator.train_on_batch(X_train_batch, valid_labels)

        # Train with fake data
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim[0]))
        generated_X = self.generator.predict(noise)
        self.discriminator.train_on_batch(generated_X, fake_labels)

    def train_generator(self, batch_size):
        valid_labels = np.ones((batch_size, 1))
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim[0]))
        self.combined_model.train_on_batch(noise, valid_labels)

    def fit(self, X_train, batch_size, epochs, save_interval=50):
        for epoch in range(epochs):
            self.train_discriminator(X_train, batch_size)
            self.train_generator(batch_size)

            # Save generated images
            if epoch % save_interval == 0:
                self.save_images(epoch)

    def save_images(self, epoch):
        # Generate Images
        r, c, = 5, 5
        noise = np.random.normal(0, 1, (r*c, self.latent_dim[0]))
        generated_images = self.generator.predict(noise)
        generated_images = generated_images * 0.5 + 0.5

        # Plot and save fig
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(generated_images[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig('./saved_images/{}.png'.format(epoch))
        plt.close()


if __name__ == '__main__':
    # Load and rescale images, from -1 to 1
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train / 127.5 - 1.

    # Train DCGAN
    dcgan = DCGAN()
    dcgan.fit(X_train, batch_size=32, epochs=10000, save_interval=50)

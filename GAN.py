import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, UpSampling2D
from tensorflow.keras.models import Sequential
import numpy as np
import os
from PIL import Image

# Parameters
image_size = 128
batch_size = 16
epochs = 5
noise_dim = 100
image_folder = 'Preprocessed-Photos'
save_interval = 1000
output_folder = 'Generated-Images'

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Define the Generator
def build_generator():
    model = Sequential([
        Dense(16 * 16 * 128, use_bias=False, input_shape=(noise_dim,)),
        BatchNormalization(),
        LeakyReLU(),
        Reshape((16, 16, 128)),
        UpSampling2D(),
        Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(),
        Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(),
        Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(),
        Conv2D(3, (7, 7), padding='same', activation='sigmoid', use_bias=False)
    ])
    return model

# Define the Discriminator
def build_discriminator():
    model = Sequential([
        Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[image_size, image_size, 3]),
        LeakyReLU(),
        Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        LeakyReLU(),
        Conv2D(256, (5, 5), strides=(2, 2), padding='same'),
        LeakyReLU(),
        Flatten(),
        Dense(1)
    ])
    return model

# Define the GAN loss functions
def gan_loss_fn(disc_real_output, disc_fake_output):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    disc_loss = bce(tf.ones_like(disc_real_output), disc_real_output) + \
                bce(tf.zeros_like(disc_fake_output), disc_fake_output)
    return disc_loss

def generator_loss_fn(disc_fake_output):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return bce(tf.ones_like(disc_fake_output), disc_fake_output)

# Load and preprocess a single image
def load_and_preprocess_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.cast(image, tf.float32) / 255.0
    return image

# Prepare dataset
def prepare_dataset(image_folder, batch_size):
    file_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.endswith('.jpg')]
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.map(lambda x: tf.py_function(func=lambda p: load_and_preprocess_image(p), inp=[x], Tout=tf.float32))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Save generated images
def save_image(image_array, step_count):
    image_array = (image_array * 255).astype(np.uint8)  # Convert pixel values from [0, 1] to [0, 255]
    image_pil = Image.fromarray(image_array)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_pil.save(os.path.join(output_folder, f'generated_{step_count}.png'))

# Training step function
@tf.function
def train_step(generator, discriminator, batch_images, disc_optimizer, gen_optimizer):
    noise = tf.random.normal([batch_size, noise_dim])
    generated_images = generator(noise, training=True)
    
    with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
        disc_real_output = discriminator(batch_images, training=True)
        disc_fake_output = discriminator(generated_images, training=True)
        
        disc_loss = gan_loss_fn(disc_real_output, disc_fake_output)
        gen_loss = generator_loss_fn(disc_fake_output)
    
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    
    disc_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    
    return gen_loss, disc_loss

# Train GAN
def train_gan(generator, discriminator, dataset, epochs, batch_size):
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    
    for epoch in range(epochs):
        print(f'Starting epoch {epoch + 1}/{epochs}')
        for batch_images in dataset:
            gen_loss, disc_loss = train_step(generator, discriminator, batch_images, disc_optimizer, gen_optimizer)
        
        print(f'Epoch {epoch + 1}: Generator Loss = {gen_loss.numpy()}, Discriminator Loss = {disc_loss.numpy()}')

        # Save generated images at specified intervals
        if (epoch + 1) % save_interval == 0:
            noise = tf.random.normal([1, noise_dim])
            generated_images = generator(noise, training=False)
            save_image(generated_images[0].numpy(), epoch + 1)

# Main script
if __name__ == '__main__':
    generator = build_generator()
    discriminator = build_discriminator()
    
    # Load dataset
    dataset = prepare_dataset(image_folder, batch_size)
    
    # Train GAN
    train_gan(generator, discriminator, dataset, epochs, batch_size)

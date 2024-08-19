import tensorflow as tf
import numpy as np
import os
import random
from PIL import Image

# Parameters
image_size = 256
batch_size = 1
epochs = 50
image_folder = 'Preprocessed-Photos'
num_diffusion_steps = 256  # Number of diffusion steps (T)
beta_start = 0.0001
beta_end = 0.02
buffer_size = 512  # Number of images to pre-load
save_interval = 10  # Save image after this many steps
output_folder = 'Generated-Images'  # Folder to save generated images

# Model architecture parameters
encoder_layers = 5         # Number of layers in the encoder
decoder_layers = 5         # Number of layers in the decoder
initial_filters = 128       # Number of filters in the first layer
filter_growth = 2          # Growth factor for filters in deeper layers (e.g., double the filters with each layer)
dense_units = 64         # Number of units in the dense layer before the decoder
use_batchnorm = True       # Whether to use batch normalization

# Load and preprocess a single image
def load_and_preprocess_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_with_crop_or_pad(image, image_size, image_size)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    return image

# Custom generator function for loading images dynamically
def dynamic_image_loader(file_paths, buffer_size):
    while True:
        random.shuffle(file_paths)  # Shuffle file paths
        for i in range(0, len(file_paths), buffer_size):
            batch_files = file_paths[i:i + buffer_size]
            images = [load_and_preprocess_image(file) for file in batch_files]
            images = tf.stack(images)
            for j in range(0, len(images), batch_size):
                yield images[j:j + batch_size]

# Create dataset
def create_dynamic_dataset(image_folder, buffer_size):
    file_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.endswith('.jpg')]
    dataset = tf.data.Dataset.from_generator(
        lambda: dynamic_image_loader(file_paths, buffer_size),
        output_signature=tf.TensorSpec(shape=(None, image_size, image_size, 3), dtype=tf.float32)
    )
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# Diffusion utilities
def get_beta_schedule(num_diffusion_steps, beta_start, beta_end):
    return np.linspace(beta_start, beta_end, num_diffusion_steps, dtype=np.float32)

def q_sample(x_start, t, noise, alphas_bar_sqrt):
    alphas_bar_sqrt_t = tf.reshape(alphas_bar_sqrt[t], (-1, 1, 1, 1))
    one_minus_alphas_bar_sqrt_t = tf.reshape(1 - alphas_bar_sqrt[t], (-1, 1, 1, 1))
    return alphas_bar_sqrt_t * x_start + one_minus_alphas_bar_sqrt_t * noise

def save_image(image_tensor, step):
    image_array = image_tensor.numpy() * 255.0  # Convert to [0, 255]
    image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    image_pil = Image.fromarray(image_array)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    image_pil.save(os.path.join(output_folder, f'step_{step}.png'))

class DiffusionModel(tf.keras.Model):
    def __init__(self, encoder_layers, decoder_layers, initial_filters, filter_growth, dense_units, use_batchnorm):
        super(DiffusionModel, self).__init__()

        # Encoder
        self.encoder = tf.keras.Sequential()
        filters = initial_filters
        for i in range(encoder_layers):
            self.encoder.add(tf.keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu'))
            if use_batchnorm:
                self.encoder.add(tf.keras.layers.BatchNormalization())
            self.encoder.add(tf.keras.layers.LeakyReLU())
            if i < encoder_layers - 1:
                self.encoder.add(tf.keras.layers.Conv2D(filters, (3, 3), strides=(2, 2), padding='same', activation='relu'))
                filters *= filter_growth

        self.encoder.add(tf.keras.layers.Flatten())
        self.encoder.add(tf.keras.layers.Dense(dense_units, activation='relu'))

        # Decoder
        self.decoder = tf.keras.Sequential()
        self.decoder.add(tf.keras.layers.Dense((image_size // (2 ** (encoder_layers - 1))) * (image_size // (2 ** (encoder_layers - 1))) * filters, activation='relu'))
        self.decoder.add(tf.keras.layers.Reshape(((image_size // (2 ** (encoder_layers - 1))), (image_size // (2 ** (encoder_layers - 1))), filters)))

        for i in range(decoder_layers):
            if i < decoder_layers - 1:
                self.decoder.add(tf.keras.layers.UpSampling2D((2, 2)))
            filters //= filter_growth
            self.decoder.add(tf.keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu'))
            if use_batchnorm:
                self.decoder.add(tf.keras.layers.BatchNormalization())
            self.decoder.add(tf.keras.layers.LeakyReLU())

        self.decoder.add(tf.keras.layers.Conv2D(3, (3, 3), padding='same', activation='sigmoid'))  # 256x256x3 output

    def call(self, inputs):
        x_t, t = inputs
        encoded = self.encoder(x_t)
        decoded = self.decoder(encoded)
        return decoded

# Training
def train_model(model, dataset, epochs, num_diffusion_steps, betas, alphas_bar_sqrt):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.MeanSquaredError()

    step_count = 0  # Initialize step counter
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        for step, x_start in enumerate(dataset):
            batch_size = tf.shape(x_start)[0]
            t = tf.random.uniform((batch_size,), minval=0, maxval=num_diffusion_steps, dtype=tf.int32)
            noise = tf.random.normal(shape=x_start.shape)
            x_t = q_sample(x_start, t, noise, alphas_bar_sqrt)
            
            with tf.GradientTape() as tape:
                x_t_pred = model([x_t, t])
                loss = loss_fn(noise, x_t_pred)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            if step % save_interval == 0:
                print(f'Step {step}, Loss: {loss.numpy()}')
                save_image(x_t_pred[0], step_count)  # Save generated image
            step_count += 1


# Main script
betas = get_beta_schedule(num_diffusion_steps, beta_start, beta_end)
alphas = 1.0 - betas
alphas_bar = np.cumprod(alphas, axis=0)
alphas_bar_sqrt = np.sqrt(alphas_bar)

dataset = create_dynamic_dataset(image_folder, buffer_size)
model = DiffusionModel(encoder_layers, decoder_layers, initial_filters, filter_growth, dense_units, use_batchnorm)
train_model(model, dataset, epochs, num_diffusion_steps, betas, alphas_bar_sqrt)

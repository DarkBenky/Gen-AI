import tensorflow as tf
import numpy as np
import os
from PIL import Image
import random

# Parameters
image_size = 128
batch_size = 32
epochs = 5
image_folder = 'Preprocessed-Photos'
save_interval = 1000
output_folder = 'Generated-Images'
buffer_size = 32  # Number of images to load at a time

# Load and preprocess a single image
def load_and_preprocess_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_with_crop_or_pad(image, image_size, image_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image

# Save image function
def save_image(image_array, step_count):
    image_pil = Image.fromarray((image_array * 255).astype(np.uint8))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    image_pil.save(os.path.join(output_folder, f'swap_{step_count}.png'))

class PixelPredictor(tf.keras.Model):
    def __init__(self, num_layers=3, num_filters=[64, 128, 64], activation='relu', dense_units=512, num_dense_layers=2):
        super(PixelPredictor, self).__init__()
        
        # Initialize convolutional layers
        self.conv_layers = []
        for i in range(num_layers):
            self.conv_layers.append(
                tf.keras.layers.Conv2D(
                    num_filters[i], (3, 3), activation=activation, padding='same'
                )
            )
        
        self.flatten = tf.keras.layers.Flatten()
        
        # Initialize dense layers
        self.dense_layers = []
        for i in range(num_dense_layers):
            self.dense_layers.append(
                tf.keras.layers.Dense(dense_units, activation=activation)
            )
        
        self.output_layer = tf.keras.layers.Dense(3, activation='sigmoid')

    def call(self, inputs):
        x = inputs
        
        # Apply convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        x = self.flatten(x)
        
        # Apply dense layers
        for dense_layer in self.dense_layers:
            x = dense_layer(x)
        
        # Apply the output layer
        return self.output_layer(x)

def predict_pixel_value(image, model):
    image_tensor = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)
    predicted_pixel = model(image_tensor)
    return predicted_pixel.numpy()[0]

# Training function
def train_pixel_predictor(model, image_paths, epochs, optimizer, loss_fn):
    num_images = len(image_paths)
    for epoch in range(epochs):
        print(f'Starting epoch {epoch + 1}/{epochs}')
        random.shuffle(image_paths)  # Shuffle images for each epoch
        
        # Batch processing
        for start in range(0, num_images, batch_size):
            end = min(start + batch_size, num_images)
            batch_paths = image_paths[start:end]
            images, targets = load_batch(batch_paths)

            images = tf.convert_to_tensor(images, dtype=tf.float32)
            targets = tf.convert_to_tensor(targets, dtype=tf.float32)

            with tf.GradientTape() as tape:
                predictions = model(images)
                # print('Predictions:', predictions)
                # print('Targets:', targets)
                loss = loss_fn(targets, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            print(f'Step {start // batch_size}: Loss = {loss.numpy()}')

def load_batch(file_paths):
    images = []
    targets = []
    for file_path in file_paths:
        image = load_and_preprocess_image(file_path)
        height, width, _ = image.shape
        i, j = random.randint(0, height - 1), random.randint(0, width - 1)
        target_pixel = image[i, j]
        image = tf.tensor_scatter_nd_update(image, [[i, j]], [[0.0, 0.0, 0.0]])
        images.append(image)
        targets.append(target_pixel)
    return np.array(images), np.array(targets)

def swap_and_fill_pixels(image, model):
    height, width, _ = image.shape
    all_pixels = [(i, j) for i in range(height) for j in range(width)]
    random.shuffle(all_pixels)

    step_count = 0
    for i, j in all_pixels:
        # Temporarily remove pixel (set it to black or another placeholder)
        indices = tf.constant([[i, j]], dtype=tf.int32)
        new_image = tf.tensor_scatter_nd_update(image, indices, [[0.0, 0.0, 0.0]])

        # Predict and fill in the pixel's new value
        new_pixel_value = predict_pixel_value(new_image, model)
        new_image = tf.tensor_scatter_nd_update(new_image, indices, [new_pixel_value])

        # Update the image with the new pixel value
        image = new_image

        step_count += 1
        if step_count % save_interval == 0:
            save_image(image.numpy(), step_count)

    # Save the final image
    save_image(image.numpy(), step_count)


# Main script
if __name__ == '__main__':
    model = PixelPredictor(num_layers=5 , num_filters=[64, 128, 256, 128, 64], activation='relu', dense_units=512, num_dense_layers=3)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    loss_fn = tf.keras.losses.MeanSquaredError()

    file_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.endswith('.jpg')]

    while True:
        # Training loop
        train_pixel_predictor(model, file_paths, epochs, optimizer, loss_fn)

        # Generate images by swapping and filling pixels
        image_file = random.choice(file_paths)
        if image_file.endswith('.jpg'):
            image = load_and_preprocess_image(image_file)
            swap_and_fill_pixels(image, model)

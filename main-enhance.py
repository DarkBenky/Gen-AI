import tensorflow as tf
import numpy as np
import os
from PIL import Image
import random
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

# Parameters
image_size = 128
batch_size = 8  # Reduced batch size
epochs = 5
image_folder = 'Preprocessed-Photos'
save_interval = 1000
output_folder = 'Generated-Images'
buffer_size = 32  # Number of images to load at a time

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load and preprocess a single image with data augmentation
def load_and_preprocess_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_with_crop_or_pad(image, image_size, image_size)
    
    # Data Augmentation
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    
    image = tf.cast(image, tf.float32) / 255.0
    return image

# Save image function
def save_image(image_array, step_count):
    image_pil = Image.fromarray((image_array * 255).astype(np.uint8))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    image_pil.save(os.path.join(output_folder, f'swap_{step_count}.png'))

# Perceptual Loss function
def perceptual_loss(y_true, y_pred):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=(image_size, image_size, 3))
    feature_extractor = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    feature_extractor.trainable = False

    # Compute feature maps for the true and predicted images
    y_true_features = feature_extractor(y_true)
    y_pred_features = feature_extractor(y_pred)

    # Compute the perceptual loss (feature loss) and pixel-wise loss
    perceptual_loss = tf.reduce_mean(tf.square(y_true_features - y_pred_features))
    pixel_loss = tf.reduce_mean(tf.square(y_true - y_pred))

    return pixel_loss + 0.1 * perceptual_loss

# Enhanced Pixel Predictor model with residual connections and batch normalization
class EnhancedPixelPredictor(tf.keras.Model):
    def __init__(self, num_layers=3, num_filters=[32, 64, 128], activation='relu', dense_units=256, num_dense_layers=2):
        super(EnhancedPixelPredictor, self).__init__()

        self.conv_blocks = []
        self.residual_blocks = []
        for i in range(num_layers):
            filters = num_filters[i]
            
            # Convolutional Block
            conv_block = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters, (3, 3), padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(activation),
                tf.keras.layers.Conv2D(filters, (3, 3), padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(activation),
            ])
            self.conv_blocks.append(conv_block)
            
            # Residual Block (to add the input back into the output)
            self.residual_blocks.append(
                tf.keras.layers.Conv2D(filters, (1, 1), padding='same')
            )
        
        self.flatten = tf.keras.layers.Flatten()

        # Dense layers with dropout
        self.dense_layers = []
        for i in range(num_dense_layers):
            self.dense_layers.append(
                tf.keras.layers.Dense(dense_units, activation=activation)
            )
            self.dense_layers.append(tf.keras.layers.Dropout(0.5))

        self.output_layer = tf.keras.layers.Dense(image_size * image_size * 3, activation='sigmoid')

    def call(self, inputs):
        x = inputs

        # Apply convolutional and residual blocks
        for conv_block, residual_block in zip(self.conv_blocks, self.residual_blocks):
            residual = residual_block(x)
            x = conv_block(x) + residual

        x = self.flatten(x)

        # Apply dense layers
        for dense_layer in self.dense_layers:
            x = dense_layer(x)

        # Apply the output layer and reshape
        x = self.output_layer(x)
        x = tf.reshape(x, (-1, image_size, image_size, 3))  # Reshape to match image dimensions

        return x

# Function to predict pixel values
def predict_pixel_value(image, model):
    image_tensor = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)
    predicted_image = model(image_tensor)
    return predicted_image.numpy()[0]

# Function to load a batch of images
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
                # Ensure that predictions have the correct shape
                predictions = tf.clip_by_value(predictions, 0.0, 1.0)  # Clip to valid range for images
                loss = loss_fn(images, predictions)  # Compare to the original images
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            print(f'Step {start // batch_size}: Loss = {loss.numpy()}')

# Function to fill in pixels progressively
def progressive_fill_pixels(image, model):
    height, width, _ = image.shape
    center_i, center_j = height // 2, width // 2

    pixels_to_fill = [(center_i, center_j)]
    filled_pixels = set(pixels_to_fill)

    step_count = 0
    while pixels_to_fill:
        i, j = pixels_to_fill.pop(0)

        # Temporarily remove the pixel
        indices = tf.constant([[i, j]], dtype=tf.int32)
        new_image = tf.tensor_scatter_nd_update(image, indices, [[0.0, 0.0, 0.0]])

        # Predict and fill the pixel's value
        new_pixel_value = predict_pixel_value(new_image, model)
        new_image = tf.tensor_scatter_nd_update(new_image, indices, [new_pixel_value])

        # Update the image
        image = new_image

        # Add neighboring pixels to the list to be filled next
        neighbors = [(i + di, j + dj) for di in [-1, 0, 1] for dj in [-1, 0, 1]]
        for ni, nj in neighbors:
            if 0 <= ni < height and 0 <= nj < width and (ni, nj) not in filled_pixels:
                pixels_to_fill.append((ni, nj))
                filled_pixels.add((ni, nj))

        step_count += 1
        if step_count % save_interval == 0:
            save_image(image.numpy(), step_count)

    # Save the final image
    save_image(image.numpy(), step_count)

# Main script
if __name__ == '__main__':
    model = EnhancedPixelPredictor(num_layers=2, num_filters=[32, 64], activation='relu', dense_units=256, num_dense_layers=2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    loss_fn = perceptual_loss

    file_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.endswith('.jpg')]

    while True:
        # Training loop
        train_pixel_predictor(model, file_paths, epochs, optimizer, loss_fn)

        # Generate images by filling pixels progressively
        image_file = random.choice(file_paths)
        if image_file.endswith('.jpg'):
            image = load_and_preprocess_image(image_file)
            progressive_fill_pixels(image, model)

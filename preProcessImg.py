import tensorflow as tf
import os
import shutil

# Parameters
image_size = 256
source_folder = 'Photos-001'
preprocessed_folder = 'Preprocessed-Photos'

# Recreate the target directory
if os.path.exists(preprocessed_folder):
    shutil.rmtree(preprocessed_folder)  # Remove the entire directory and its contents
os.makedirs(preprocessed_folder)  # Create a new, empty directory

def preprocess_and_save_image(file_path, target_folder):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)

    # Resize image while maintaining aspect ratio
    image = tf.image.resize(image, [image_size, image_size], preserve_aspect_ratio=True)

    # Crop the center of the image to the target size
    image = tf.image.resize_with_crop_or_pad(image, image_size, image_size)

    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]

    # Save preprocessed image
    image_name = os.path.basename(file_path)
    target_path = os.path.join(target_folder, image_name)
    image_encoded = tf.image.encode_jpeg(tf.image.convert_image_dtype(image, tf.uint8))
    tf.io.write_file(target_path, image_encoded)

# Process all images in the source folder
file_paths = [os.path.join(source_folder, fname) for fname in os.listdir(source_folder) if fname.endswith('.jpg')]
for file_path in file_paths:
    try:
        preprocess_and_save_image(file_path, preprocessed_folder)
    except Exception as e:
        print(f"Error processing image {file_path}: {e}")

print(f"Preprocessing complete. Images saved to {preprocessed_folder}")
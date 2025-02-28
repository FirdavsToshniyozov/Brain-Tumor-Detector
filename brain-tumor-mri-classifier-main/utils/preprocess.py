import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle

from setup import downloader, unziper, extractor


def setup_dataset(dataset_url, downloads_path):
    # Download dataset
    print("[1] Downloading dataset...")
    downloader.process([dataset_url], downloads_path)

    # Unzip dataset
    print("[2] Unzipping dataset...")
    unziper.process(downloads_path)

    print("Dataset downloaded and unzipped successfully.")


def extract_contour(image):
    return extractor.extract_contour(image, True)


def extract_images(source_path, dataset_path):
    # Extract images
    print("Extracting images...")
    extractor.process(source_path, dataset_path)

    print("Images cropped successfully.")


def preprocess_data(source_path, destination_path, img_size=256):
    extractor.clear_screen()

    if os.path.exists(destination_path):
        shutil.rmtree(destination_path)

    extractor.make_folder(destination_path)

    for root, dirs, files in os.walk(source_path):
        sub_dir = os.path.relpath(root, source_path)
        extractor.make_folder(os.path.join(destination_path, sub_dir))

        # extract each file and update the progress bar
        if files:
            progress_bar = tqdm(total=len(files), desc=f'Processing {sub_dir} images', dynamic_ncols=True)
            for file in files:
                if not file.endswith('.jpg'):
                    continue

                # Read the original image
                original_path = os.path.join(root, file)
                img = cv2.imread(original_path)
                processed_img = preprocess_image(img, img_size)

                # Save the optimized image
                processed_path = os.path.join(destination_path, sub_dir, file)
                cv2.imwrite(processed_path, processed_img)
                progress_bar.update(1)

            progress_bar.close()


def preprocess_image(image, size=256):
    # Resize image
    image = cv2.resize(image, dsize=(size, size), interpolation=cv2.INTER_CUBIC)

    # Convert the image to grayscale
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize the image
    image = image.astype(np.float32)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    # remove images noise.
    image = cv2.bilateralFilter(image, 2, 50, 50)

    image = (image * 255).astype(np.uint8)

    return image


def save_augmented_images(data, destination_path, labels):
    extractor.clear_screen()

    if os.path.exists(destination_path):
        shutil.rmtree(destination_path)

    extractor.make_folder(destination_path)

    num_batches = len(data[0])
    progress_bar = tqdm(total=num_batches, desc=f'Saving augmented images', dynamic_ncols=True)

    # Iterate over the generated batches
    for i, (batch_x, batch_y) in enumerate(zip(*data)):
        # Get the label of the image
        label = batch_y

        label_name = labels[label]
        label_dir = os.path.join(destination_path, label_name)

        # Create a directory for the label if it doesn't exist
        extractor.make_folder(label_dir)

        #batch_x = batch_x.reshape((batch_x.shape[0], 256, 256))


        # Iterate over the images in the batch
        for j in range(batch_x.shape[0]):
            #img = np.expand_dims(batch_x[j], axis=-1)
            #img = Image.fromarray(img.astype('uint8'), 'L')

            image_path = os.path.join(label_dir, f'Tr-{label_name[:2]}_{i * batch_x.shape[0] + j}.jpg')
            cv2.imwrite(image_path, batch_x[j])

        progress_bar.update(1)

        # Exit the loop if all batches have been processed
        #if i == num_batches - 1:
         #   break

    progress_bar.close()


def load_data(dataset_path, labels):
    """ Load each label dataset into list.
    Parameters:
        dataset_path(str): Name of the path for dataset.
        labels(str): Name of the path for the label dataset.
    Returns: 2 lists of data & labels
    """

    X = []
    y = []
    for label in labels:
        label_data = os.path.join(dataset_path, label)
        for filename in os.listdir(label_data):
            file_path = os.path.join(label_data, filename)
            image = cv2.imread(file_path)
            #image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            X.append(image)
            y.append(labels.index(label))

    X = np.array(X)
    y = np.array(y)

    # Shuffle the data
    X, y = shuffle(X, y)

    return X, y

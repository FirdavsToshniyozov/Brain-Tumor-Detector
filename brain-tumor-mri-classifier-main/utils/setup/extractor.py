import os
import cv2
import imutils
from tqdm import tqdm
import matplotlib.pyplot as plt


def clear_screen():
    """Clears the console screen irrespective of os used"""
    import platform
    if platform.system() == 'Windows':
        os.system('cls')
        return
    os.system('clear')


def make_folder(target_folder):
    """Creates folder if there is no folder in the specified path.
    Parameters:
        target_folder(str): path of the folder which needs to be created.

    Returns: None
    """
    if not (os.path.isdir(target_folder)):
        # print(f'Creating {target_folder} folder')
        os.mkdir(target_folder)


def extract_contour(image, plot=False):
    """ Finds the extreme points on the image and crops the rectangular out of them
    """

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur it slightly to reduce the noise and smooth the edges.
    gray_blured = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply binary threshold
    thresh = cv2.threshold(gray_blured, 45, 255, cv2.THRESH_BINARY)[1]

    # Perform a series of erosion
    thresh = cv2.erode(thresh, None, iterations=2)

    # Dilate images to remove any small regions of noise
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in threshold image
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Grab contours
    contours = imutils.grab_contours(contours)

    # Grab the largest one
    c = max(contours, key=cv2.contourArea)

    # Find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    ADD_PIXELS = 0
    new_image = image[extTop[1] - ADD_PIXELS:extBot[1] + ADD_PIXELS,
                extLeft[0] - ADD_PIXELS:extRight[0] + ADD_PIXELS].copy()

    # Plot
    if plot:
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.tick_params(axis='both', which='both', top=False, bottom=False, left=False, right=False, labelbottom=False,
                        labeltop=False, labelleft=False, labelright=False)
        plt.title('Original Image')
        plt.subplot(1, 2, 2)
        plt.imshow(new_image)
        plt.tick_params(axis='both', which='both', top=False, bottom=False, left=False, right=False, labelbottom=False,
                        labeltop=False, labelleft=False, labelright=False)
        plt.title('Cropped Image')
        plt.show()

    return new_image


def crop_images(source_path, destination_path):
    """ Read images, crop, save them.
    Parameters:
        source_path(str): Name of the path for the original dataset.
        destination_path(str): Path where the filename is to be saved.
    Returns: None
    """

    for root, dirs, files in os.walk(source_path):
        # Get the subdirectory path relative to the original directory
        sub_dir = os.path.relpath(root, source_path)
        # Create the subdirectory in the optimized directory if it doesn't exist
        make_folder(os.path.join(destination_path, sub_dir))

        # extract each file and update the progress bar
        if len(files) > 1:
            # print(f'Processing {destination_path}/{sub_dir} images')

            # create progress bar with total number of files to optimize
            progress_bar = tqdm(total=len(files), desc=f'Processing {sub_dir} images', dynamic_ncols=True)

            for file in files:
                # Only process JPG files
                if not file.endswith('.jpg'):
                    continue

                # Read the original image
                original_path = os.path.join(root, file)
                img = cv2.imread(original_path)

                # Crop the image
                cropped_img = extract_contour(img)

                # Save the optimized image
                optimized_path = os.path.join(destination_path, sub_dir, file)
                cv2.imwrite(optimized_path, cropped_img)

                progress_bar.update(1)

            progress_bar.close()


def process(source_path, destination_path):
    # Clears screen.
    clear_screen()

    # Make destination dir if not exists
    make_folder(destination_path)

    # Process images
    crop_images(source_path, destination_path)

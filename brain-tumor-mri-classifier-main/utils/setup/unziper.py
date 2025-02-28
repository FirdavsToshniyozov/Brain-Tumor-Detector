from zipfile import ZipFile
import os
from tqdm import tqdm


def clear_screen():
    """Clears the console screen irrespective of os used"""
    import platform
    if platform.system() == 'Windows':
        os.system('cls')
        return
    os.system('clear')


def unzip_file(source_name, destination):
    """ Unzips a zip file and stores the contents in destination folder.
    Parameters:
        source_name(str): Full path of the source path
        destination(str): Full folder path where contents of source_name will be stored.

    Returns: None
    """
    with ZipFile(source_name, 'r') as zipfile:
        # extracting all the files
        print(f'\tExtracting files of {source_name}')

        # get list of all files to extract
        file_list = zipfile.namelist()
        # create progress bar with total number of files to extract
        progress_bar = tqdm(total=len(file_list), desc='Extracting Images', dynamic_ncols=True)

        # extract each file and update the progress bar
        for file in file_list:
            zipfile.extract(file, destination)
            progress_bar.update(1)
        progress_bar.close()

        print(f'\tDone with {source_name}')


def make_folder(target_folder):
    """Creates folder if there is no folder in the specified path.
    Parameters:
        target_folder(str): path of the folder which needs to be created.

    Returns: None
    """
    if not (os.path.isdir(target_folder)):
        # print(f'Creating {target_folder} folder')
        os.mkdir(target_folder)


def process(downloads_path):
    # Clears the screen.
    clear_screen()

    # Get a list of all the zip files in the folder
    zipped_downloads = [f for f in os.listdir(downloads_path) if f.endswith('.zip')]

    # Unzip each file in the list
    for file in zipped_downloads:
        # Remove the ".zip" extension from the filename to get the dataset name
        file_name = os.path.splitext(file)[0]

        # Make destination directory
        destination = os.path.join(downloads_path, file_name)
        make_folder(destination)

        # Unzip the file to the folder
        file_path = os.path.join(downloads_path, file)
        unzip_file(file_path, destination)

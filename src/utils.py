#!usr/bin/env python
"""
Created Dec 3 2022

By Sharif Saleki

Utility functions to mostly read and write things
"""

from pathlib import Path
import json
import zipfile
from typing import Union

import cv2

import pandas as pd


def load_params(params_file: str):
    """
    Loads parameters for all modules of the experiment including the task, CNN, RNN

    Parameters
    ----------
    params_file : str
        Path to the parameters file

    Returns
    -------
    params : dict
        Containing Task, CNN, RNN, Experiment keys
    """
    with open(file=str(params_file), mode='r') as f:
        params = json.load(f)

    return params


def extract_zipfile(zip_location: Union[Path, str], extracted_location: Union[Path, str]):
    """
    Unzips the Contents of File present in zip_location onto the folder extracted_location

    Parameters
    ----------
    zip_location : str
        Path to the zipfile
    extracted_location : str
        Path to the extracted files
    """
    with zipfile.ZipFile(zip_location, 'r') as reference:
        reference.extractall(extracted_location)


def resize_image(image_path: Union[Path, str], new_size: tuple, output_path):
    """
    Resizes the image present in image_path to new_size

    Parameters
    ----------
    image_path : str
        Path to the image

    new_size : tuple
        New size of the image

    output_path : str
        Path to the output image
    """
    # Reading the File as an Image
    color_image = cv2.imread(str(image_path))

    # Converting the Color Image to Grayscale Image
    grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # Resizing the image
    resized_image = cv2.resize(grayscale_image, new_size)

    # Saving if output_path is provided
    if output_path:
        cv2.imwrite(str(output_path), resized_image)

    return resized_image


def make_shape_dataset(images_dir: Union[Path, str], save_dir: Union[Path, str]):
    """
    Makes a dataset of shapes from the images present in images_dir

    Parameters
    ----------
    images_dir : str or Path
        Path to the images directory

    save_dir : str or Path
        Path to the directory where the dataset will be saved
    """
    file_names = []
    classes = []

    if isinstance(images_dir, str):
        images_dir = Path(images_dir)

    for file in images_dir.iterdir():
        file_name = file.name
        file_names.append(file_name)

        if "Circle" in file_name:
            classes.append(0)
        elif "Triangle" in file_name:
            classes.append(1)
        elif "Square" in file_name:
            classes.append(2)
        elif "Octagon" in file_name:
            classes.append(3)
        elif "Nonagon" in file_name:
            classes.append(4)
        elif "Star" in file_name:
            classes.append(5)
        elif "Hexagon" in file_name:
            classes.append(6)
        elif "Pentagon" in file_name:
            classes.append(7)
        elif "Heptagon" in file_name:
            classes.append(8)
        else:
            print("Unknown Class: ", file_name.split("_")[0])

    # Creating a DataFrame
    df = pd.DataFrame(data={"FILE_NAME": file_names, "LABEL": classes})

    # Saving the DataFrame
    if save_dir:
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)

    save_file = save_dir / "shapes_dataset.csv"
    df.to_csv(save_file, index=False)

    return df


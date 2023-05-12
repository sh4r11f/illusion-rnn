#!/usr/bin/env python3
"""
Script to make TAM datasets and save to file
"""
from pathlib import Path
from src.utils import load_params
from src.train_test_models import generate_env_dataset
import gc

# Directories
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
DATASETS_DIR = DATA_DIR / "datasets"
STIMULUS_DIR = DATA_DIR / "stimuli"

# Load parameters
# params = load_params(params_file=str(ROOT_DIR / "parameters.json"))["TASK"]
parameters = [
    {
        "dt": 50,
        "sigma": 0,
        "sequence_length": 10,
        "batch_size": 64,
        "img_size": 64
    },
    {
        "dt": 50,
        "sigma": 0,
        "sequence_length": 100,
        "batch_size": 16,
        "img_size": 64
    },
    {
        "dt": 50,
        "sigma": 0.05,
        "sequence_length": 10,
        "batch_size": 64,
        "img_size": 256
    },
    {
        "dt": 50,
        "sigma": 0.05,
        "sequence_length": 100,
        "batch_size": 16,
        "img_size": 256
    },

]

# Variables
box_shapes = ["square", "circle", "triangle"]
stimulus_orientations = ["vertical", "horizontal"]
motion_types = ["TAM", "TAM_basic", "TAM_outline", "cnt", "track"]
# motion_types = ["TAM_outline"]


if __name__ == '__main__':

    # Generate datasets
    for p, params in enumerate(parameters):
        for bs in box_shapes:
            for so in stimulus_orientations:
                for mt in motion_types:

                    print(f"Generating dataset: parameter set {p + 1}/{len(parameters)} | {mt} task | {bs} shape | {so} orientation.")
                    if mt == "TAM":
                        stim_dir = STIMULUS_DIR / "TAM_task"
                    elif mt == "TAM_basic":
                        stim_dir = STIMULUS_DIR / "TAM_basic"
                    elif mt == "TAM_outline":
                        stim_dir = STIMULUS_DIR / "TAM_outline"
                    else:
                        stim_dir = STIMULUS_DIR / "motion_task"

                    e, d = generate_env_dataset(
                        box_shape=bs,
                        motion_type=mt,
                        stimulus_orientation=so,
                        stimulus_directory=stim_dir,
                        parameters=params,
                        dataset_directory=DATASETS_DIR,
                        save=True
                    )
                    print("Done \n ------------------ \n")

                    del e
                    del d
                    gc.collect()

#!/usr/bin/env python3
"""
Train models on TAM datasets and save to file

Usage:
    train_models.py [options]

"""
import pickle
from pathlib import Path
import gc

from src.utils import load_params
from src.train_test_utils import make_network, train_network

# Directories
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
DATASETS_DIR = DATA_DIR / "datasets"
MODELS_DIR = DATA_DIR / "models"

# Load parameters
params = load_params(params_file=str(ROOT_DIR / "parameters.json"))
task_params = params["TASK"]
model_params = params["NETWORK"]

# Select datasets with these parameters
stim_params = f"img-size-{task_params['img_size']}_batch-size-{task_params['batch_size']}_noise-{task_params['sigma']}_seq-len-{task_params['sequence_length']}"

# Variables
train_datasets = [
    "TAM_basic",
    "TAM",
    "TAM_outline",
    "cnt",
    "track",
]
directions = ["horizontal", "vertical", "all"]

# Model
for d, datasets_name in enumerate(train_datasets):
    for direction in directions:

        print(f"Training model on {d + 1}/{len(train_datasets) * len(directions)} datasets: {direction} {datasets_name}")

        # Directory
        datasets_dir = DATASETS_DIR / datasets_name / stim_params
        datasets = []

        # Load all datasets
        if direction in ["horizontal", "vertical"]:
            files = [file for file in datasets_dir.iterdir() if direction in str(file)]
        else:
            files = [file for file in datasets_dir.iterdir()]
        for dataset in files:
            with open(dataset, "rb") as f:
                datasets.append(pickle.load(f))

        # Make network
        model = make_network(params=model_params, dataset=datasets[0])

        # Train network
        _, _, convergence = train_network(
            network=model,
            datasets=datasets,
            dataset_names=[datasets_name],
            params=model_params,
            motion_direction=direction,
            save_dir=MODELS_DIR / stim_params
        )

        print(f"Done. Convergence status: {convergence} \n ----------------------------------------------------------- \n")

        # Clear memory
        del datasets
        del model
        gc.collect()

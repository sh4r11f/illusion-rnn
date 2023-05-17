#!/usr/bin/env python3
"""
Script to train various basic RNNs on the TAM dataset.

@author: Sharif Saleki

Usage:
    train_basic_rnns.py [options]

Options:
    -h --help                   Show this screen.
    --dataset=<str>             Dataset to use. [default: tam]
    --model=<str>               Model to use. [default: lstm]
    --batch_size=<int>          Batch size. [default: 32]
    --epochs=<int>              Number of epochs. [default: 100]
    --lr=<float>                Learning rate. [default: 0.001]
    --hidden_size=<int>         Hidden size. [default: 128]
    --num_layers=<int>          Number of layers. [default: 1]
    --dropout=<float>           Dropout. [default: 0.0]
    --bidirectional             Bidirectional. [default: False]
    --cuda                      Use cuda.
    --seed=<int>                Random seed. [default: 42]
    --log_interval=<int>        Log interval. [default: 100]
    --save_model                Save model.
    --save_dir=<str>            Save directory. [default: saved_models]
    --save_prefix=<str>         Save prefix. [default: basic_rnn]
"""
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import torch

# Custom imports
from src.environments import TAMTask, MotionTask, load_motion_from_image, load_tam_from_image
from src.networks import TAMNet

import neurogym as ngym


def generate_env_dataset(
        box_shape: str,
        motion_type: str,
        stimulus_orientation: str,
        stimulus_directory: Path,
        parameters: dict,
        dataset_directory: Path = None,
        save: bool = False,
):
    """
    Generate environments and datasets for training.

    Returns
    -------

    """
    dt = parameters["dt"]
    noise = parameters["sigma"]
    img_size = parameters["img_size"]
    batch_size = parameters["batch_size"]
    seq_len = parameters["sequence_length"]

    # Create environment
    if motion_type in ["cnt", "track"]:
        stimuli = load_motion_from_image(img_size, stimulus_directory)
        env = MotionTask(
            dt=dt,
            box_shape=box_shape,
            motion_type=motion_type,
            stim_ori=stimulus_orientation,
            stimuli=stimuli,
            sigma=noise,
            img_size=img_size
        )
    elif motion_type in ["TAM", "TAM_basic", "TAM_outline"]:
        stim_type = "tam" if motion_type != "TAM_outline" else "outline"
        stimuli = load_tam_from_image(img_size, stim_type, stimulus_directory)
        env = TAMTask(
            dt=dt,
            box_shape=box_shape,
            stim_type=stim_type,
            stimuli=stimuli,
            stim_ori=stimulus_orientation,
            sigma=noise,
            img_size=img_size
        )
    else:
        raise ValueError("Invalid motion type.")

    # Generate dataset object
    env.reset(no_step=True)
    dataset = ngym.Dataset(
        env,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    # Save dataset
    if save:
        if dataset_directory is not None:

            folder_name = f'img-size-{img_size}_batch-size-{batch_size}_noise-{noise}_seq-len-{seq_len}'
            data_dir = dataset_directory / motion_type / folder_name
            data_dir.mkdir(parents=True, exist_ok=True)

            stim_name = f'{motion_type}_{box_shape}_{stimulus_orientation}.pkl'

            with open(data_dir / stim_name, 'wb') as f:
                pickle.dump(dataset, f)

    return env, dataset


def make_network(params: dict, dataset: ngym.Dataset):
    """
    Make network.

    Parameters
    ----------
    params: dict

    dataset: ngym.Dataset

    Returns
    -------

    """
    # Create network to train on the normal stimuli
    network = TAMNet(
        dataset=dataset,
        hidden_size=params["hidden_size"],
        output_size=dataset.env.action_space.n,
        learning_rate=params["learning_rate"],
        device=torch.device(params["device"])
    )

    return network


def train_network(network, datasets, dataset_names, params, motion_direction, save_dir):
    """
    Train network.

    Parameters
    ----------
    network: TAMNet
        Network to train.

    datasets: list
        List of datasets.

    dataset_names: list
        List of dataset names.

    params: dict
        Dictionary of parameters.

    motion_direction: str
        Motion direction.

    save_dir: Path
        Directory to save the network.

    Returns
    -------

    """
    # Train network
    train_loss, train_acc = network.train_rnn(
        datasets=datasets,
        n_epochs=params["n_epochs"]
    )
    if train_loss[-1] < 0.005:
        converged = True
    else:
        converged = False

    # Save network
    if save_dir is not None:
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        filename = f'basic-RNN_units-{params["hidden_size"]}_lr-{params["learning_rate"]}_dataset-{"-".join(dataset_names)}_direction-{motion_direction}_convergence-{converged}'
        torch.save(network.RNN.state_dict(), save_dir / f"{filename}.pt")

        # Save loss and accuracy
        df = pd.DataFrame({
            "Epoch": np.arange(len(train_loss)) + 1,
            "Model": [f'{"-".join(dataset_names)}_{motion_direction}'] * len(train_loss),
            "Loss": train_loss,
            "Accuracy": train_acc,
            "Converged": [converged] * len(train_loss)
        })

        df.to_csv(save_dir / f'{filename}.csv')

    return train_loss, train_acc, converged


def test_network(network, network_name, dataset, dataset_name, n_iter, save_dir):
    """
    Test network.

    Parameters
    ----------
    network: TAMNet
        Network to test.

    dataset: ngym.Dataset
        Dataset to test on.

    dataset_name: str
        Name of the dataset.

    n_iter: int
        Number of iterations to test.

    save_dir: Path
        Directory to save the network.

    Returns
    -------

    """
    # Freeze network
    network.RNN.eval()

    # Test network
    info, activity, accuracy = network.test_rnn(n_iter, dataset)
    print(f'Accuracy on {dataset_name} = {accuracy}')

    if save_dir:
        filename = f'activity_{network_name}_{dataset_name}_accuracy-{accuracy}.pkl'
        with open(save_dir / filename, 'wb') as f:
            pickle.dump(activity, f)

    return info, activity, accuracy


def plot_dataset(n_trials, dataset, save_dir=None):
    """
    Plot dataset.

    Parameters
    ----------
    n_trials: int
        Number of trials to plot.

    dataset: ngym.Dataset
        Dataset to plot.

    save_dir: Path
        Directory to save the plots.

    Returns
    -------
    None
    """
    env = dataset.env
    n_times = env.task_len

    # Figure
    ax_size = 12
    fig, axs = plt.subplots(n_trials, n_times, figsize=(n_times * ax_size, n_trials * ax_size))

    # Add space between subplots
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    # Title
    # try:
    #     fig.suptitle(f'{env.box_shape} {env.motion_type} {env.stim_ori}', fontsize=26)
    # except AttributeError:
    #     fig.suptitle(f'{env.box_shape} {env.stim_type} {env.stim_ori}', fontsize=26)

    # Generate data
    images, labels = dataset()

    # Convert labels to choice names
    labels = labels.astype(object)
    for choice_name, choice in env.choice_names.items():
        # print(choice_name, choice)
        labels[labels == choice] = choice_name

    # Plot
    for trial in range(n_trials):
        for tp in range(n_times):
            axs[trial, tp].imshow(images[tp, trial, :, :], cmap='gray')
            axs[trial, tp].axis('off')
            axs[trial, tp].set_title(f'Time in trial: {tp * env.dt}', fontsize=50)

    if save_dir:
        try:
            filename = f'sample-trials_motion_square.jpg'
        except AttributeError:
            filename = f'sample-trials_motion_square.jpg'
        plt.savefig(save_dir / filename, dpi=300)

    plt.show()


def plot_training_curves(results, save_dir):
    """
    Plot training curve.

    Parameters
    ----------
    results: dict
        Dictionary of results with model name as key and accuracy and loss and nested keys.

    save_dir: Path
        Directory to save the plots.

    Returns
    -------
    None
    """
    # Turn into dataframe
    df = pd.DataFrame.from_dict(
        {(i, j): results[i][j] for i in results.keys() for j in results[i].keys()})

    # Stack the DataFrame to collapse the MultiIndex columns into a single index
    df_stacked = df.stack(level=0).reset_index()

    # Rename the columns
    df_stacked.columns = ['Epoch', 'Model', 'Accuracy', 'Loss']

    # Plot
    sns.set_theme(style="darkgrid")

    fig, axs = plt.subplots(figsize=(30, 15))
    fig.suptitle('Training Curves', fontsize=26)

    plot = sns.lineplot(
        data=df_stacked,
        x='Epoch',
        y='Accuracy',
        hue='Model',
        style='Model',
        markers=True,
        dashes=False,
        ax=axs
    )

    if save_dir:
        filename = f'training-curve_models-{"-".join(list(results.keys()))}.png'
        plt.savefig(save_dir / filename, dpi=300)


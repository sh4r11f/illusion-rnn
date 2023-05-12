#!usr/bin/env python
"""
Created Dec 3 2022

By Sharif Saleki

Utility functions to mostly read and write things
"""

from pathlib import Path
import json


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

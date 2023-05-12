#!usr/bin/env python
"""
Created Nov 28 2022

By Sharif Saleki

Recurrent neural network class and functions for performing a Transformational Apparent Motion (TAM) task.

"""
# System imports
from typing import Union
import time
from collections import defaultdict

# Scientific import
import numpy as np
from sklearn.metrics import accuracy_score

# Plotting imports
import matplotlib.pyplot as plt
import seaborn as sns

# Neural network imports
import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
import neurogym as ngym


class CTRNN(nn.Module):
    """
    Defines a continuous-time RNN.

    Inputs:
        input: tensor of shape (seq_len, batch, input_size)
        hidden: tensor of shape (batch, hidden_size), initial hidden activity
                if None, hidden is initialized through self.init_hidden()

    Outputs:
        output: tensor of shape (seq_len, batch, hidden_size)
        hidden: tensor of shape (batch, hidden_size), final hidden activity

    Parameters
    ----------
        input_size : int
            Number of input neurons

        hidden_size: int
            Number of hidden neurons

        dt: float
            Time step in ms. If None, dt equals time constant tau.

    """
    def __init__(self, input_size: int, hidden_size: int, dt: float = None, **kwargs):

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.tau = 100

        # Learning rate
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        self.alpha = alpha

        # Define linear hidden layers
        self.input2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def init_hidden(self, input_shape: torch.tensor):
        """
        Initializes the hidden layer activity

        Parameters
        ----------
        input_shape : torch.tensor (seq_len, batch, input_size)

        Returns
        -------
        torch.tensor
            All zeros (batch_size, hidden_size)
        """
        batch_size = input_shape[1]

        return torch.zeros(batch_size, self.hidden_size)

    def recurrence(self, act_input: torch.tensor, hidden: torch.tensor):
        """
        Run network for one time step.

        Parameters
        ----------
        act_input : tensor of shape (batch, input_size)

        hidden : tensor of shape (batch, hidden_size)

        Returns
        -------
        h_new : tensor of shape (batch, hidden_size),
            Network activity at the next time step
        """
        # ReLu layer
        h_new = torch.relu(self.input2h(act_input) + self.h2h(hidden))
        h_new = hidden * (1 - self.alpha) + h_new * self.alpha

        return h_new

    def forward(self, act_input: torch.tensor, hidden: torch.tensor = None):
        """
        Propagate input through the network.

        Parameters
        ----------
        act_input : tensor of shape (batch, input_size)

        hidden : tensor of shape (batch, hidden_size)

        Returns
        -------
        tuple (output : torch.tensor, hidden : torch.tensor)
            output: tensor of shape (seq_len, batch, hidden_size)
            hidden: tensor of shape (batch, hidden_size), final hidden activity
        """
        # If hidden activity is not provided, initialize it
        if hidden is None:
            hidden = self.init_hidden(act_input.shape).to(act_input.device)

        # Loop through time
        output = []
        steps = range(act_input.size(0))
        for i in steps:
            hidden = self.recurrence(act_input[i], hidden)
            output.append(hidden)

        # Stack together output from all time steps
        output = torch.stack(output, dim=0)  # (seq_len, batch, hidden_size)

        return output, hidden


class RNNNet(nn.Module):
    """
    Recurrent network model.

    Inputs:
        x: tensor of shape (Seq Len, Batch, Input size)

    Outputs:
        out: tensor of shape (Seq Len, Batch, Output size)
        rnn_output: tensor of shape (Seq Len, Batch, Hidden size)

    Parameters
    ----------
    input_size: int, input size

    hidden_size: int, hidden size

    output_size: int, output size
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, **kwargs):

        super().__init__()

        # Continuous time RNN
        self.rnn = CTRNN(input_size, hidden_size, **kwargs)

        # Add an output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.tensor):
        """
        Step forward in RNN.

        Parameters
        ----------
        x : torch.tensor
            Tensor of shape (Seq Len, Batch, Input size)

        Returns
        -------
        tuple (out : torch.tensor, rnn_output : torch.tensor)
            out: tensor of shape (Seq Len, Batch, Output size)
            rnn_output: tensor of shape (Seq Len, Batch, Hidden size)

        """
        rnn_output, _ = self.rnn(x)
        out = self.fc(rnn_output)

        return out, rnn_output


class TAMNet:
    """
    Makes instances of CNNs an RNNs and trains them.
    """
    def __init__(
            self,
            dataset: ngym.Dataset,
            hidden_size: int,
            output_size: int,
            learning_rate: float,
            device: torch.device,
            cnn_name: str = None,
    ):

        # Initialize
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.device = device
        self.dataset = dataset

        # Instantiate CNN
        if cnn_name is not None:
            self.CNN, self.image_transformer = self._init_cnn(cnn_name)
        else:
            self.CNN = None

        # Instantiate an RNN
        self.RNN = self._init_rnn()

    def _init_cnn(self, name: str) -> tuple:
        """
        Initializes a CNN and an image transformer.

        Parameters
        ----------
        name : str
            Name of the CNN e.g. VGG16, AlexNet, ResNet50, etc.

        Returns
        -------
        tuple
            (torch.nn.Module, torchvision.transforms)

        """
        # Transformer
        image_transformer = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(227) if name == 'alexnet' else transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # CNN
        cnn = torch.hub.load('pytorch/vision:v0.10.0', name, pretrained=True).to(self.device)

        return cnn, image_transformer

    def _init_rnn(self) -> RNNNet:
        """
        Instantiates an RNN based on whether

        Returns
        -------
        RNN : RNNNet
        """
        # Get input dimensions from dataset
        inputs, _ = self.dataset()
        inputs = inputs[0][0].reshape(1, 1, *inputs.shape[2:])

        # If using a CNN first run one pass of the input to get layer feature dimension. Otherwise, just
        # find the dimension from dataset's H x W.
        if self.CNN:
            input_size = self.run_cnn(inputs).shape[2]
        else:
            input_size = inputs.reshape(*inputs.shape[:2], -1).shape[-1]

        # Make the RNN
        rnn = RNNNet(
            input_size=input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            dt=self.dataset.env.dt
        )

        # send to device
        rnn = rnn.to(self.device)

        return rnn

    def run_cnn(self, images: np.ndarray) -> np.ndarray:
        """
        Runs a trained CNN on a batch of images

        Parameters
        ----------
        images : np.ndarray
            Input images to the CNN

        Returns
        -------
        features : np.ndarray
            Extracted features from the images by the model
        """
        # Initiate features dimension W x H x N
        features = np.zeros((images.shape[0], images.shape[1], self.hidden_size))

        # Run through timepoints
        for t_point in range(images.shape[0]):

            # Run through batches
            for batch in range(images.shape[1]):

                # Select the image in this time point and batch (T, B, C, H, W)
                img = images[t_point, batch, :, :, :]

                # Swap the dimensions
                img = np.transpose(img, (1, 2, 0))

                # Transform the input to what the CNN likes
                img = self.image_transformer(img)

                # Add a mini batch dimension because the model expects it
                img = img.unsqueeze(dim=0)

                # Send to device
                img = img.to(self.device)

                # Run the model
                feats = self.CNN(img)

                # Save extracted features
                features[t_point, batch] = feats.cpu().detach().numpy()

        return features

    def train_rnn(self, datasets=None, n_epochs=100) -> tuple:
        """
        Function to train an RNN model.

        Parameters
        ----------
        datasets : list of ngym.Dataset
            Dataset objects to train on.

        n_epochs : int
            Number of training iterations.

        Returns
        -------
        tuple (list, list) of training loss and accuracy values.
        """
        # Use Adam optimizer
        optimizer = optim.Adam(self.RNN.parameters(), lr=self.learning_rate)

        # Define loss function
        criterion = nn.CrossEntropyLoss()
        running_loss = 0

        # Timestamps
        start_time = time.time()

        # Initialize lists to store the loss and accuracy values
        train_loss_list = []
        train_acc_list = []

        # Early stopping
        patience = 10
        best_loss = np.inf
        epochs_without_improvement = 0

        # Loop over training batches
        print('Training started...')

        for i in range(n_epochs):
            # print(f"Iteration: {i}")

            # Get image sequences from dataset object
            if datasets:
                inputs = []
                labels = []
                for dataset in datasets:
                    ins, labs = dataset()
                    inputs.append(ins)
                    labels.append(labs)
                inputs = np.concatenate(inputs, axis=1)
                labels = np.concatenate(labels, axis=1)
            else:
                inputs, labels = self.dataset()

            # Flatten labels
            labels = labels.flatten()

            # Modify inputs by either changing shape or sending them through the CNN
            if self.CNN:

                # get features from the inputs
                inputs = self.run_cnn(inputs)

            else:

                # Flatten images
                inputs = inputs.reshape(*inputs.shape[:2], -1)

            # Turn to tensors
            inputs = torch.from_numpy(inputs).type(torch.float)
            labels = torch.from_numpy(labels).type(torch.long)

            # Transfer to device
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Basic pytorch training
            optimizer.zero_grad()  # zero the gradient buffers
            output, _ = self.RNN(inputs)  # Run RNN

            # Reshape to (SeqLen x Batch, OutputSize)
            output = output.view(-1, self.output_size)

            # Compute loss
            loss = criterion(output, labels)
            loss.backward()

            # Compute accuracy
            acc = accuracy_score(labels.cpu().detach().numpy(), torch.argmax(output, dim=1).cpu().detach().numpy())

            # Update
            optimizer.step()

            # Compute the running loss every 100 steps and print
            loss_value = loss.item()
            train_loss_list.append(loss_value)
            train_acc_list.append(acc)

            running_loss += loss.item()

            if i % 100 == 99:
                running_loss /= 100
                print('Step {}, Loss {:0.4f}, Time {:0.1f}s'.format(i + 1, running_loss, time.time() - start_time))
                running_loss = 0

            # if loss_value < best_loss:
            #     best_loss = loss_value
            #     epochs_without_improvement = 0
            # else:
            #     epochs_without_improvement += 1
            #
            # if epochs_without_improvement == patience:
            #     print(f"Early stopping at epoch {i}.")
            #     break

        return train_loss_list, train_acc_list

    def test_rnn(self, n_trials: int, dataset=None) -> tuple:
        """
        Tests a trained model on a generated dataset

        Parameters
        ----------
        n_trials : int
            Number of trials

        Returns
        -------
        tuple (trial_infos, activity, performance)

            trial_infos : np.ndarray
                Populated with dictionaries

            activity : np.ndarray
                Populated with np.ndarray

            performance : float
                Overall performance of the network
        """
        # Reset dataset's environment (it's a copy of the original TAM environment at a different memory location)
        if dataset:
            env = dataset.env
        else:
            env = self.dataset.env

        env.reset(no_step=True)

        # Initialize variables for logging
        activity = np.zeros(n_trials, dtype=object)  # to record activity
        trial_infos = np.zeros(n_trials, dtype=object)

        # Run the trials
        for i in range(n_trials):

            # Initiate the dictionary of this trial
            trial_infos[i] = defaultdict()

            # Sample a new trial
            _ = env.new_trial()

            # Observation and ground-truth of this trial
            ob, gt = env.ob, env.gt

            # Convert to numpy, add batch dimension to input
            if self.CNN:

                # change the input shape
                ob = ob[:, np.newaxis, :]

                # run them through the CNN to extract features
                inputs = self.run_cnn(ob)

            else:

                # change the input shape
                inputs = ob.reshape(*ob.shape[:1], -1)

            # transform to tensors
            inputs = torch.from_numpy(inputs).type(torch.float)
            inputs = inputs.to(self.device)

            # Compute performance
            action_pred, rnn_activity = self.RNN(inputs)

            # Convert back to numpy
            action_pred = action_pred.cpu().detach().numpy()[:, 0, :]

            # Readout final choice at last time step
            choice = np.argmax(action_pred[-1, :])

            # Compare to ground truth
            correct = choice == gt[-1]

            # Record activity
            rnn_activity = rnn_activity[:, 0, :].cpu().detach().numpy()
            activity[i] = rnn_activity

            # Record trial information: ground truth, choice, correctness
            trial_infos[i]["ground_truth"] = gt[-1]
            trial_infos[i]["choice"] = choice
            trial_infos[i]["correct"] = correct

        performance = np.mean([info["correct"] for info in trial_infos])

        return trial_infos, activity, performance

    def _trial_type_cnn(self):
        """
        Runs CNN for each trial type in the stimulus set and extracts the features for each designated layer to make
        the process more efficient: the CNN doesn't need to be run on every trial so training becomes much
        faster.


        Returns
        -------

        """


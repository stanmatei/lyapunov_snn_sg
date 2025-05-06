import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import stork.datasets
from stork.models import RecurrentSpikingModel
from stork.nodes import InputGroup, ReadoutGroup, LIFGroup
from stork.connections import Connection
from stork.generators import StandardGenerator
from stork.initializers import FluctuationDrivenCenteredNormalInitializer
import seaborn as sns
import pickle, os, gc
import tqdm
import ray
import pathlib
import snntorch.spikeplot as splt
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import math
from models import SurrGradSpike, SNNLyapunov

def plot_lyapunov_and_loss(lyapunov_spectrum, losses):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    lyapunov_line, = ax1.plot([], [], "k", label="Lyapunov Spectrum")
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Lyapunov Exponent")
    ax1.set_title("Lyapunov Spectrum")
    ax1.legend()

    loss_line, = ax2.plot([], [], "r", label="Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Loss over Epochs")
    ax2.legend()

    lyapunov_line.set_ydata(lyapunov_spectrum)
    lyapunov_line.set_xdata(range(len(lyapunov_spectrum)))
    ax1.relim()
    ax1.autoscale_view()

    loss_line.set_ydata(losses)
    loss_line.set_xdata(range(len(losses)))
    ax2.relim()
    ax2.autoscale_view()

    plt.show()

def plot_mambrane_voltages(membrane_voltages: list, num_samples, num_neurons):
    # Generate random indices
    random_indices = np.random.choice(membrane_voltages[0].shape[1], num_samples, replace=False)
    random_neuron = np.random.choice(membrane_voltages[0].shape[2], num_neurons, replace=False)

    for i in random_indices:
        for j in random_neuron:
            for v in membrane_voltages:
              plt.plot(v[:, i, j].detach().cpu().numpy(), label=f'Sample {i}, Neuron {j}')
            plt.xlabel('Time Step')
            plt.ylabel('Membrane Potential')
            plt.title('Sample Membrane Voltage Evolutions')
            plt.legend()
            plt.show()

num_samples = 1

def plot_rasters(spike_outputs: list, num_samples):
  random_samples = np.random.choice(spike_outputs[0].shape[1], num_samples, replace=False)
  for so in spike_outputs:
    print(f"Overall Firing Rate: {torch.mean(so).item()}")
  for i in random_samples:
    for so in spike_outputs:
      print(f"Firing Rate: {torch.mean(so[:, i, :]).item()}")
      spike_sample_data = so[:, i, :]
      fig = plt.figure(facecolor="w", figsize=(10, 5))
      ax = fig.add_subplot(111)
      splt.raster(spike_sample_data, ax, s=1.5, c="black")
      plt.title(f"Sample {i}")
      plt.xlabel("Time step")
      plt.ylabel("Neuron Number")
      plt.show()

def plot_mean_firing_rates(spike_outputs: list):

  for idx, so in enumerate(spike_outputs):
    mean_spiking_per_sample = torch.mean(so, dim = (0, 2))
    sns.kdeplot(mean_spiking_per_sample.detach().cpu().numpy(), label=f'Model {idx}')
  plt.xlabel('Mean Spiking Rate per Sample')
  plt.ylabel('Density')
  plt.title('KDE of Mean Spiking Rate per Sample')
  plt.legend()
  plt.show()

  for idx, so in enumerate(spike_outputs):
    mean_spiking_per_neuron = torch.mean(so, dim = (0, 1))
    sns.kdeplot(mean_spiking_per_neuron.detach().cpu().numpy(), label=f'Model {idx}')
  plt.xlabel('Mean Spiking Rate per Neuron')
  plt.ylabel('Density')
  plt.title('KDE of Mean Spiking Rate per Neuron')
  plt.legend()
  plt.show()

def plot_membrane_voltage_dist(membrane_voltages: list):
  for idx, v in enumerate(membrane_voltages):
    flat_membrane = v.flatten().detach().cpu().numpy()
    sns.kdeplot(flat_membrane, label = f"Model {idx}")
  plt.xlabel('Membrane Potential')
  plt.ylabel('Density')
  plt.title('Membrane Voltage Distribution')
  plt.legend()
  plt.show()

def plot_rec_eigenvalues(model: SNNLyapunov):
  v1 = model.snn_layer.v1.weight
  eigenvalues = torch.linalg.eigvals(v1)
  plt.figure(figsize=(8, 6))
  plt.scatter(eigenvalues.real.cpu().detach().numpy(), eigenvalues.imag.cpu().detach().numpy(), marker='o', s=10)
  plt.xlabel("Real part")
  plt.ylabel("Imaginary part")
  plt.title("Recurrent Weight Eigenvalues")
  plt.grid(True)
  plt.show()

def plot_gradients_dht_dh0(grads, prediction_offset, spikes = None, include_eps = False, step_size = 50):
  if len(grads.shape) < 2:
    grads = np.expand_dims(grads, axis = 0)

  for idx, dht_dh0 in enumerate(grads):
    log_dht_dh0 = np.log(dht_dh0)
    x_axis = np.arange(1, len(log_dht_dh0) * step_size + 1, step_size)
    plt.plot(x_axis, log_dht_dh0, label = f"Model {idx}")

  if spikes is not None:
      reduce_input = torch.sum(spikes, dim = -1).squeeze()
      spike_times = torch.where(reduce_input > 0)[0].cpu().numpy()
      for i in spike_times:
        plt.axvline(x=i, color='gray', alpha=0.35)

  if include_eps:
    eps_float32 = torch.finfo(torch.float32).eps
    plt.axhline(y = np.log(eps_float32))

  g_scale = SurrGradSpike.scale

  plt.xlabel("t")
  plt.ylabel("Log Norm of dht_dh0")
  plt.title(f"Log Norm of dht_dh0 over t for g = {g_scale} Prediction offset = {prediction_offset}")
  plt.legend()
  plt.show()


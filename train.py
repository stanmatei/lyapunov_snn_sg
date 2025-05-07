import argparse
from models import SurrGradSpike, SNNLyapunov
from streaming_randman import get_streaming_randman
from utils import get_dht_dh0, str2bool
from plotting import plot_gradients_dht_dh0, plot_lyapunov_and_loss, plot_mambrane_voltages, plot_mean_firing_rates, plot_membrane_voltage_dist, plot_rasters, plot_rec_eigenvalues
import torch
import torch.optim as optim
from torch.autograd.functional import jacobian
from torch.func import jacfwd, jacrev, vmap
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
from functools import partial
import torch.nn.utils.parametrize as parametrize
from scipy.stats import ortho_group
import copy
import math
import datetime
import json
import wandb

class RandmanDataset(Dataset):
    def __init__(self, randman_data) -> None:
        super().__init__()
        self.dataset = randman_data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx][0], self.dataset[idx][1]
    
def prefloss(model, dataloader, n_epochs, lr, batch_size, n_samples_streaming, seed, device, use_scheduler):

  torch.manual_seed(seed)
  np.random.seed(seed)
  optimizer = optim.Adam(model.parameters(), lr = lr)
  if use_scheduler:
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

  dataset = dataloader.dataset
  train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

  num_labels = len(model.prediction_times)

  # Initialize lists to store the loss and Lyapunov spectrum for plotting
  losses = []
  lyapunov_spectra = []
  list_dht_dh0 = []
  membranes_out = []
  spikes_out = []

  for epoch in tqdm.tqdm(range(1, n_epochs + 1)):

      inputs, labels = get_streaming_randman(n_samples=n_samples_streaming, dataloader=train_dataloader, batch_size=batch_size)

      inputs = inputs.transpose(1, 0).to(device)
      labels = labels[:, :num_labels].to(device)

      torch.manual_seed(epoch)
      optimizer.zero_grad()

      lyapunov_spectrum, out, mem_out, spike_out = model.get_lyapunov_spectrum(inputs)
      loss = torch.mean(lyapunov_spectrum**2)

      loss.backward()
      optimizer.step()
      if use_scheduler:
        scheduler.step()

      losses.append(loss.item())
      lyapunov_spectra.append(lyapunov_spectrum.detach().cpu().numpy())
      membranes_out.append(mem_out.detach().cpu().numpy())
      spikes_out.append(spike_out.detach().cpu().numpy())

      wandb.log({"prefloss_loss":loss.item()})
      wandb.log({"prefloss_spike_rate":torch.mean(spike_out).item()})
      wandb.log({"prefloss_largest_le":lyapunov_spectrum[0][0].item()})


      if epoch % 10 == 0:
          #plot_rasters([spike_out], 1)
          #plot_mambrane_voltages([mem_out], 1, 1)

          h0 = torch.zeros((inputs.shape[1], model.nb_hidden * 2), dtype=torch.float32, requires_grad = True).to(device)
          dht_dh0 = get_dht_dh0(model, inputs, h0, model.prediction_times[-1], step_size = 50)
          list_dht_dh0.append(dht_dh0)
          wandb.log({"prefloss_dht_dh0":dht_dh0[-1]})

          print(f"Epoch {epoch}: Loss = {loss.item()}")
          #plot_lyapunov_and_loss(lyapunov_spectrum[0].detach().cpu().numpy(), losses)
          #print(f"g = {model.g}, grad = {model.g.grad}")

  #plot_gradients_dht_dh0(np.array(list_dht_dh0))
  optimizer.zero_grad()
  #torch.cuda.empty_cache()
  #gc.collect()
  print("Pre-Flossing complete.")

  results = {
      "losses": losses,
      "lyapunov_spectra":lyapunov_spectra,
      "list_dht_dh0": list_dht_dh0,
      #"membranes_out": membranes_out,
      #"spikes_out":spikes_out,
  }

  return model, results

def train_model(model, train_dataloader, test_dataloader, n_epochs, lr, batch_size, gradient_flossing_period, n_samples_streaming, seed, device, use_scheduler):

  torch.manual_seed(seed)
  np.random.seed(seed)
  # Optimization setup
  optimizer = optim.Adam(model.parameters(), lr = lr)
  if use_scheduler:
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

  dataset = train_dataloader.dataset
  train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

  dataset = test_dataloader.dataset
  test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

  losses = []
  accuracies = []
  ls = torch.zeros((batch_size, model.nle), dtype=torch.float32).to(device)
  lyapunov_spectra = [ls.cpu().detach().numpy()]
  criterion = torch.nn.CrossEntropyLoss()

  num_labels = len(model.prediction_times)

  n_test = 1

  for epoch in tqdm.tqdm(range(1, n_epochs + 1)):

    inputs, labels = get_streaming_randman(n_samples=n_samples_streaming, dataloader=train_dataloader, batch_size=batch_size)

    inputs = inputs.transpose(1, 0).to(device)
    labels = labels[:, :num_labels].to(device)

    optimizer.zero_grad()

    if epoch % gradient_flossing_period == 0:
      lyapunov_spectrum, out, _, _ = model.get_lyapunov_spectrum(inputs)
      loss = criterion(out, labels) + torch.mean(lyapunov_spectrum**2)
      #lyapunov_spectra.append(lyapunov_spectrum.detach().cpu().numpy())
    else:
      out, _, _ = model(inputs)
      loss = criterion(out, labels)

    loss.backward()
    optimizer.step()
    if use_scheduler:
        scheduler.step()
    losses.append(loss.item())

    wandb.log({"train_loss":loss.item()})

    if epoch % 20 == 0:
      with torch.no_grad():
        print(f"Epoch {epoch}: Loss = {loss.item()}")
        rand_idx = np.random.randint(len(lyapunov_spectra[-1]))
        #plot_lyapunov_and_loss(lyapunov_spectra[-1][rand_idx], losses)
        acc = 0
        for t in range(n_test):

          inputs, labels = get_streaming_randman(n_samples=n_samples_streaming, dataloader=test_dataloader, batch_size=batch_size)
          inputs = inputs.transpose(1, 0).to(device)
          labels = labels[:, :num_labels].to(device)

          out, _, _ = model(inputs)
          correct = torch.argmax(out, dim = 1) == labels
          acc += torch.sum(correct)

        n_batch = inputs.shape[1]
        acc = acc / (n_test * n_batch * num_labels)
        wandb.log({"test_acc":acc})
        print(f"Epoch {epoch}, Accuracy: {acc}")
        accuracies.append(acc)

    results = {
       "losses": losses,
       "accuracies": accuracies,
    }
        
  return model, results

def run(args):

    if args.device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")  # Set device to GPU
            print("GPU is available")
        else:
            device = torch.device("cpu")  # Set device to CPU
            print("GPU is not available")
    else:
       device = torch.device("cpu")

    experiment_name = str(json.dumps(vars(args)))

    #os.environ['WANDB_API_KEY'] = args.wandb_key

    wandb.login()
    wandb.init(
        entity="snn_nlp",
        project="lyapunov_snn",
        config=vars(args),
    )
    
    duration = args.nb_time_steps * args.dt

    data, labels = stork.datasets.make_tempo_randman(
        dim_manifold=args.dim_manifold,
        nb_classes=args.nb_classes,
        nb_units=args.nb_inputs,
        nb_steps=args.nb_time_steps,
        step_frac=args.step_frac,
        nb_samples=args.nb_samples,
        nb_spikes=args.nb_spikes,
        alpha=args.alpha,
        seed=args.randmanseed,
    )

    ds_kwargs = dict(nb_steps=args.nb_time_steps, nb_units=args.nb_inputs, time_scale=1.0)

    # Split into train, test and validation set
    datasets = [
        stork.datasets.RasDataset(ds, **ds_kwargs)
        for ds in stork.datasets.split_dataset(
            data, labels, splits=[0.8, 0.1, 0.1], shuffle=False
        )
    ]
    ds_train, ds_valid, ds_test = datasets

    train_dataloader = DataLoader(RandmanDataset(ds_train), batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(RandmanDataset(ds_valid), batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(RandmanDataset(ds_test), batch_size=args.batch_size, shuffle=True)

    prediction_times = []
    sample_len = args.nb_time_steps // 2
    total_len = args.n_samples_streaming * sample_len

    for i in range(1, args.n_samples_streaming):
        temp = i * sample_len + args.prediction_offset
        if temp < total_len:
            prediction_times.append(temp)
        else:
            break

    kwargs = {
        'tau_mem': args.tau_mem,
        'tau_readout': args.tau_readout,
        'tau_syn': args.tau_syn,
        'time_step': args.time_step,
        'nb_inputs': args.nb_inputs,
        'nb_hidden': args.nb_hidden,
        'nb_outputs': args.nb_outputs,
        'iext': args.iext,
        'g': torch.tensor(args.g, dtype=torch.float32, requires_grad=False),
        'nle': args.nle,
        'seedONS':args.seedONS,
        'ONSstep': args.ONSstep,
        'prediction_times': prediction_times,
        'seed_init': args.seed_init,
        'sigma_v1': args.sigma_v1,
        'sigma_Win': args.sigma_Win,
        'mu_v1': args.mu_v1,
        'mu_Win': args.mu_Win,
        'trainable_tau': args.trainable_tau,
        'trainable_dt': args.trainable_dt,
        'device':device
    }

    prefloss_kwargs = {
        'model' : None,
        'dataloader' : train_dataloader,
        'n_epochs' : args.n_preflossing_epochs,
        'lr' : args.lr_pf,
        'batch_size' : args.prefloss_batch_size,
        'n_samples_streaming' : args.n_samples_streaming,
        'seed':args.seed_train,
        "device": device,
        "use_scheduler": args.use_scheduler,
    }

    train_kwargs = {
        "model" : None,
        "train_dataloader" : train_dataloader,
        "test_dataloader" : test_dataloader,
        "n_epochs" : args.n_epochs,
        "lr" : args.lr_main,
        "batch_size" : args.batch_size,
        "gradient_flossing_period" : args.gradient_flossing_period,
        "n_samples_streaming" : args.n_samples_streaming,
        "seed" : args.seed_train,
        "device" : device,
        "use_scheduler": args.use_scheduler,
    }
    
    results = {}
    model = SNNLyapunov(**kwargs)
    model= model.to(device)
    
    if args.prefloss == True:
       prefloss_kwargs["model"] = model
       model, prefloss_results = prefloss(**prefloss_kwargs)
       results["prefloss_results"] = prefloss_results
    
    train_kwargs["model"] = model
    model, training_results = train_model(**train_kwargs)

    results["training_results"] = training_results

    args_dict = vars(args)

    results["args"] = args_dict

    d, t = str(datetime.datetime.now()).split(" ")
    output_location = os.path.join(args.output_dir, d)
    output_location = os.path.join(output_location, t)

    #os.makedirs(output_location, exist_ok=True)

    #with open(os.path.join(output_location, 'args.json'), 'w') as f:
    #    json.dump(args_dict, f, indent=4)

    #with open(os.path.join(output_location, 'results.pkl'), 'wb') as f:
    #    pickle.dump(results, f)

    #torch.save(model, os.path.join(output_location, "model.pt"))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str)

    parser.add_argument("--device", type=str, choices=["cpu", "cuda"])
    parser.add_argument("--dim_manifold", type=int, default=4)
    parser.add_argument("--nb_classes", type=int, default=2)
    parser.add_argument("--nb_inputs", type=int, default=128)
    parser.add_argument("--nb_time_steps", type=int, default=100)
    parser.add_argument("--step_frac", type=float, default=0.5)
    parser.add_argument("--nb_samples", type=int, default=1000)
    parser.add_argument("--nb_spikes", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=3)
    parser.add_argument("--randmanseed", type=int, default=2)
    parser.add_argument("--dt", type=float, default=2e-3)
    parser.add_argument("--batch_size", type=int, default=250)
    
    parser.add_argument("--tau_mem", type=float, default=20e-3)
    parser.add_argument("--tau_readout", type=float, default=100)
    parser.add_argument("--tau_syn", type=float, default=10e-3)
    parser.add_argument("--time_step", type=float, default=1000.)
    parser.add_argument("--nb_hidden", type=int, default=128)
    parser.add_argument("--nb_outputs", type=int, default=2)
    parser.add_argument("--iext", type=float, default=0)
    parser.add_argument("--g", type=float, default=5)
    parser.add_argument("--nle", type=int, default=20)
    parser.add_argument("--seedONS", type=int, default=1)
    parser.add_argument("--ONSstep", type=int, default=100)
    parser.add_argument("--prediction_offset", type=int, default=80)
    parser.add_argument("--n_samples_streaming", type=int, default=16)
    parser.add_argument("--prefloss_batch_size", type=int, default=1)
    parser.add_argument("--lr_pf", type=float, default=1e-2)
    parser.add_argument("--lr_main", type=float, default=4e-3)
    parser.add_argument("--n_epochs", type=int, default=3000)
    parser.add_argument("--n_preflossing_epochs", type=int, default=100)
    parser.add_argument("--gradient_flossing_period", type=int, default=4000)
    parser.add_argument("--seed_init", type=int, default=1)
    parser.add_argument("--sigma_v1", type=float, default=5)
    parser.add_argument("--sigma_Win", type=float, default=5)
    parser.add_argument("--mu_v1", type=float, default=0.0)
    parser.add_argument("--mu_Win", type=float, default=0.0)
    parser.add_argument("--trainable_tau", type=str2bool, default=False)
    parser.add_argument("--trainable_dt", type=str2bool, default=False)
    parser.add_argument("--seed_train", type=int, default=1)
    parser.add_argument("--prefloss", type=str2bool, default=False)
    #parser.add_argument("--wandb_key", type=str)
    parser.add_argument("--use_scheduler", type=str2bool, default=False)
    
    args = parser.parse_args()

    run(args)
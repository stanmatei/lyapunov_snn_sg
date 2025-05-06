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
from dysts.flows import MackeyGlass
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

class SurrGradSpike(torch.autograd.Function):
    """
    We use  fast sigmoid as a surrogate gradient funtion as in Zenke & Ganguli (2018).
    """
    scale = 5 # controls steepness of surrogate gradient
    generate_vmap_rule = True

    @staticmethod
    def forward(input):
        #ctx.save_for_backward(input)
        out = (input >= 0.).to(torch.float)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SurrGradSpike.scale*torch.abs(input)+1.0)**2
        return grad

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        # Save any tensors needed for backward pass here
        # This is a minimal implementation, you might need to save more depending on your function's logic
        ctx.save_for_backward(inputs[0])

class SNNLayer(torch.nn.Module):
    def __init__(
            self,
            nb_inputs: int,
            nb_hidden: int,
            tau_mem: float,
            tau_syn: float,
            time_step: float,
            iext: float,
            g: any,
            device: any,
            seed_init: int = 1,
            sigma_v1: float = 0.01,
            sigma_Win: float = 0.01,
            mu_v1: float = 0.0,
            mu_Win: float = 0.0,
            trainable_tau: bool = False,
            trainable_dt: bool = False
          ) -> None:
        super().__init__()

        #fwd_weight_scale = 3.0
        #rec_weight_scale = 1e-2*fwd_weight_scale

        torch.manual_seed(seed_init)
        np.random.seed(seed_init)
        self.tau_syn = torch.tensor(tau_syn, dtype = torch.float).to(device)
        self.tau_mem = torch.tensor(tau_mem, dtype = torch.float).to(device)

        if trainable_tau:
          self.tau_syn = torch.nn.Parameter(torch.tensor(tau_syn, dtype = torch.float)).to(device)
          self.tau_mem = torch.nn.Parameter(torch.tensor(tau_mem, dtype = torch.float)).to(device)
        else:
          self.tau_syn = torch.tensor(tau_syn, dtype = torch.float).to(device)
          self.tau_mem = torch.tensor(tau_mem, dtype = torch.float).to(device)

        if trainable_dt:
          self.delta_t = torch.nn.Parameter(torch.tensor(1. / time_step), requires_grad=True).to(device)
        else:
          self.delta_t = torch.tensor(1. / time_step, dtype = torch.float).to(device)

        self.nb_inputs = nb_inputs
        self.nb_hidden = nb_hidden
        self.iext = iext
        self.g = g

        self.Win = torch.nn.Linear(nb_inputs, nb_hidden, bias=False)
        self.v1 = torch.nn.Linear(nb_hidden, nb_hidden, bias=False).to(device)

        Win_std = sigma_Win/np.sqrt(nb_inputs)
        v1_std = sigma_v1/np.sqrt(nb_hidden)

        torch.nn.init.normal_(self.Win.weight, mean=mu_Win, std=Win_std)
        torch.nn.init.normal_(self.v1.weight, mean=mu_v1, std=v1_std)

        self.spike_fn = SurrGradSpike.apply
        self.device = device


    def forward(self, state, x):
        x = self.Win(x)
        mem, syn = state.chunk(2, dim = -1)
        mem = mem.squeeze(-1).to(self.device)
        syn = syn.squeeze(-1).to(self.device)
        alpha = torch.exp(-self.delta_t / self.tau_syn).to(self.device)
        beta = torch.exp(-self.delta_t/ self.tau_mem).to(self.device)
        mthr = mem - 1
        out = self.spike_fn(mthr)
        new_syn = alpha * syn + x + self.v1(out)
        new_mem = (beta * mem + (1. - beta) * syn) * (1. - out)
        next_state = torch.cat([new_mem, new_syn], dim = -1)
        return next_state, (next_state, out)

class ReadoutLayer(torch.nn.Module):
    def __init__(
            self,
            nb_hidden: int,
            tau_mem: float,
            tau_syn: float,
            time_step: float,
            device: any,
          ) -> None:
        super().__init__()
        self.nb_hidden = nb_hidden
        self.tau_syn = torch.tensor(tau_syn, dtype = torch.float).to(device)
        self.tau_mem = torch.tensor(tau_mem, dtype = torch.float).to(device)
        self.delta_t = torch.nn.Parameter(torch.tensor(1. / time_step), requires_grad=True).to(device)
        self.device = device

    def forward(self, state, x):
        alpha = torch.exp(-self.delta_t / self.tau_syn).to(self.device)
        beta = torch.exp(-self.delta_t/ self.tau_mem).to(self.device)
        mem, syn = state.chunk(2, dim = -1)
        mem = mem.squeeze(-1)
        syn = syn.squeeze(-1)
        new_syn = alpha * syn + x
        new_mem = beta * mem + (1. - beta) * syn
        next_state = torch.cat([new_mem, new_syn], dim = -1)
        out = new_mem
        return next_state, out

class SNNLyapunov(torch.nn.Module):
    def __init__(
            self,
            nb_inputs: int,
            nb_hidden: int,
            nb_outputs: int,
            tau_mem: float,
            tau_syn: float,
            tau_readout: float,
            time_step: float,
            iext: float,
            g: any,
            device: any,
            nle: int,
            seedONS: int,
            ONSstep: int,
            prediction_times: list,
            seed_init: int = 1,
            sigma_v1: float = 0.01,
            sigma_Win: float = 0.01,
            mu_v1: float = 0.0,
            mu_Win: float = 0.0,
            trainable_tau: bool = False,
            trainable_dt: bool = False
          ) -> None:
        super().__init__()
        torch.manual_seed(seed_init)
        np.random.seed(seed_init)
        self.nb_inputs = nb_inputs
        self.nb_hidden = nb_hidden
        self.nb_outputs = nb_outputs
        self.iext = iext
        self.g = g
        self.nle = nle
        self.seedONS = seedONS
        self.decoder = torch.nn.Linear(self.nb_hidden, self.nb_outputs)
        self.snn_layer = SNNLayer(nb_inputs=nb_inputs,
                                  nb_hidden=nb_hidden,
                                  tau_mem=tau_mem,
                                  tau_syn=tau_syn,
                                  time_step=time_step,
                                  iext=iext,
                                  g=g,
                                  device=device,
                                  seed_init=seed_init,
                                  sigma_v1=sigma_v1,
                                  sigma_Win=sigma_Win,
                                  mu_v1=mu_v1,
                                  mu_Win=mu_Win,
                                  trainable_tau=trainable_tau,
                                  trainable_dt=trainable_dt,
                                )
        self.ONSstep = ONSstep
        self.prediction_times = prediction_times
        self.device = device

    def get_lyapunov_spectrum(self, x_all):
        nb_batch = x_all.shape[1]
        n_time_steps = x_all.shape[0]
        mem = torch.zeros((nb_batch, self.nb_hidden), dtype=torch.float32)
        syn = torch.zeros((nb_batch, self.nb_hidden), dtype=torch.float32)
        state = torch.cat([mem, syn], dim = -1).to(self.device)
        readout_state = torch.ones((nb_batch, self.nb_outputs * 2), dtype=torch.float32).to(self.device) * 1e-2
        mem_output = []
        spike_output = []
        ls = torch.zeros((nb_batch, self.nle), dtype=torch.float32).to(self.device)
        torch.manual_seed(self.seedONS)
        Q, R = torch.linalg.qr(torch.randn(nb_batch, 2*self.nb_hidden, self.nle).to(self.device))

        for step, x in enumerate(x_all):
          D, (state, out_spikes) = vmap(jacrev(self.snn_layer.forward, argnums = 0, has_aux = True))(state, x)
          mem, _ = state.chunk(2, dim = -1)
          mem = mem.squeeze(-1)
          mem_output.append(mem)
          spike_output.append(out_spikes)
          Q = torch.einsum("...ij, ...jk-> ...ik", D, Q)
          if step % self.ONSstep == 0 and self.nle > 0:
            Q, R = torch.linalg.qr(Q)
            ls = ls + torch.log(torch.abs(torch.diagonal(R, dim1 = -2, dim2 = -1)))/ self.ONSstep
        mem_output = torch.stack(mem_output)
        spike_output = torch.stack(spike_output)
        out = self.decoder(mem_output[self.prediction_times])
        out = out.permute(1, 2, 0) #for CrossEntropyLoss
        return ls, out, mem_output, spike_output

    def forward(self, x_all, get_spectrum = False):
        #x_all.shape = (length, n_batch, dim_hidden)
        nb_batch = x_all.shape[1]
        n_time_steps = x_all.shape[0]
        mem = torch.zeros((nb_batch, self.nb_hidden), dtype=torch.float32)
        syn = torch.zeros((nb_batch, self.nb_hidden), dtype=torch.float32)
        state = torch.cat([mem, syn], dim = -1).to(self.device)
        readout_state = torch.ones((nb_batch, self.nb_outputs * 2), dtype=torch.float32).to(self.device) * 1e-2
        mem_output = []
        spike_output = []
        ls = torch.zeros((nb_batch, self.nle), dtype=torch.float32).to(self.device)

        for step, x in enumerate(x_all):
          _, (state, out_spikes) = self.snn_layer(state, x)
          mem, _ = state.chunk(2, dim = -1)
          mem = mem.squeeze(-1)
          mem_output.append(mem)
          spike_output.append(out_spikes)

        mem_output = torch.stack(mem_output)
        spike_output = torch.stack(spike_output)
        out = torch.amax(mem_output, dim = 0)
        out = self.decoder(mem_output[self.prediction_times])
        out = out.permute(1, 2, 0) #for CrossEntropyLoss
        return out, mem_output, spike_output
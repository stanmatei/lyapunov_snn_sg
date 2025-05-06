import torch
import torch.optim as optim
import tqdm
import argparse

def apply_forward(model, x_all, h0):
  h_t = h0
  h_all = []
  all_mem = []
  for step, x in enumerate(x_all):
    _, (h_t, _) = model.snn_layer(h_t, x)
    mem, _ = h_t.chunk(2, dim = -1)
    mem = mem.squeeze(-1)
    all_mem.append(mem)
  all_mem = torch.stack(all_mem)
  loss = torch.mean(all_mem, dim = -1)
  return loss

def get_dht_dh0(model, x_all, h0, max_prediction_step, step_size = 50):
  dht_dh0 = []
  optimizer = optim.Adam(model.parameters())
  z = apply_forward(model, x_all, h0)
  z = z.squeeze()
  for prediction_step in tqdm.tqdm(range(0, max_prediction_step + 1, step_size)):
    ht = z[prediction_step]
    dh0 = torch.autograd.grad(ht, h0, create_graph=True)[0]
    dh0_norm = torch.norm(dh0).item()
    dht_dh0.append(dh0_norm)
    optimizer.zero_grad()
  return dht_dh0


def str2bool(v):
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')
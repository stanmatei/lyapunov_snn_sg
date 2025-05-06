from torch.utils.data import Dataset, DataLoader
import stork.datasets
import torch

def get_streaming_randman(n_samples, dataloader, batch_size):

  it = iter(dataloader)
  input_ = []
  labels = []

  for i in range(n_samples):

    try:
      x, y = next(it)
    except StopIteration:
      it = iter(dataloader)
      x, y = next(it)

    if x.shape[0] < batch_size:
      try:
        x, y = next(it)
      except StopIteration:
        it = iter(dataloader)
        x, y = next(it)

    end_step = x.shape[1] // 2
    x = x[:, :end_step]
    input_.append(x)
    labels.append(y.unsqueeze(-1))

  input_ = torch.cat(input_, dim = 1)
  labels = torch.cat(labels, dim = 1)

  return input_, labels

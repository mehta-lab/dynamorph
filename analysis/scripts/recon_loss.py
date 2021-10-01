from HiddenStateExtractor.vq_vae import *
import torch

cs = [0, 1]
cs_mask = [2, 3]
input_shape = (128, 128)
gpu = False
path = '/mnt/comp_micro/Projects/CellVAE'

### Load Data ###
fs = pickle.load(open('./HiddenStateExtractor/file_paths_bkp.pkl', 'rb'))

dataset = torch.load('StaticPatchesAll.pt')
dataset = rescale(dataset)
B4_dataset = torch.load('../data_temp/B4_all_adjusted_static_patches.pt')
B4_dataset = rescale(B4_dataset)

model = VQ_VAE(alpha=0.0005, gpu=gpu)
model.load_state_dict(torch.load('./HiddenStateExtractor/save_0005_bkp4.pt', map_location='cpu'))

np.random.seed(123)
r_losses = []
for i in np.random.choice(np.arange(len(dataset)), (5000,), replace=False):
  sample = dataset[i:(i+1)][0]
  output, loss = model.forward(sample)
  r_loss = loss['recon_loss']
  r_losses.append(r_loss.data.cpu().numpy())

B4_r_losses = []
for i in np.random.choice(np.arange(len(B4_dataset)), (5000,), replace=False):
  sample = B4_dataset[i:(i+1)][0]
  output, loss = model.forward(sample)
  r_loss = loss['recon_loss']
  B4_r_losses.append(r_loss.data.cpu().numpy())

#0.00756±0.01691
#0.00795±0.00617
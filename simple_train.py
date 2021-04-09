import numpy as np
import os
import pickle
from HiddenStateExtractor.vq_vae_supp import reorder_with_trajectories, vae_preprocess, train
from HiddenStateExtractor.vq_vae import VQ_VAE
import torch as t
from torch.utils.data import TensorDataset, DataLoader

### Settings ###
cs = [1]
model_output_dir = "./retardance_only_model"
device = "cuda:1"

### Prepare Data ###
path = '/gpfs/CompMicro/projects/dynamorph/microglia/raw_for_segmentation'
fs = pickle.load(open(os.path.join(path, 'JUNE', 'raw', 'D_file_paths.pkl'), 'rb'))
dataset = pickle.load(open(os.path.join(path, 'JUNE', 'raw', 'D_static_patches.pkl'), 'rb'))
dataset_mask = pickle.load(open(os.path.join(path, 'JUNE', 'raw', 'D_static_patches_mask.pkl'), 'rb'))
relations = pickle.load(open(os.path.join(path, 'JUNE', 'raw', 'D_static_patches_relations.pkl'), 'rb'))

# Reorder
dataset, relation_mat, inds_in_order = reorder_with_trajectories(dataset, relations, seed=123)
fs = [fs[i] for i in inds_in_order]
dataset_mask = dataset_mask[np.array(inds_in_order)]
dataset = vae_preprocess(dataset, use_channels=cs)

dataset = TensorDataset(t.from_numpy(dataset).float())
dataset_mask = TensorDataset(t.from_numpy(dataset_mask).float())




os.makedirs(os.path.join(model_output_dir, "stage1"), exist_ok=True)
os.makedirs(os.path.join(model_output_dir, "stage2"), exist_ok=True)

# Stage 1 training
model = VQ_VAE(num_inputs=1, alpha=0., channel_var=np.ones((1,)), device=device)
model = model.to(device)
model = train(model, 
              dataset, 
              os.path.join(model_output_dir, "stage1"),
              relation_mat=relation_mat, 
              mask=dataset_mask,
              n_epochs=100, 
              lr=0.0001, 
              batch_size=128, 
              device=device,
              shuffle_data=False,
              transform=True)


model = VQ_VAE(num_inputs=1, alpha=0.0005, channel_var=np.ones((1,)), device=device)
model = model.to(device)
model.load_state_dict(t.load(os.path.join(model_output_dir, "stage1", "model_epoch99.pt")))
model = train(model, 
              dataset, 
              os.path.join(model_output_dir, "stage2"),
              relation_mat=relation_mat, 
              mask=dataset_mask,
              n_epochs=400, 
              lr=0.0001, 
              batch_size=128, 
              device=device,
              shuffle_data=False,
              transform=True)


### Check coverage of embedding vectors ###
# used_indices = []
# for i in range(500):
#     sample = dataset[i:(i+1)][0].cuda()
#     z_before = model.enc(sample)
#     indices = model.vq.encode_inputs(z_before)
#     used_indices.append(np.unique(indices.cpu().data.numpy()))
# print(np.unique(np.concatenate(used_indices)))

### Generate latent vectors ###
# z_bs = {}
# z_as = {}
# for i in range(len(dataset)):
#     sample = dataset[i:(i+1)][0].cuda()
#     z_b = model.enc(sample)
#     z_a, _, _ = model.vq(z_b)
#     f_n = fs[inds_in_order[i]]
#     z_as[f_n] = z_a.cpu().data.numpy()
#     z_bs[f_n] = z_b.cpu().data.numpy()
  

### Visualize reconstruction ###
# def enhance(mat, lower_thr, upper_thr):
#     mat = np.clip(mat, lower_thr, upper_thr)
#     mat = (mat - lower_thr)/(upper_thr - lower_thr)
#     return mat

# random_inds = np.random.randint(0, len(dataset), (10,))
# for i in random_inds:
#     sample = dataset[i:(i+1)][0].cuda()
#     cv2.imwrite('sample%d_0.png' % i, 
#         enhance(sample[0, 0].cpu().data.numpy(), 0., 1.)*255)
#     cv2.imwrite('sample%d_1.png' % i, 
#         enhance(sample[0, 1].cpu().data.numpy(), 0., 1.)*255)
#     output = model(sample)[0]
#     cv2.imwrite('sample%d_0_rebuilt.png' % i, 
#         enhance(output[0, 0].cpu().data.numpy(), 0., 1.)*255)
#     cv2.imwrite('sample%d_1_rebuilt.png' % i, 
#         enhance(output[0, 1].cpu().data.numpy(), 0., 1.)*255)
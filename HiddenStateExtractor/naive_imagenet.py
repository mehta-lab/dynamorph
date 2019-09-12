import os
import numpy as np
import h5py
import pickle

from classification_models.resnet import ResNet50
from classification_models.resnet import preprocess_input as preprocess_input_resnet50
import classification_models.keras_applications as ka

from keras.models import Model
import cv2

DATA_ROOT = '/mnt/comp_micro/Projects/CellVAE'
CHANNEL_MAX = [65535., 65535.]

def read_file_path(root):
  files = []
  for dir_name, dirs, fs in os.walk(root):
    for f in fs:
      if f.endswith('.h5'):
        files.append(os.path.join(dir_name, f))
  return files

def initiate_model():
  model = ResNet50((224, 224, 3), weights='imagenet')
  target_layer = [l for l in model.layers if l.name == 'pool1'][0]
  hidden_extractor = Model(model.input, target_layer.output)
  hidden_extractor.compile(loss='mean_squared_error', optimizer='sgd')
  return hidden_extractor, preprocess_input_resnet50

def initiate_model_inception():
  model = ka.inception_resnet_v2.InceptionResNetV2(input_shape=(224, 224, 3), 
                                                   weights='imagenet', 
                                                   include_top=False,
                                                   pooling='avg')
  preprocess_fn = ka.inception_resnet_v2.preprocess_input
  return model, preprocess_fn

def preprocess(f_n, cs=[0, 1], channel_max=CHANNEL_MAX):
  dat = h5py.File(f_n, 'r')['masked_mat']
  if cs is None:
    cs = np.arange(dat.shape[2])
  stacks = []
  for c in cs:
    patch_c = cv2.resize(np.array(dat[:, :, c]).astype(float), (224, 224))
    stacks.append(np.stack([patch_c] * 3, 2))
  
  x = np.stack(stacks, 0)
  x = x/np.array(channel_max).reshape((-1, 1, 1, 1))
  x = x * 255.
  return x

def predict(fs, 
            extractor, 
            preprocess_fn, 
            batch_size=128, 
            cs=[0, 1],
            channel_max=CHANNEL_MAX):
  temp_xs = []
  for ct, f_n in enumerate(fs):
    if ct > 0 and ct % 1000 == 0:
      print(ct)
    x = preprocess_fn(preprocess(f_n, cs=cs, channel_max=channel_max))
    temp_xs.append(x)
    if len(temp_xs) >= batch_size:
      temp_ys = extractor.predict(np.concatenate(temp_xs, 0))
      slice_num = len(temp_ys) // len(temp_xs)
      for i in range(len(temp_xs)):
        y = temp_ys[i*slice_num:(i+1)*slice_num]
        ys.append(y)
      temp_xs = []
  temp_ys = extractor.predict(np.concatenate(temp_xs, 0))
  slice_num = len(temp_ys) // len(temp_xs)
  for i in range(len(temp_xs)):
    y = temp_ys[i*slice_num:(i+1)*slice_num]
    ys.append(y)
  assert len(ys) == len(fs)
  return ys

if __name__ == '__main__':
  fs = read_file_path(DATA_ROOT + '/Data/StaticPatches')
  extractor, preprocess_fn = initiate_model()
  ys = predict(fs, extractor, preprocess_fn, cs=[0, 1], channel_max=CHANNEL_MAX)
  
  output = {}
  for f_n, y in zip(fs, ys):
    output[f_n] = y
  with open('/mnt/comp_micro/Projects/CellVAE/Data/EncodedResNet50.pkl', 'wb') as f:
    pickle.dump(output, f)
    
  
  

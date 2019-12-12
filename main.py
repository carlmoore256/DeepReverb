from comet_ml import Experiment
# import keras
# import os
# import numpy as np
# import matplotlib.pyplot as plt
import datetime
from base_model import BaseModel
from data_loader import DataLoader
from train_model import TrainModel
from evaluate_model import EvaluateModel
# experiment = Experiment('OKhPlin1BVQJFzniHu1f3K1t3')

# Hyperparameters
img_width, img_height = 128, 128
num_classes = 19
epochs = 5
batch_size = 16

ts =  datetime.datetime.now().timestamp()
train_data = './data/train/'
val_data = './data/val/'
test_data = './data/test/'
saved_model = './saved_models/saved_model-' + str(ts) + '.h5'
saved_weights = './saved_weights/saved_weight-' + str(ts) + '.h5'

save_model = True
save_weights = True

load_weights = False
load_weight_path = './saved_weights/saved_weight-1576127088.591156.h5'

train = TrainModel()

# returns a dataloader class containing the data streams
dataloader = train.BeginTraining(epochs, batch_size, img_width, img_height, num_classes)
eval =

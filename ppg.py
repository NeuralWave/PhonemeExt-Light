import os
from torch.utils.data import Dataset, DataLoader
import torch

import DatasetHandler
import Utils
import Model
import Params
import Transforms
import Train
import time

timit_dataset_train = DatasetHandler.PPGDataset(data_path='timit_train.torch')
dataload_train = DataLoader(timit_dataset_train, batch_size=Params.train_params.BZ, shuffle=Params.train_params.SHUFFLE, num_workers=Params.train_params.NUM_WORKERS,collate_fn=Transforms.PhnMelCollate(),pin_memory=False)

timit_dataset_test = DatasetHandler.PPGDataset(data_path='timit_test.torch',train=False)
dataload_test = DataLoader(timit_dataset_test, batch_size=Params.train_params.BZ, shuffle=Params.train_params.SHUFFLE, num_workers=Params.train_params.NUM_WORKERS,collate_fn=Transforms.PhnMelCollate(),pin_memory=False)

######################################################################################################

timit_trainer = Train.Trainer(Model.PRnet().cuda(),dataload_train=dataload_train,dataload_test=dataload_test)

start = time.time()
timit_trainer.train(epochs=100, test_interval=10, per_interval=20)
end = time.time()

print("Time = ", end-start, " s")

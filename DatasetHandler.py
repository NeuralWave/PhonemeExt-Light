import os
import torch
import pandas as pd
from os.path import exists
from torch.utils.data import Dataset, DataLoader
import torchaudio
import librosa.display
import matplotlib.pyplot as plt

import Utils
import Transforms
import Params

class PPGDataset(Dataset):

    def __init__(self, data_path, train=True,transform=None):
        self.data_path = data_path
        self.transform = transform
        self.train = train
        data_exist = exists(data_path)

        if not data_exist:
            print("generating default preprocessed data..")
            if train:
                timit_dataset = Utils.raw_db(root_dir="TIMIT/TRAIN",phonemes_list_path="phone_labels.txt",transform=Transforms.ToTorchMspec(Params.default),savename="timit_train")
            else:
                timit_dataset = Utils.raw_db(root_dir="TIMIT/TEST",phonemes_list_path="phone_labels.txt",transform=Transforms.ToTorchMspec(Params.default),savename="timit_test")
            Utils.torchsave(timit_dataset.name,raw_dataset=timit_dataset)
        self.audios, self.phones = torch.load(self.data_path)

    def __len__(self):
        return len(self.phones)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio = self.audios[idx]
        phones = self.phones[idx]
        sample = {'audio': audio, 'phones': phones}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def display(self,idx):
        print(self.phones[idx])
        librosa.display.specshow(self.audios[idx].numpy(), y_axis='mel', x_axis='time', sr=Params.default.SR, fmin=Params.default.FMIN, fmax=Params.default.FMAX, hop_length=Params.default.STFT_STEPSZ)
        plt.show()
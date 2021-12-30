import os
import torch
import pandas as pd
import numpy as np
from itertools import groupby
from torch.utils.data import Dataset, DataLoader
import torchaudio
from Levenshtein import distance as lev

class raw_db():

    def __init__(self, root_dir, phonemes_list_path, savename='data', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.audios_paths = []
        self.phonemes_paths = []
        self.name = savename

        self.phonemes_list_path = phonemes_list_path
        self.phonemes_list = pd.read_csv(self.phonemes_list_path,header=None)
        self.phonemes_list = self.phonemes_list.iloc[:,0]
        self.phonemes_list = self.phonemes_list.values.tolist()
        self.phonemes_list.insert(0,'-')

        self.words_paths = []
        for (root,dirs,files) in os.walk(root_dir):
            for files in files:
                if files.endswith(".WAV"):  
                    name = root+'/'+files
                    name = name[:-3]
                    self.audios_paths.append(name+"WAV")
                    self.phonemes_paths.append(name+"PHN")
                    self.words_paths.append(name+"WRD")

    def __len__(self):
        return len(self.audios_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio, sr = torchaudio.load(self.audios_paths[idx])
        phones_aux = pd.read_csv(self.phonemes_paths[idx],sep = " ",header=None)
        phones_aux = phones_aux.iloc[:,2]
        phones = []
        for p in phones_aux:
            phones.append(self.phonemes_list.index(p))
        sample = {'audio': audio,'sr': sr, 'phones': phones, 'path':self.audios_paths[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample

def torchsave(name,raw_dataset):

    mspecs = []
    phones = []
    for i in range(0,len(raw_dataset)):
        mspecs.append(raw_dataset[i]['audio'])
        x = raw_dataset[i]['phones']
        x = torch.tensor(x)
        phones.append(x)
    torch.save([mspecs,phones], name+'.torch')
    
def normalize_mpsec(mspec,nMelChannels):
    mspec.reshape(-1)
    mspec = (mspec - mspec.mean()) / mspec.std()
    mspec.reshape(nMelChannels, -1)
    mspec = torch.tensor(mspec)
    return mspec

def normalize_mpsec_positive(mspec,nMelChannels):
    mspec.reshape(-1)
    mspec = (mspec - mspec.min()) / mspec.std()
    mspec.reshape(nMelChannels, -1)
    mspec = torch.tensor(mspec)
    return mspec

def normalize_mpsec_maxdivide_shift(mspec,nMelChannels):
    mspec = mspec/100
    mspec.reshape(-1)
    mspec = mspec - mspec.min()
    mspec.reshape(nMelChannels, -1)
    mspec = torch.tensor(mspec)
    return mspec

def Delete_spaces_and_fold_phsec(plist,folding):
    plist = plist.numpy()
    plist = [elem for elem in plist if elem != 0 and elem != 16]    
    
    for idx in range(len(plist)):
        for equiv in folding:
            for idy in range(len(equiv)):
                if plist[idx] == equiv[idy]:
                    plist[idx] = equiv[0]

    plist = np.array([k for k,g in groupby(plist)])

    return list(plist)


def compute_per(ref, hyp, normalize=True):
    phone_set = set(ref + hyp)
    phone2char = dict(zip(phone_set, range(len(phone_set))))
    # Map phones to a single char array
    # NOTE: Levenshtein packages only accepts strings
    phones_ref = [chr(phone2char[p]) for p in ref]
    phones_hyp = [chr(phone2char[p]) for p in hyp]
    per = lev(''.join(phones_ref), ''.join(phones_hyp))
    if normalize:
        per /= len(ref)
    return per 


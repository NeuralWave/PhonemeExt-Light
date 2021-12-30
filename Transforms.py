import librosa
import numpy as np
import torch

import Params
import Utils

class ToTorchMspec(object):

    def __init__(self, params, mode=Params.normalization_modes.DEFAULT):
        self.nMelChannels = params.NMELCHANNELS
        self.sr = params.SR
        self.window = params.WINDOW
        self.stft_samples = params.STFT_SAMPLES
        self.stft_stepSz = params.STFT_STEPSZ
        self.fmin = params.FMIN
        self.fmax = params.FMAX
        self.mode = mode

    def __call__(self, sample):
        waveform = sample['audio'].numpy()
        y = waveform[0]
        y = librosa.resample(y, sample['sr'], self.sr)

        mspec = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.nMelChannels, fmin=self.fmin, fmax=self.fmax, n_fft=self.stft_samples, hop_length=self.stft_stepSz, window=self.window)
        mspec = librosa.power_to_db(mspec, np.max)

        if self.mode == "NONE":
            mspec = torch.tensor(mspec)
        if self.mode == "NORMALIZE":
            mspec = Utils.normalize_mpsec(mspec,self.nMelChannels)
        if self.mode == "NORMALIZE_POSITIVE":
            mspec = Utils.normalize_mpsec_positive(mspec,self.nMelChannels)
        if self.mode == "MAX_DIVIDE":
            mspec = Utils.normalize_mpsec_maxdivide_shift(mspec,self.nMelChannels)

        return {'audio': mspec,'sr': self.sr, 'phones': sample['phones'], 'path':sample['path']}

class PhnMelCollate():
    def __init__(self):
        self.sil = 0 

    def __call__(self, batch):
        input_lengths, ids_sorted_decreasing = torch.sort(torch.LongTensor([len(x['phones']) for x in batch]),dim=0, descending=True)
        max_input_len = input_lengths[0]

        phnSeq_padded = torch.LongTensor(len(batch), max_input_len)
        phnSeq_padded.fill_(self.sil)
        for i in range(len(ids_sorted_decreasing)):
            phnSeq = batch[ids_sorted_decreasing[i]]['phones']
            phnSeq_padded[i, :phnSeq.size(0)] = phnSeq


        num_mels = batch[0]['audio'].size(0)
        max_target_len = max([x['audio'].size(1) for x in batch])

        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()

        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]]['audio']
            mel_padded[i, :, :mel.size(1)] = mel
            output_lengths[i] = mel.size(1)

        return { 'phones': phnSeq_padded, 'phones_len' : input_lengths , 'audio': mel_padded, 'audio_len' : output_lengths}
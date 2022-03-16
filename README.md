# PhonemeExt-Light
Lightweight version of the phoneme extraction network

Lightweight version of the phoneme extraction network

This network implements the architecture shown in the paper [Towards End-to-End Speech Recognition with Deep Convolutional Neural Networks](https://arxiv.org/pdf/1701.02720.pdf) . It uses Convolutional Neural Networks together with Connectionist Temporal Classification for achieving 16.7% PER in the TIMIT TEST phoneme dataset at a low computational cost.

An implementation of the training process is implemented in `ppg.py`.

Note: The TIMIT TEST and TRAIN datasets should be put under their respective folders in `/TIMIT`.



# Tacotron-2-keras (Without Wavenet vocoder)

Keras implementations of Deep mind's Tacotron-2. A deep neural network architecture described in this paper: [Natural TTS synthesis by conditioning Wavenet on MEL spectogram predictions](https://arxiv.org/pdf/1712.05884.pdf)

## Current state:
- [ ] write a Keras inpimentation of Tacotron-2 *(in progress)*
- [ ] achieve a high quality human-like text to speech synthesizer based on DeepMind's paper
- [ ] achieve a high speed of training and work ofr multi-GPU systems.
- [ ] provide a pre-trained Tacotron-2 model
- [ ] provide compatibility with Mozilla [LPCNet](https://github.com/mozilla/LPCNet) project (Optional)

## Note:
Our preprocessing only supports Ljspeech and Ljspeech-like datasets (M-AILABS speech data)! If running on datasets stored differently, you will probably need to make your own preprocessing script.

## Model Architecture:

![Tacotron-2-model](https://camo.githubusercontent.com/7bdc61ffb468c3daf1af3b5cef2ccc16c3473cd9/68747470733a2f2f707265766965772e6962622e636f2f625538734c532f5461636f74726f6e5f325f4172636869746563747572652e706e67)

The model described by the authors can be divided in two parts:

 - Spectrogram prediction network
 - Vocoder (e.g. Wavenet vocoder)

To have an in-depth exploration of the model architecture, training procedure and preprocessing logic, refer to our [wiki](https://github.com/Stevel705/Tacotron-2-keras/wiki)

## Ussage:
0. Clone a repository
```
$ git clone https://github.com/Stevel705/Tacotron-2-keras.git
```
1. Download LJ-like dataset (e.g. [english Speech Dataset](https://keithito.com/LJ-Speech-Dataset/))
2. Extract dataset to `Tacotron-2-keras\data` folder
3. Run `$ python3 1_create_audio_dataset.py` to process an audio
4. Run `$ python3 2_create_text_dataset.py` to create a text data
5. Train tacotron `$ python3 3_train.py`
6. Test pretrained model `$ python3 4_test.py` *(optional)*
7. Synthesize mels and speech `$ python3 5_syntezer.py` *(in progress)*

## Lisense:
MIT Lisense

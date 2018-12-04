# gosha20777 code
import argparse
from multiprocessing import cpu_count
import os
from tqdm import tqdm
from processing import preprocessor
import hparams as hparams


def preprocess(args, input_folders, out_dir):
	mel_dir = os.path.join(out_dir, 'mels')
	wav_dir = os.path.join(out_dir, 'audio')
	linear_dir = os.path.join(out_dir, 'linear')
	os.makedirs(mel_dir, exist_ok=True)
	os.makedirs(wav_dir, exist_ok=True)
	os.makedirs(linear_dir, exist_ok=True)
	metadata = preprocessor.build_from_path(input_folders, mel_dir, linear_dir, wav_dir, args.n_jobs, tqdm=tqdm)
	write_metadata(metadata, out_dir)

def write_metadata(metadata, out_dir):
	with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
		for m in metadata:
			f.write('|'.join([str(x) for x in m]) + '\n')
	mel_frames = sum([int(m[4]) for m in metadata])
	timesteps = sum([int(m[3]) for m in metadata])
	sr = hparams.SAMPLING_RATE
	hours = timesteps / sr / 3600
	print('Write {} utterances, {} mel frames, {} audio timesteps, ({:.2f} hours)'.format(
		len(metadata), mel_frames, timesteps, hours))
	print('Max input length (text chars): {}'.format(max(len(m[5]) for m in metadata)))
	print('Max mel frames length: {}'.format(max(int(m[4]) for m in metadata)))
	print('Max audio timesteps length: {}'.format(max(m[3] for m in metadata)))

def norm_data(args):
	print('Selecting data folders..')
	supported_datasets = ['LJSpeech-1.1', 'M-AILABS']
	if args.dataset not in supported_datasets:
		raise ValueError('dataset value entered {} does not belong to supported datasets: {}'.format(
			args.dataset, supported_datasets))

	if args.dataset == 'LJSpeech-1.1':
		return [os.path.join(args.base_dir, args.dataset)]

	
	if args.dataset == 'M-AILABS':
		supported_languages = ['en_US', 'en_UK', 'fr_FR', 'it_IT', 'de_DE', 'es_ES', 'ru_RU', 
			'uk_UK', 'pl_PL', 'nl_NL', 'pt_PT', 'fi_FI', 'se_SE', 'tr_TR', 'ar_SA']
		if args.language not in supported_languages:
			raise ValueError('Please enter a supported language to use from M-AILABS dataset! \n{}'.format(
				supported_languages))

		supported_voices = ['female', 'male', 'mix']
		if args.voice not in supported_voices:
			raise ValueError('Please enter a supported voice option to use from M-AILABS dataset! \n{}'.format(
				supported_voices))

		path = os.path.join(args.base_dir, args.language, 'by_book', args.voice)
		supported_readers = [e for e in os.listdir(path) if 'DS_Store' not in e]
		if args.reader not in supported_readers:
			raise ValueError('Please enter a valid reader for your language and voice settings! \n{}'.format(
				supported_readers))

		path = os.path.join(path, args.reader)
		supported_books = [e for e in os.listdir(path) if e != '.DS_Store']

		if args.merge_books:
			return [os.path.join(path, book) for book in supported_books]

		else:
			if args.book not in supported_books:
				raise ValueError('Please enter a valid book for your reader settings! \n{}'.format(
					supported_books))

			return [os.path.join(path, args.book)]


def run_preprocess(args):
	input_folders = norm_data(args)
	output_folder = os.path.join(args.base_dir, args.output)

	preprocess(args, input_folders, output_folder)


def main():
	print('initializing preprocessing..')
	parser = argparse.ArgumentParser()
	parser.add_argument('--base_dir', default='data/')
	parser.add_argument('--dataset', default='LJSpeech-1.1')
	parser.add_argument('--language', default='en_US')
	parser.add_argument('--voice', default='female')
	parser.add_argument('--reader', default='mary_ann')
	parser.add_argument('--merge_books', type=bool, default=False)
	parser.add_argument('--book', default='northandsouth')
	parser.add_argument('--output', default='training_data')
	parser.add_argument('--n_jobs', type=int, default=cpu_count())
	args = parser.parse_args()

	run_preprocess(args)


if __name__ == '__main__':
	main()

# Stvel705 code
'''
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from tqdm import tqdm
from processing.proc_audio import get_padded_spectros
from hparams import *
import tensorflow as tf
sess = tf.Session()

print('Loading the data...')
metadata = pd.read_csv('data/LJSpeech-1.1/metadata.csv',
                       dtype='object', quoting=3, sep='|', header=None)

# metadata = metadata.iloc[:500]

# audio filenames
dot_wav_filenames = metadata[0].values


mel_spectro_data = []
spectro_data = []
decoder_input = []
print('Processing the audio samples (computation of spectrograms)...')
for filename in tqdm(dot_wav_filenames):
    file_path = 'data/LJSpeech-1.1/wavs/' + filename + '.wav'
    fname, mel_spectro, spectro = get_padded_spectros(file_path, r,
                                                      PREEMPHASIS, N_FFT,
                                                      HOP_LENGTH, WIN_LENGTH,
                                                      SAMPLING_RATE,
                                                      N_MEL, REF_DB,
                                                      MAX_DB)

    decod_inp_tensor = tf.concat((tf.zeros_like(mel_spectro[:1, :]),
                                  mel_spectro[:-1, :]), 0)
    decod_inp = sess.run(decod_inp_tensor)
    decod_inp = decod_inp[:, -N_MEL:]

    # Padding of the temporal dimension
    dim0_mel_spectro = mel_spectro.shape[0]
    dim1_mel_spectro = mel_spectro.shape[1]
    padded_mel_spectro = np.zeros((MAX_MEL_TIME_LENGTH, dim1_mel_spectro))
    padded_mel_spectro[:dim0_mel_spectro, :dim1_mel_spectro] = mel_spectro

    dim0_decod_inp = decod_inp.shape[0]
    dim1_decod_inp = decod_inp.shape[1]
    padded_decod_input = np.zeros((MAX_MEL_TIME_LENGTH, dim1_decod_inp))
    padded_decod_input[:dim0_decod_inp, :dim1_decod_inp] = decod_inp

    dim0_spectro = spectro.shape[0]
    dim1_spectro = spectro.shape[1]
    padded_spectro = np.zeros((MAX_MAG_TIME_LENGTH, dim1_spectro))
    padded_spectro[:dim0_spectro, :dim1_spectro] = spectro

    mel_spectro_data.append(padded_mel_spectro)
    spectro_data.append(padded_spectro)
    decoder_input.append(padded_decod_input)


print('Convert into np.array')
decoder_input_array = np.array(decoder_input)
mel_spectro_data_array = np.array(mel_spectro_data)
spectro_data_array = np.array(spectro_data)

print('Split into training and testing data')
len_train = int(TRAIN_SET_RATIO * len(metadata))

decoder_input_array_training = decoder_input_array[:len_train]
decoder_input_array_testing = decoder_input_array[len_train:]

mel_spectro_data_array_training = mel_spectro_data_array[:len_train]
mel_spectro_data_array_testing = mel_spectro_data_array[len_train:]

spectro_data_array_training = spectro_data_array[:len_train]
spectro_data_array_testing = spectro_data_array[len_train:]


print('Save data as pkl')
joblib.dump(decoder_input_array_training,
            'data/decoder_input_training.pkl')
joblib.dump(mel_spectro_data_array_training,
            'data/mel_spectro_training.pkl')
joblib.dump(spectro_data_array_training,
            'data/spectro_training.pkl')

joblib.dump(decoder_input_array_testing,
            'data/decoder_input_testing.pkl')
joblib.dump(mel_spectro_data_array_testing,
            'data/mel_spectro_testing.pkl')
joblib.dump(spectro_data_array_testing,
            'data/spectro_testing.pkl')
'''
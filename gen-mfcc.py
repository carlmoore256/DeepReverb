import os.path
import librosa
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
from scipy.io import wavfile
import soundfile as sf
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--recursive", help="process all files in root directory", default=False)
parser.add_argument("--directory", help="audio (wav) file directory", required=True)
parser.add_argument("--label", help="class label, such as 'guitar' or 'tuba'", required=True)
parser.add_argument("--numfilt", help="number of bark filters", type=int, default=26)
args = parser.parse_args()

# parser = argparse.ArgumentParser()
# parser.add_argument("square", type=int,
#                     help="display a square of a given number")
# args = parser.parse_args()
# answer = args.square**2
# print(answer)

numfilt = args.numfilt
iterate = True

label = args.label
SAMPLE_FOLDER = args.directory + '/' + label + '/'

files_in_path = [f for f in listdir(SAMPLE_FOLDER) if '._' not in f and isfile(join(SAMPLE_FOLDER, f))]

train_data_dir = './data/train/' + label + '/'
test_data_dir = './data/test/' + label + '/'
val_data_dir = './data/val/' + label + '/'

this_dir = train_data_dir
if not os.path.exists('./data/train/' + label + '/'):
    os.makedirs('./data/train/' + label + '/')

numiter = 0

if iterate:
  numiter = len(files_in_path)
else:
  numiter = 1;

for i in range(numiter):

    audio, sample_rate = sf.read(SAMPLE_FOLDER + files_in_path[i])
    filename, file_ext = os.path.splitext(files_in_path[i])
    # audio = audio.astype(float)
    def normalize_audio(audio):
      audio = audio / np.max(np.abs(audio))
      return audio

    print('creating mfcc image of: ' + files_in_path[i])

    mfccs = librosa.feature.mfcc(audio, sr=sample_rate, n_mfcc=numfilt, norm='ortho')

    mfccs = mfccs * 0.1
    # print(filename)
    # mfccs = (mfccs + 1) * 0.75
    # mfccs = np.power(mfccs,2)
    # max = np.max(mfccs)
    # if max > 1:
    #     print('clipped value ' + str(max))
    # mfccs = (mfccs - np.min(mfccs)) / (np.max(mfccs) - np.min(mfccs))
    # mfccs = np.power(mfccs,5)
    cm = plt.get_cmap('inferno')
    colored_image = cm(mfccs)
    Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8)).save(train_data_dir + filename + '.png')

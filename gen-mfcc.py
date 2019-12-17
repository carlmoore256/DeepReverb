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
parser.add_argument("--input_directory", help="input audio file directory", required=True)
parser.add_argument("--split", help="output destination (train, val, test)", required=True)
parser.add_argument("--label", help="class label, such as 'guitar' or 'tuba'", required=True)
parser.add_argument("--numfilt", help="number of bark filters", type=int, default=26)
parser.add_argument("--split_block", help="split image into uniform boxes, dispose excess", default=False, required=False)
args = parser.parse_args()

numfilt = args.numfilt
iterate = True

label = args.label
SAMPLE_FOLDER = args.input_directory + '/' + label + '/'
split_block = args.split_block

files_in_path = [f for f in listdir(SAMPLE_FOLDER) if '._' not in f and isfile(join(SAMPLE_FOLDER, f))]

train_data_dir = './data/train/' + label + '/'
val_data_dir = './data/val/' + label + '/'
test_data_dir = './data/test/' + label + '/'

if args.split == 'train':
    output_dir = train_data_dir
elif args.split == 'val':
    output_dir = val_data_dir
elif args.split == 'test':
    output_dir = test_data_dir
else:
    print('splits directory not properly specified, enter "train", "val", or "test"')
    quit()

if not os.path.exists(output_dir + label + '/'):
    os.makedirs(output_dir + label + '/')

numiter = 0

if iterate:
  numiter = len(files_in_path)
else:
  numiter = 1;

for i in range(numiter):

    audio, sample_rate = sf.read(SAMPLE_FOLDER + files_in_path[i])
    filename, file_ext = os.path.splitext(files_in_path[i])

    # audio = audio.astype(float)
    arr_size = audio.ndim
    if arr_size > 1:
        audio = np.add(audio[:,0], audio[:,1])
        audio = np.asfortranarray(audio)

    audio = audio / np.max(np.abs(audio))

    print('creating mfcc image of: ' + files_in_path[i])
    mfccs = librosa.feature.mfcc(audio, sr=sample_rate, n_mfcc=numfilt, norm='ortho')
    mfccs = mfccs * 0.1
    cm = plt.get_cmap('inferno')
    colored_image = cm(mfccs)
    output_img = Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8))

    if split_block:
        size = mfccs.shape
        num_blocks = int(np.floor(size[1]/size[0])-1)
        for j in range(num_blocks):
            box = (j*numfilt, 0, (j+1)*numfilt, numfilt)
            sub_image = output_img.crop(box)
            sub_image.save(output_dir + filename + 'sub-img' + str(j) + '.png')

        print('saved ' + str(num_blocks) + ' mfcc images to: ' + output_dir + filename + 'sub-img' + str(j) + '.png')
    else:
        output_img.save(output_dir + filename + '.png')
        print('saved mfcc image to: ' + output_dir + filename + 'sub-img' + str(j) + '.png')

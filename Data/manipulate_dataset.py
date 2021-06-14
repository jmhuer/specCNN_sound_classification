#this code takes the 10 folds and clips (>2 in length) and clips them to 2 seconds

import os
import soundfile as sf
import librosa
import csv
import torchaudio
from .collect import get_cnt

folder = ["fold1","fold2","fold3","fold4","fold5","fold6","fold8", "fold9", "fold10"]
base = "Data/UrbanSound8K/audio/"
target = "Data/Audio/"
cnt2 = 0
import numpy
import torch.utils.tensorboard as tb2
import tensorboardX as tb
from os import path

for f in folder:
    for filename in os.listdir(base + f ):    # Load MIDI file into PrettyMIDI object
        if filename!=".DS_Store":
            full_path = base + f + "/" + filename

            data, samplerate = librosa.load(full_path, sr=8000)
            two_seconds_length = int(2/(1/samplerate))

            if len(data) >= two_seconds_length:
                data = data[0:two_seconds_length]
                time = (1/samplerate)*len(data)

                print(numpy.ndarray.min(data))
                print(numpy.ndarray.max(data))

                train_logger = tb.SummaryWriter(path.join('logs', 'train'), flush_secs=1)

                print("cnt: {} \t name: {} \t sample_rate: {} \t length: {} \t".format(cnt2, full_path,samplerate,time))

                if time>=2.0:cnt2 += 1

                # label  = classes[int(filename[filename.find('-') + 1])]
                label = int(filename[filename.find('-') + 1])

                name = target + str(label) + "_" + str(get_cnt(target, label)) + ".wav"
                datas = []
                datas.append([name, label])
                with open(target + "labels.csv", "a") as file:
                    writer = csv.writer(file)
                    writer.writerows(datas)

                ##add label


                # waveform, sample_rate = torchaudio.load('audio/518-4-0-1.wav')
                #
                # reconstructed = waveform.view(128, 125)
                # print(reconstructed.size())
                # exit()

                librosa.output.write_wav(name , data, samplerate, norm=False)
            #
            # exit()
            # # sf.write('new_file.wav', data, samplerate)



print("time2: ",cnt2)
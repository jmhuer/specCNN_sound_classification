from __future__ import print_function

from Model import tensor_transform
from Model import load_data
import torchaudio
import numpy as np
from scipy.io import wavfile

from .microphone_stream import MicrophoneStream


import torch


RATE = 16000
CHUNK = int(RATE * 3.0)

if __name__ == '__main__':


    # window = [0.5] * FLAGS.avg_window

    with MicrophoneStream(RATE, CHUNK) as stream:

        audio_generator = stream.generator()
        for chunk in audio_generator:
            print("recording in progress...")

            try:
                arr = np.frombuffer(chunk, dtype=np.int16)

                #f = tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir="nothing")
                wavfile.write('live_data/new.wav', RATE, arr)

            except (KeyboardInterrupt, SystemExit):
                print('Shutting Down -- closing file')


            PATH = './model1.pth'
            net = v.to('cpu')
            net.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))

            waveform, sample_rate = torchaudio.load(filepath='live_data/new.wav')
            spectogram = (tensor_transform['valid'])(waveform)
            # spectogram = spectogram.repeat(1, 3, 1, 1)

            pred = model(spectogram)
            final_pred = torch.argmax(pred, dim = 1).float()
            if final_pred == 0:
                print('Child')
            else:
                print('Adult')
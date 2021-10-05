import cv2
from Model import load_weights, image_transforms, tensor_transform, MyResNet, CNNClassifier, newest_model
from PIL import Image
import numpy as np
import time
from Model import classes
from Model import tensor_transform
import torchaudio
import numpy as np
from scipy.io import wavfile
from .microphone_stream import MicrophoneStream


class running_majority:
    '''
    this class takes predictions and outputs running majority prediction
    '''

    class TopNHeap:
        """
        A heap that keeps the top N elements around
        """

        def __init__(self, N):
            self.elements = []
            self.N = N

        def add(self, e):
            from heapq import heappush, heapreplace
            if len(self.elements) < self.N:
                heappush(self.elements, e)
            elif self.elements[0] < e:
                heapreplace(self.elements, e)

    def __init__(self, frame_window=10):
        global classes
        self.h = running_majority.TopNHeap(frame_window)
        self.word_counter = {}
        for c in classes:
            self.word_counter[c] = 0

    def add(self, pred_class):
        self.h.add((float(time.time()), pred_class))

    def predict(self):
        words = [h[1] for h in self.h.elements]
        return max(set(words), key=words.count)


def inference(model, wav):
    '''image pre_processing'''
    image_t = tensor_transform['valid'](wav)
    # torchvision.utils.save_image(image_t, "test1.jpg") ##weird this messes things up
    p = model.predict(image_t)
    return p


def run_live_inference(model, camera_source):
    RATE = 16000
    CHUNK = int(RATE * 3.0)
    majority = running_majority(frame_window=3)

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        for chunk in audio_generator:
            print("recording in progress...")
            try:
                arr = np.frombuffer(chunk, dtype=np.int16)

                #f = tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir="nothing")
                wavfile.write('Inference/live_data/new.wav', RATE, arr)

            except (KeyboardInterrupt, SystemExit):
                print('Shutting Down -- closing file')


            waveform, sample_rate = torchaudio.load(filepath='Inference/live_data/new.wav')

            pred = inference(model, waveform)
            majority.add(pred)
            pred = majority.predict()
            print("Detected class: ", pred)

if __name__ == '__main__':
    import argparse

    '''
    We default to most recent model created
    '''
    default_model_path = newest_model()

    parser = argparse.ArgumentParser()
    # Put custom arguments here
    parser.add_argument('-s', '--mic_source', type=int, default=0)
    parser.add_argument('-m', '--model_path', type=str, default=default_model_path)
    args = parser.parse_args()

    print('Using model path: {}'.format(args.model_path))
    model = MyResNet()
    model = load_weights(model, args.model_path)

    run_live_inference(model, args.mic_source)

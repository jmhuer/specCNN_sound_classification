from __future__ import print_function
import numpy as np
import tempfile
from scipy.io import wavfile

from laugh_detector.microphone_stream import MicrophoneStream

import sys

sys.path.append('./audio')

#
# flags = tf.app.flags
#
# flags.DEFINE_float(
#     'sample_length', 3.0,
#     'Length of audio sample to process in each chunk'
# )
#
# flags.DEFINE_integer(
#     'avg_window', 10,
#     'Size of window for running mean on output'
# )
#
# flags.DEFINE_string(
#     'recording_directory', 'rec',
#     'Directory where recorded samples will be saved'
#     'If None, samples will not be saved'
# )

# FLAGS = flags.FLAGS

RATE = 16000
CHUNK = int(RATE * 3.0)  # 3 sec chunks



if __name__ == '__main__':

    # window = [0.5] * FLAGS.avg_window

    with MicrophoneStream(RATE, CHUNK) as stream:

        audio_generator = stream.generator()
        for chunk in audio_generator:
            print("recording in progress...")

            try:
                arr = np.frombuffer(chunk, dtype=np.int16)
                f = tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir="nothing")
                wavfile.write(f, RATE, arr)

            except (KeyboardInterrupt, SystemExit):
                print('Shutting Down -- closing file')


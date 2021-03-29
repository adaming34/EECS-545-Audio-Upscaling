import sys

import sounddevice as sd
import soundfile as sf

import numpy as np


from ring_buffer import RingBuffer

class DataRecorder(object):
    def __init__(self, **kwargs):
        self.stream = sd.InputStream(callback=self._cb, **kwargs)
        
        # Buffer up to 5 seconds of sound idk
        self.buf = RingBuffer(
            capacity=int(self.stream.samplerate*5),
            cols=int(self.stream.channels),
            allow_overwrite=False)
    
    def _cb(self, data, n, t, status):
        if status:
            print(status, file=sys.stderr)
        self.buf.extend(data)
    
    # Returns a block of zeros which pads the input to t using the latency
    def get_pad_to(self, t):
        dur = t - self.stream.latency
        return np.zeros((int(dur*self.stream.samplerate), self.stream.channels))





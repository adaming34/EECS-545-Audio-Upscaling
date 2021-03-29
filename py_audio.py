#!/usr/bin/env python3
"""Create a recording with arbitrary duration.

The soundfile module (https://PySoundFile.readthedocs.io/) has to be installed!

"""
import argparse
import tempfile
import sys
from contextlib import ExitStack

import sounddevice as sd
import soundfile as sf
import numpy  # Make sure NumPy is loaded before it is used in the callback
assert numpy  # avoid "imported but unused" message (W0611)

import recorder

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    'filenames', nargs='*',
    help='audio file to store recording to')
parser.add_argument(
    '-d', '--devices', type=int_or_str, nargs='*', metavar='D',
    help='input devices (numeric ID or substring)')
parser.add_argument(
    '-r', '--samplerate', type=int, help='sampling rate')
parser.add_argument(
    '-c', '--channels', type=int, default=1, help='number of input channels')
parser.add_argument(
    '-t', '--subtype', type=str, help='sound file subtype (e.g. "PCM_24")')
args = parser.parse_args(remaining)

try:
    if not args.devices:
        args.devices = [sd.default.device[0]]
    if args.samplerate is None:
        device_info = sd.query_devices(args.devices[0], 'input')
        # soundfile expects an int, sounddevice provides a float:
        args.samplerate = int(device_info['default_samplerate'])
    if not args.filenames:
        args.filenames = [f'{d}.wav' for d in args.devices]

    files = [sf.SoundFile(fname, mode='w', samplerate=args.samplerate, channels=args.channels, subtype=args.subtype) for fname in args.filenames]
    
    
    recs = [recorder.DataRecorder(device=d, samplerate=args.samplerate, channels=args.channels) for d in args.devices]
    
    with ExitStack() as stack:
        for r, f in zip(recs, files):
            stack.enter_context(r.stream)
            stack.enter_context(f)
            f.write(r.get_pad_to(1.0)) # Sync all recordings at 1.0s
        
        print('#' * 80)
        print('press Ctrl+C to stop the recording')
        print('#' * 80)
        
        while True:
            for r, f in zip(recs, files):
                if len(r.buf):
                    f.write(r.buf.popall())
except KeyboardInterrupt:
    print('\nRecordings finished: ' + repr(args.filenames))
    parser.exit(0)
    

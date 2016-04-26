'''
Author: Joao Nuno Carvalho
email:  joaonunocarv ( __AT__ ) gmail.com
Description: This Pyhton 3.5 program is a real-time implementation of simultaneous 4 string guitar simulation with
             the algorithm of Karplus Strong. This program uses PyAudio lib and NumPy.
             In Windows I used the Anaconda 3.5 distribution, see PyAudio page to learn how to install PyAudio.

            For Karplus-Strong details see the paper:
            Digital Synthesis of Plucked-String and Drum Timbres
            Kevin Karplus and Alex Strong
            http://www.jstor.org/stable/3680062?seq=1#page_scan_tab_contents


The MIT License (MIT)

Copyright (c) 2016 Joao Nuno Carvalho

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of
the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import pyaudio
import time
import numpy as np
import wave


WIDTH = 2
CHANNELS = 2
RATE = 44100

p = pyaudio.PyAudio()

number_of_strings = 4
# Initialize the 4 flag_plunk to False.
flag_plunk = [False, False, False, False]
# Initialize the 4 Karplus String buffer sizes each with a different size for a different string tone.
buffer_ks_size = [200, 210, 220, 230]

buffer_ks = []
for string in range(0, number_of_strings):
    buffer_ks.append( np.zeros(buffer_ks_size[string], dtype=np.float32) )

# Initialize the array with a plunk of random numbers.

#buffer_ks_plunk = np.random.uniform(-1, 1, buffer_ks_size)
buffer_ks_plunk = []
for string in range(0, number_of_strings):
    buffer_ks_plunk.append( np.random.uniform(-1, 1, buffer_ks_size[string]) )

ptr_out = [1, 1, 1, 1] # pointer to karplus_strong 4 buffer.
ptr_in  = [0, 0, 0, 0] # pointer to   "       "    "  "

factor  =  0.499 # 0.495 # Decaying factor:  0 < factor <= 0.5

def plunk_the_string(string_number):
    global buffer_ks, buffer_ks_plunk
    buffer_ks[string_number] = np.copy(buffer_ks_plunk[string_number])

def copy_buffer_ks_to_buffer_output(result, frame_count):
    global buffer_ks, ptr_in, ptr_out, number_of_strings
    for i in range(0, frame_count):
        # Left channel.
        result[i, 0] = 0.0
        for string in range(0, number_of_strings):
            result[i, 0] += buffer_ks[string][ptr_in[string]]
        result[i, 0] /= number_of_strings
        # Righ channel.
        result[i, 1] = result[i, 0]  # Copy the left channel value to the right channel.

        for string in range(0, number_of_strings):
            buffer_ks[string][ptr_in[string]] = factor * ( buffer_ks[string][ptr_in[string]] + buffer_ks[string][ptr_out[string]] )

        # Update the global pointers of the circukar buffer.
        for string in range(0, number_of_strings):
            if ptr_in[string] < buffer_ks_size[string] - 1:
                ptr_in[string] += 1
            else:
                ptr_in[string] = 0
            if ptr_out[string] < buffer_ks_size[string] - 1:
                ptr_out[string] += 1
            else:
                ptr_out[string] = 0

    return result

frame_count_global = 0

frames_to_file = []

result = np.zeros((1024,2), dtype=np.float32)


chunk_length = 0

# To access the left channel  result[:, 0], and to access right channel result[:, 1]
def decode(in_data, channels):
    """
    Convert the byte stream of interleaving (and incoming) [L0, R0, L1, R1, ...]
    into a matrix with each row for each channel left channel
    of [L0, L1, L2, ...] and right channel of [R0, R1, R2, ...]
    This is done by the reshape(chunk_size, channels) instruction.
    """
    result = np.fromstring(in_data, dtype=np.float32)
    chunk_length = len(result) / channels
    assert chunk_length == int(chunk_length)
    result = np.reshape(result, (chunk_length, channels))
    return result

def encode(signal):
    # Convert a 2D numpy array into a byte stream for PyAudio.
    # Signal has chunk_size rows and channels columns.
    interleaved = signal.flatten()
    out_data = interleaved.astype(np.float32).tostring()
    return out_data

def encode_int16(signal):
    # Convert a 2D numpy array into a byte stream for PyAudio
    # Signal has chunk_size rows and channels columns.
    # The output is scalled to a int16 value not a -1 to 1 value.
    interleaved = signal.flatten()
    interleaved = interleaved * ((2**15) - 1)
    out_data = interleaved.astype(np.int16).tostring()
    return out_data

def callback(in_data, frame_count, time_info, flag):
    global flag_plunk, b, a, result, number_of_strings, frame_count_global, frames_to_file #global variables for filter coefficients and array

    frame_count_global = frame_count
    for string in range(number_of_strings):
        if flag_plunk[string] == True:
            flag_plunk[string] = False
            plunk_the_string(string)

    result = copy_buffer_ks_to_buffer_output(result, frame_count)
    out_data = encode(result)

    frames_to_file.append(encode_int16(result))
    return (out_data, pyaudio.paContinue)

def _find_getch():
    """
    This function is used to obtain the correct function (Windows/Linux) to get a single char from
    the stdin input stream. It will capature the keystrokes of each key for each string of the guitar.
    """
    try:
        import termios
    except ImportError:
        # Non-POSIX. Return msvcrt's (Windows') getch.
        import msvcrt
        return msvcrt.getch

    # POSIX system. Create and return a getch that manipulates the tty.
    import sys, tty
    def _getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    return _getch

getch = _find_getch()

stream = p.open(format=pyaudio.paFloat32,
                channels=CHANNELS,
                rate=RATE,
                output=True,
                input=False, # True
                stream_callback=callback)

stream.start_stream()

print('frame_count inside callback:', frame_count_global )
print('Press q, w, e, r to plunk a stering (p to quit): ')
while stream.is_active():
    # time.sleep(0.1) # 0.5
    # ch = input("Press q, w, e, r to plunk a stering (p to quit): ")
    ch = getch()
    ch = ch.decode('ASCII')

    if ch == 'q':
        flag_plunk[0] = True
    if ch == 'w':
        flag_plunk[1] = True
    if ch == 'e':
        flag_plunk[2] = True
    if ch == 'r':
        flag_plunk[3] = True
    if ch == 'p':
        # Exit and write to file.
        stream.stop_stream()
        time.sleep(1)

stream.stop_stream()
stream.close()

p.terminate()


# Save the output to WAV file.
WAVE_OUTPUT_FILENAME = "Guitar_output.wav"
#FORMAT = pyaudio.paFloat32
FORMAT = pyaudio.paInt16

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames_to_file))
wf.close()

print('Session saved to WAV file ... Guitar_output.wav')
#print('Format:', p.get_sample_size(FORMAT))

# This code includes source from 
# https://www.github.com/carykh/jumpcutter 
# under the MIT License:

# MIT License

# Copyright (c) 2019 carykh

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import argparse
from genericpath import isfile
import re
import os
import json
import glob
import time
import subprocess
from typing import List
import numpy as np
from scipy.io import wavfile
from fractions import Fraction
from colorama import Fore, Style
from logging import error, warning
from pathlib import Path
from tqdm import tqdm
from math import ceil, floor

def createTemp(dirname='./TEMP'):
    try:
        os.makedirs(dirname)
    except FileExistsError:
        warning(f'{Fore.YELLOW}Directory already exists.{Fore.RESET}')
    except OSError:
        error(f'{Fore.RED}An unexpected error occured while creating the directory.{Fore.RESET}')
    return dirname

def getMaxVolume(s):
    maxv = float(np.max(s))
    minv = float(np.min(s))
    return max(maxv,-minv)

def getWAVinfo(filename:str, stderr:bool=False):
    pipe = subprocess.run(f'ffprobe -v quiet -print_format json -show_format -show_streams {filename}',
                          stdout = subprocess.PIPE,
                          stderr = subprocess.PIPE,
                          bufsize=10**8)
    
    if stderr:
        return (pipe.stdout.decode('utf-8'), pipe.stderr.decode('utf-8'))
    else:
        return json.loads(pipe.stdout.decode('utf-8'))

def getFPS(info:dict) -> float:
    framerate = info['streams'][0]['r_frame_rate']
    framerate = Fraction(framerate).limit_denominator(100)
    framerate = float(framerate)
    assert framerate <= 60.0
    return framerate

def getSR(info:dict) -> int:
    sample_rate = info['streams'][1]['sample_rate']
    sample_rate = int(sample_rate)
    assert sample_rate <= 44100
    return sample_rate

def extractAudio(input_file:str, sample_rate:int, temp_folder='./TEMP', output_file='audio.wav'):
    fulldir = os.path.join(temp_folder, output_file)
    fulldir = Path(fulldir)
    if fulldir.is_file():
        warning(f'{Fore.YELLOW}File {fulldir.name} already exists. Skipping.{Fore.RESET}')
        return fulldir
    
    command = "ffmpeg -i "+input_file+" -ab 160k -ac 2 -ar "+str(sample_rate)+" -vn "+temp_folder+f"/{output_file}"
    pipe = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize = 10**8)
    return fulldir

def getTimeFromArray(arr, the_spf, the_sr):
    # Amount of Frames * samples per Frame / sample rate / 60
    return np.around(len(arr) * the_spf / the_sr / 60, 3)

def chunkify(arr):
    arr = list(arr)
    output = []
    partial = []

    for i,element in enumerate(arr):
        if i == len(arr)-1: 
            next_element = None;
        else:
            next_element = arr[i+1];

        partial.append(element)
        if element != next_element:
            output.append(partial)
            partial = []

    return output

def compress_chunks(chunks):
    output = []
    compressed_chunk = None
    for chunk in chunks:
        compressed_chunk = (chunk[0], len(chunk))
        output.append(compressed_chunk)
    return output

def printGraphics(total_length:int, sequence:list):
    pixels = total_length
    upscalling_factor = len(sequence)
    print()
    for i in range(pixels):
        upscaled_index = int (i * (upscalling_factor / pixels))
        k = sequence[upscaled_index]
        if k == 1.0:
            print(f'{Fore.GREEN}',end='')
        else:
            print(f'{Fore.RED}',end='')
        print('█',end='')

def process_video(filename:str, silent_threshold=0.03, frame_margin=1, frame_quality=3):
    TEMP_FOLDER = createTemp('./TEMP')
    info = getWAVinfo(filename)
    FRAME_RATE = getFPS(info)
    SAMPLE_RATE = getSR(info)

    audio_file = extractAudio(filename, SAMPLE_RATE, temp_folder=TEMP_FOLDER, output_file='audio.wav')

    sampleRate, audioData = wavfile.read(str(audio_file))

    if sampleRate != SAMPLE_RATE:
        warning(f'{Fore.YELLOW}Provided sample rate ({SAMPLE_RATE}) doesn\'t match with read sample rate ({sampleRate}). Using the read one.{Fore.WHITE}')
    else:
        del sampleRate

    audioSampleCount = audioData.shape[0]
    maxAudioVolume = getMaxVolume(audioData)

    samplesPerFrame = SAMPLE_RATE/FRAME_RATE
    audioFrameCount = int(ceil(audioSampleCount/samplesPerFrame))
    hasLoudAudio = np.zeros((audioFrameCount))

    cbar = tqdm(range(audioFrameCount), unit=' chunk', unit_divisor=1000, unit_scale=True)
    cbar.set_description_str('Processing chunks')
    cbar.colour = 'cyan'

    for chunk in cbar:
        upscaled_index = int(chunk*samplesPerFrame)
        start = upscaled_index
        end = min(upscaled_index + int(samplesPerFrame), audioSampleCount)
        audiochunks = audioData[start:end]
        maxchunksVolume = float(getMaxVolume(audiochunks)) / maxAudioVolume

        if maxchunksVolume >= silent_threshold:
            hasLoudAudio[chunk] = 1

    cbar = tqdm(range(audioFrameCount), unit=' frame', unit_divisor=1000, unit_scale=True)
    cbar.set_description_str('Processing frame spreadage')
    cbar.colour = 'cyan'

    shouldIncludeFrame = np.zeros((audioFrameCount))
    for frame in cbar:
        start = int(max(0,frame-frame_margin))
        end = int(min(audioFrameCount,frame+1+frame_margin))
        shouldIncludeFrame[frame] = np.max(hasLoudAudio[start:end])

    included_frames = np.where(shouldIncludeFrame == 1)[0]

    new_time = getTimeFromArray(included_frames, samplesPerFrame, SAMPLE_RATE)
    total = getTimeFromArray(shouldIncludeFrame, samplesPerFrame, SAMPLE_RATE)
    percentage = np.around(new_time * 100 / total, 2)
    difference = np.around(total - new_time, 3)
    # TODO: Save compressed chunks

    #print(f'jumpcutted = {new_time} min | total = {total} min | percentage = {percentage}% | difference = -{difference} min')
    print(f'{Style.BRIGHT}Input time = {total} minutes')
    print(f'Skippy\'d = {new_time} minutes. {percentage}% of total duration. Reduced by {difference} minutes.{Style.RESET_ALL}{Fore.RESET}')

    # TODO: Make printGraphics() use compressed chunks
    printGraphics(100, shouldIncludeFrame)

    print('\n')
    print(f'{Style.RESET_ALL}{Fore.RESET}Proceed? [Y/N]', end=' ')
    choice = input()

    if choice == 'n' or choice == 'N':
        try:
            os.remove(str(audio_file.absolute()))
            os.removedirs('./TEMP')
        except OSError:
            pass

    elif choice == 'y' or choice == 'Y':
        # TODO: extractFrames() ffmpeg -i "+INPUT_FILE+" -qscale:v "+str(FRAME_QUALITY)+" "+TEMP_FOLDER+"/frame%06d.jpg -hide_banner
        
        # TODO: ⮦ This code ⮧

        # chunks = [[0,0,0]]
        # shouldIncludeFrame = np.zeros((audioFrameCount))
        # for i in range(audioFrameCount):
        #     if (i >= 1 and shouldIncludeFrame[i] != shouldIncludeFrame[i-1]): # Did we flip?
        #         chunks.append([chunks[-1][1],i,shouldIncludeFrame[i-1]])

        # chunks.append([chunks[-1][1],audioFrameCount,shouldIncludeFrame[i-1]])
        # chunks = chunks[1:]

        # outputAudioData = np.zeros((0,audioData.shape[1]))
        # outputPointer = 0

        # lastExistingFrame = None
        # for chunk in chunks:
        #     audioChunk = audioData[int(chunk[0]*samplesPerFrame):int(chunk[1]*samplesPerFrame)]
            
        #     sFile = TEMP_FOLDER+"/tempStart.wav"
        #     eFile = TEMP_FOLDER+"/tempEnd.wav"
        #     wavfile.write(sFile,SAMPLE_RATE,audioChunk)
        #     with WavReader(sFile) as reader:
        #         with WavWriter(eFile, reader.channels, reader.samplerate) as writer:
        #             tsm = phasevocoder(reader.channels, speed=NEW_SPEED[int(chunk[2])])
        #             tsm.run(reader, writer)
        #     _, alteredAudioData = wavfile.read(eFile)
        #     leng = alteredAudioData.shape[0]
        #     endPointer = outputPointer+leng
        #     outputAudioData = np.concatenate((outputAudioData,alteredAudioData/maxAudioVolume))

        #     #outputAudioData[outputPointer:endPointer] = alteredAudioData/maxAudioVolume

        #     # smooth out transitiion's audio by quickly fading in/out
            
        #     if leng < AUDIO_FADE_ENVELOPE_SIZE:
        #         outputAudioData[outputPointer:endPointer] = 0 # audio is less than 0.01 sec, let's just remove it.
        #     else:
        #         premask = np.arange(AUDIO_FADE_ENVELOPE_SIZE)/AUDIO_FADE_ENVELOPE_SIZE
        #         mask = np.repeat(premask[:, np.newaxis],2,axis=1) # make the fade-envelope mask stereo
        #         outputAudioData[outputPointer:outputPointer+AUDIO_FADE_ENVELOPE_SIZE] *= mask
        #         outputAudioData[endPointer-AUDIO_FADE_ENVELOPE_SIZE:endPointer] *= 1-mask

        #     startOutputFrame = int(math.ceil(outputPointer/samplesPerFrame))
        #     endOutputFrame = int(math.ceil(endPointer/samplesPerFrame))
        #     for outputFrame in range(startOutputFrame, endOutputFrame):
        #         inputFrame = int(chunk[0]+NEW_SPEED[int(chunk[2])]*(outputFrame-startOutputFrame))
        #         didItWork = copyFrame(inputFrame,outputFrame)
        #         if didItWork:
        #             lastExistingFrame = inputFrame
        #         else:
        #             copyFrame(lastExistingFrame,outputFrame)

        #     outputPointer = endPointer

        # wavfile.write(TEMP_FOLDER+"/audioNew.wav",SAMPLE_RATE,outputAudioData)

        # command = "ffmpeg -framerate "+str(frameRate)+" -i "+TEMP_FOLDER+"/newFrame%06d.jpg -i "+TEMP_FOLDER+"/audioNew.wav -strict -2 "+OUTPUT_FILE
        # subprocess.call(command, shell=True)
        pass


    print(choice, end=' \n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str,  help='Input file to be modified.')
    parser.add_argument('--silent_threshold', type=float, default=0.12, help="An index between 0 and 1. A value of 0 will keep everything; a value of 1 will only retain the silence. Default = 0.12")
    parser.add_argument('--frame_margin', type=float, default=1, help="A frame margin for smoothing out the transitions. Default = 1")
    parser.add_argument('--frame_quality', type=int, default=3, help="Quality of the frame output by FFMPEG. Default = 3")

    args = parser.parse_args()

    assert args.input_file != None , "Error, null input file. Aborting."

    process_video(filename=args.input_file,
                  silent_threshold=args.silent_threshold,
                  frame_margin=args.frame_margin,
                  frame_quality=args.frame_quality)
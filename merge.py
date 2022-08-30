import os
import glob
import json
import datetime
import subprocess
from pprint import pprint as print
from fractions import Fraction
from numpy import around as round
from tqdm import tqdm
import ffpb

def convertSeconds(seconds):
    return [round(float(t),0) for t in str(datetime.timedelta(seconds=seconds)).split(':')]

def strtime(seconds):
    h,m,s = convertSeconds(seconds)
    return f'{int(h):02d}:{int(m):02d}:{int(s):02d}'

def getWAVinfo(filename:str, stderr:bool=False):
    pipe = subprocess.run(f'ffprobe -v quiet -print_format json -show_format -show_streams {filename}',
                          stdout = subprocess.PIPE,
                          stderr = subprocess.PIPE,
                          bufsize=10**8)
    
    if stderr:
        return (pipe.stdout.decode('utf-8'), pipe.stderr.decode('utf-8'))
    else:
        return json.loads(pipe.stdout.decode('utf-8'))

def getSeconds(info):
    return float(info['streams'][0]['duration'])

files = list(filter(os.path.isfile, glob.glob(os.path.join(os.getcwd(), '*.mp4'))))
files.sort(key=lambda x: os.path.getctime(x))

with open('files.txt', 'w') as output:
    for file in files:
        output.write("file '" + file + "'\n")

with open('times.txt', 'w') as fp:
    total = 0
    current = 0
    for i,file in enumerate(files):
        info = getWAVinfo(file)
        total += getSeconds(info)
        fp.write(f'{strtime(current)} - {i+1}\n')
        current = total

ffpb.main(argv='-f concat -safe 0 -i files.txt merged.mp4', tqdm=tqdm)
os.remove('files.txt')


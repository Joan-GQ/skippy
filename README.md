<div align='center'>
  <img src="./assets/skippy-logo.png">
</div>

Skip.py is a video editing Python script. Based on a silent threshold, it can automatically speed up silent sections of a video, or cut them completely. It uses `ffmpeg` under the hood to process video and audio information.

![](assets/skippy-test.png)

It's based off of [carykh's jumpcutter](https://www.github.com/carykh/jumpcutter), but it has a full built in command line interface to let the user decide whether it wants to process the video or not, showing an estimate of the minimum possible time after the time warping. It also autodetects the input's framerate and samplerate using `ffprobe` to avoid errors. It also uses [`ffpb`](https://www.github.com/althonos/ffpb) for tracking the progress of `ffmpeg`.



# Requirements

- A working Python installation.
  You may need to install aditional libraries. (See `skip.py` for the imports).

- A working FFMPEG installation added to `PATH`.



# Usage

`python .\skip.py --input_file 'path/to/video.mp4' --silent_threshold 0.11 --frame_margin 1 --frame_quality 3`

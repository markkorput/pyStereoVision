# pyStereoVision

# Install Dependencies
```
pip install -r requirements.txt
```

# Usage

## Stereo Recording (2 cameras -> 2 video files)

By default record.py uses camera IDs 0 (left) and 1 (right) and attempts to record at 720p 24fps to ```saved-media/video_L.avi``` and```saved-media/video_R.avi```
```
python -m SV.record
```

The following invocation records from cameras 5 (left) and 8 (right) and attempts to record at 480p 30fps to ```saved-media/test2_L.avi``` and ```saved-media/test2_R.avi```
```
python -m SV.record -l 5 -r 8 -d 480p -f 30 -o saved-media/test2.avi
```

## Calibrate camera (2 videos -> preview)
The x and y specify the number of columns and rows spectively of the chessboard (that should appear in the video)
```
python -m SV.calibrate -v saved-media/test2.avi -d 0.25 -x 9 -y 6
```
# opssat-sar-ai tools

All tools are organized in the subfolders in this folder.
A single Docker image in this directory will serve to be able to run all the tools

Build image:
```
./build_image.sh
```

Start:
```
docker-compose up
```

## Running the spectrogram labelling tool
Make sure the input ```.cf32``` files are in your local directory that is being mapped to the container's /input root directory by the ```docker-compose.yml``` file. The ```label.py``` script will look for all ```.cf32``` files in this dir.
```
$ docker exec -it sar-ai-tools /bin/bash
cd /tools/labelling
python3 label.py /input labelled_dataset.csv
```

Use keys:
- F: advance to next file
- A: revert to previous file
- R: clear all selected bounding boxes
- B: toggle ROI lenght (there are short beacons and long beacons)

Use mouseclick:
- LEFT: drop a beacon bounding gox

TODO: when advancing to next spectrum, record the positions of all bounding boxes, write these coords together with filename to the specified output file in .csv format,  then it will clear all bounding boxes before moving to the next spectrum!

## Plotting spectogram with Matplotlib
```
$ docker exec -it sar-ai-tools /bin/bash
cd /tools/plotting
python3 plot.py -f float32 -p spec -fw 512 /input/<CF32_FILE> 
```

## Plotting spectrogram with renderfall
```
$ docker exec -it sar-ai-tools /bin/bash
renderfall -n 512 -v -f float32 -l 256 -w hann  /input/<CF32_FILE> 
```





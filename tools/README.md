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
Run above commands first. A GUI will open. You can put this GUI to the side (do not close it) as we want to run another program, so we open a shell into the container.
Make sure the input ```.cf32``` files are in your local directory that is being mapped to the container's /input directory by the ```docker-compose.yml``` file. The ```label.py``` script will look for all ```.cf32``` files in this dir.
```
$ docker exec -it sar-ai-tools /bin/bash
cd /tools/labelling
python3 label.py my_assigned_labelset.csv
```

Later on the user can add his/her csv chunk to the ```master_labelset.csv```:
```
cat my_assigned_labelset.csv >> master_labelset.csv
```

Use keys:
- F: advance to next file
- A: revert to previous file
- R: clear all selected bounding boxes
- B: toggle ROI lenght (there are short beacons and long beacons, and an ROI for cutoff beacons)

Use mouseclick:
- LEFT: drop a beacon bounding box

IMPORTANT NOTE: Note that the actual csv file contents are written when the application is closed via the 'X' button in the window!

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





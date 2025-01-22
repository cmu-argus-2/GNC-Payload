# Vision Training

## Setup
Ensure that your conda environment is setup up and sourced and run:

```conda env create -f environment.yml``` 

This will create an environment called `sat_env_vision` that has all necessary packages to run the scripts.


## To Check
You will have to create your own google earth engine project on `https://earthengine.google.com/noncommercial/`. For this you need to use your personal email address as CMU domains don't seem to work here. In `earthenginedl.py` make sure to replace the project argument in the `ee.Initialize()` function with your own project name. Subsequently you may need to authenticate yourself. You can do this either through the commandline by running `earthengine authenticate` or in `earthenginedl.py` by setting `ee.Authenticate(force=True)`. This will create a new set of credentials for your project.


## Run
### Download Dataset
To Download the Data into VisionTrainingGround/Landsat_Data, run the `run_downloader.sh` script (make sure it is executable by running ```chmod +x ./run_downloader.sh```)
```./run_downloader.sh```

### Run YOLO
To train the YOLO model, use the `run_YOLO.sh` script. Ensure it is executable by running:
```chmod +x ./run_YOLO.sh```
Then execute the script:
```./run_YOLO.sh```
This script will set up the environment and begin the YOLO model training or evaluation.

### Run RCNet
To train RCNet model, execute the `run_RCnet.sh` script. Make sure it is executable:
```chmod +x ./run_RCnet.sh```
Run the script using:
```./run_RCnet.sh```

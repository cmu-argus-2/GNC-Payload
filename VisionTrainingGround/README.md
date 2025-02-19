# Vision Training

## Setup
Ensure that your conda environment is setup up and sourced and run:

```conda env create -f environment.yml``` 

This will create an environment called `sat_env_vision` that has all necessary packages to run the scripts.


## To Check
You will have to create your own Google Earth engine project on `https://earthengine.google.com/noncommercial/`. For this you need to use your personal email address as CMU domains don't seem to work here.

Next, copy `user_config.example.yaml` to `user_config.yaml` and fill in your project name.
```
cd ../
cp user_config.example.yaml user_config.yaml
```
Subsequently, you may need to authenticate yourself. You can do this either through the commandline by running `earthengine authenticate` or in `earthenginedl.py` by setting `ee.Authenticate(force=True)`. This will create a new set of credentials for your project.

### Train YOLO
To train the YOLO model, use the `train_YOLO.sh` script. Ensure it is executable by running:
```chmod +x ./train_YOLO.sh```
Then execute the script:
```./train_YOLO.sh```
This script will set up the environment and begin the YOLO model training or evaluation.

### Train RCNet
To train RCNet model, execute the `train_RCnet.sh` script. Make sure it is executable:
```chmod +x ./train_RCnet.sh```
Run the script using:
```./train_RCnet.sh```

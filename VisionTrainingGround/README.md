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

## Running the Pipeline

### Directory Structure
The following directory structure will be created as the pipeline is run:
- /training_directory
  - rc_model_weights.pth
  - /{region_id}
    - 00000.png
    - 00000_mgrs_regions.npy
    - 00000_lat_lon.npy
    - ...
    - saliency_map.tif
    - bounding_boxes.csv
    - yolo_model_weights.pt
    - LD_training
      - dataset.yaml
      - /train
        - /images
          - 00000.png (symlink)
          - ...
        - /labels
          - 00000.txt
          - ...
      - /test
        - ...
      - /val
        - ...
  - ...

### Generate Training Data
To generate training data, run `DataPipeline/generate_training_data.py`. This script will generate the training data for
both the RC and LD pipelines.

### Run the LD Network Training Pipeline 
To run the LD network training pipeline, use the `LD/run_LD_training_pipeline.sh` script.

### Train RCNet
To train RCNet model, execute the `train_RCnet.sh` script. Make sure it is executable:
```chmod +x ./train_RCnet.sh```
Run the script using:
```./train_RCnet.sh```

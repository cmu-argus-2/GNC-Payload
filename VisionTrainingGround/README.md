# Vision Training

## Setup
Ensure that your conda environment is setup up and sourced and run:

```conda env create -f environment.yml``` 

This will create an environment called `sat_env_vision` that has all necessary packages to run the scripts.


## To Check
Depending on where you have place the repository in your file directory, it may be necessary to adjust the `LANDMARK_BASE` and `FINAL_OUTPUT_PATH` variables as well as the python binary paths at the end of the script.
(You may also be using conda rather than miniconda). 

I have created a Google Earth engine project called: `ee-vision-pipeline-training` from where the pictures will be downloaded. You may need to request access to enable download.
Alternatively, it is possible to create your own project on `https://earthengine.google.com/noncommercial/`. For this you need to use your personal email address as CMU domains don't seem to work here. In `earthenginedl.py` make sure to replace the project argument in the `ee.Initialize()` function with your own project name. Subsequently you may need to authenticate yourself. You can do this either through the commandline by running `earthengine authenticate` or in `earthenginedl.py` by setting `ee.Authenticate(force=True)`. This will create a new set of credentials for your project.


## Run
To run simply run the `run_pipeline.sh` script (make sure it is executable by running ```chmod +x ./run_pipeline.sh```)

```./run_pipeline.sh```

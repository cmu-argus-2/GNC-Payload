# GNC-Payload
This repo strives to house everything associated with the payload of the Argus satellite. This includes:
- Scripts for downloading Landsat and Sentinel data (image_downloader folder)
- Scripts for training and running inference of the region classifier and landmark detector (vision_inference folder)
- Orbit determination algorithms (orbit_determination folder)
- Code for mocking IMU measurements (sensors folder)
- Code for simulating images given a position and orientation of the satellite in ECEF (image_simulation folder)

Code from the following repos was used as a starting point:
- [Earth Engine Downloader](https://github.com/cmuabstract/eedl)
- [Vision Pipeline Inference](https://github.com/cmu-argus-1/FSW-Jetson/tree/main/flight/vision)
- [Image Simulation](https://github.com/kyledmccleary/earth-vis-argus)
- [Vision Training Pipelines](https://github.com/cmu-argus-1/VisionTrainingGround)

## Setting Up
### Preferred Method
An `environment_training.yml` has been provided. The easiest way to set up a v-env is to run:

```
conda env create -f environment_training.yml
```

### Back-up
Depending on the your OS this may not work (tested on Ubuntu 22 (works) and Windows (doesn't seem work)). In this case set up manually via:

```
conda create --name MY_ENV python=3.12
```
Activate your environment MY_ENV and run the requirements file:

```
conda activate MY_ENV
pip install -r ./requirements.txt
```

This should set up the environment and packages needed but this can be flaky if pip uses cached packages it finds on your machine. 

### Special Installations
The version of Brahe on PyPI is not up to date, and we have a custom version of ultralytics, so you have to install these packages directly from the latest code on GitHub using the following commands. TODO: can we integrate this into environment_training.yml?
```
pip install git+https://github.com/duncaneddy/brahe.git@master
pip install git+https://github.com/cmu-argus-1/custom_ultralytics@main
```

## Pre-Commit Hook 
A pre-commit hook can be set up to run pylint and black formatter.

The hook can be set up by running the following command in the root directory of the repository:

```
pre-commit install
```

Then every time you run `git commit ...` the pre-commit hook will run the linter and formatter and check that everything is in order. 
Only commit to your branch if ALL tests pass. Otherwise address and correct the shown errors.
These tools can also be run separately or on specific folders or files by specifying for example:

```
pylint ./sensors/bias.py
black ./orbit_determination/nonlinear_least_squares_od.py
pylint ./vision_inference
```

# JETSON SETUP #
This guide describes how to set up an conda environment for the Jetson.

This guide assumes that the Jetson has already been flashed using the sdkmanager via Jetpack 6.0, installing CUDA runtime and sdk components.
Note that this environment will only allow you to run vision inference and the filter. To 

## 1. Ensure required packages are downloaded
```
sudo apt-get -y update; 
sudo apt-get install -y  python3-pip libopenblas-dev;
```

## 2. Create Conda Environment with Python 3.10
Create a conda environment using Python 3.10 and activate it
```
conda create -y -n <my_env> python=3.10
```
```
conda activate <my_env>
```

## 3. Download torch whl 
Go to https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/ and download the 24.05 torch wheel. 
torch-2.4.0a0+07cecf4168.nv24.05.14710581-cp310-cp310-linux_aarch64.whl
Set `TORCH_INSTALL` to the path where the downloaded wheel is located:
```
export TORCH_INSTALL=path/to/torch-2.4.0a0+07cecf4168.nv24.05.14710581-cp310-cp310-linux_aarch64.whl
```

## 4. Setup numpy and torch whl
```
python3 -m pip install --upgrade pip; python3 -m pip install numpy==1.26.4; python3 -m pip install --no-cache $TORCH_INSTALL
```

## 5. (Setup and) activate swap memory
In order to build torchvision in the next step, we require swap memory. In case it already exists just run the final line
```
sudo fallocate -l 6G /swapfile;
sudo chmod 600 /swapfile;
sudo mkswap /swapfile;
sudo swapon /swapfile; # Run this line only if you have already previously set up swap memory.
```

## 6. Clone and setup torchvision
Torchvision needs to be built from source for the Jetson.
```
git clone --branch v0.17.0 https://github.com/pytorch/vision.git
cd vision
```
Ensure you're in the <my_env> environment, ensure the following packages are installed and build torchvision.
```
sudo apt-get install libjpeg-dev zlib1g-dev
python setup.py install
```

## 7. Install quaternion package from conda-forge
```
conda install -c conda-forge quaternion numpy=1.26.4
```

## 8. Install jetson requirements from txt
```
pip install -r jetson_setup/jetson_requirements.txt
```

## 9. Install custom ultralytics from source
```
cd ..
git clone https://github.com/cmu-argus-1/custom_ultralytics.git
cd custom_ultralytics
```
In the root directory, adjust the `pyproject.toml` file. In line 22, change
```
requires = ["setuptools>=43.0.0", "wheel"]
```
to
```
requires = ["setuptools>=61.0", "wheel"]
```
Then from the ultralytics root directory run:
```
pip install -e .
```
TODO: Once I get rights to the ultralytics branch, I will push an update that fixes this automatically.

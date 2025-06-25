# C++ sim setup
cd simulation
sh install.sh
cd ../

# Copy the montecarlo folder into FSW to store results
# Make a Results directory and copy the params.yaml file
# On the first run, the C++ default yaml file runs. For later runs, this file can be changed
mkdir -p montecarlo/configs
if [ ! -f montecarlo/configs/params.yaml ]; then
    cp simulation/montecarlo/configs/params.yaml montecarlo/configs/
fi
#!/bin/bash

# Venv settings
env_name="venv"
python_version="3.9"

# Create a venv using anaconda
command="conda create -n $env_name python=$python_version anaconda"
$command


source activate $env_name
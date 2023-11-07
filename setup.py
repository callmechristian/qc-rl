import os
import subprocess
import venv

# Venv settings
env_name = "venv"
python_version = "3.8"

# Create a venv using anaconda
command = f"conda create -n {env_name} python={python_version} anaconda"
process = subprocess.run(command, shell=True, check=False)

# Activate the virtual environment
activate_script = "conda activate venv"
command = f"{activate_script} && ./setup.sh"

if os.name == 'posix':
    process = subprocess.run(command, shell=True, check=False)
else:
    raise OSError("Operating system not supported!")

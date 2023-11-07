import os
import subprocess
import venv

# Create a virtual environment
venv_dir = "./venv"
venv.create(venv_dir, with_pip=True)

# Activate the virtual environment
activate_script = ". ./venv/bin/activate"
command = f"{activate_script} && ./setup.sh"

if os.name == 'posix':
    process = subprocess.run(command, shell=True, check=False)
else:
    raise OSError("Operating system not supported!")

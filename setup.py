import os
import subprocess
if os.name == 'posix':
    process = subprocess.run('./setup.sh', shell=True, check=True)
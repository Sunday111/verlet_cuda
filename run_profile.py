import subprocess
from pathlib import Path
import os

script_dir = Path(__file__).parent.resolve()
bin_dir = script_dir / "build" / "bin"
program_path = bin_dir / "cuda_gl"
env = os.environ.copy()
env["CUDA_PROFILE"] = "1"
subprocess.run(args=[program_path], env=env, cwd=bin_dir)

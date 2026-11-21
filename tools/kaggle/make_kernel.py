"""Build a Kaggle *script* job that executes one notebook with papermill and saves the executed
notebook (all cell outputs, also when a cell fails) into /kaggle/working, so
`kaggle kernels output` downloads it together with the Excel files.

usage: make_kernel.py <notebook.ipynb> <slug> <out_dir> [kernel_source ...]

Environment overrides: KAGGLE_USER (default daneshselwal), KAGGLE_DATASET
(default daneshselwal/reguq-uq-data).
"""
import base64
import json
import os
import sys

nb_path, slug, out_dir = sys.argv[1:4]
user = os.environ.get("KAGGLE_USER", "daneshselwal")
dataset = os.environ.get("KAGGLE_DATASET", "daneshselwal/reguq-uq-data")
kernel_sources = sys.argv[4:]
os.makedirs(out_dir, exist_ok=True)
payload = base64.b64encode(open(nb_path, "rb").read()).decode()
name = os.path.basename(nb_path)

run_py = f'''# Runs "{name}" with papermill; the executed notebook is saved as an output file.
import base64, os, subprocess, sys, time
NAME = {name!r}
os.chdir("/kaggle/working")
with open("/kaggle/tmp_input.ipynb", "wb") as f:
    f.write(base64.b64decode({payload!r}))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "papermill"], check=True)
out = "/kaggle/working/EXECUTED_" + NAME
t0 = time.time()
r = subprocess.run([sys.executable, "-m", "papermill", "/kaggle/tmp_input.ipynb", out,
                    "--cwd", "/kaggle/working", "-k", "python3", "--log-output",
                    "--request-save-on-cell-execute", "--autosave-cell-every", "60"])
print(f"papermill exit code {{r.returncode}} after {{(time.time() - t0) / 60:.1f}} min")
print("RUN_STATUS:", "OK" if r.returncode == 0 else "FAILED (see EXECUTED notebook for the failing cell)")
'''
open(os.path.join(out_dir, "run.py"), "w").write(run_py)
meta = {
    "id": f"{user}/{slug}", "title": slug, "code_file": "run.py", "language": "python",
    "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False,
    "enable_internet": True, "dataset_sources": [dataset],
    "kernel_sources": kernel_sources, "competition_sources": [],
}
json.dump(meta, open(os.path.join(out_dir, "kernel-metadata.json"), "w"), indent=1)
print("built", out_dir, "for", name)

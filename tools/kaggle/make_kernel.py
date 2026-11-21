"""Build a Kaggle *script* job that executes one notebook with papermill and saves the executed
notebook (all cell outputs, also when a cell fails) into /kaggle/working, so
`kaggle kernels output` downloads it together with the Excel files.

usage: make_kernel.py <notebook.ipynb> <slug> <out_dir> [kernel_source ...]
                      [--user USER] [--datasets owner/a,owner/b] [--no-internet]

  --user         Kaggle user that owns the job (default: $KAGGLE_USER, else daneshselwal).
                 queue_push.sh renames the id at push time using $KAGGLE_USER.
  --datasets     comma-separated dataset sources (default: $KAGGLE_DATASET, else
                 daneshselwal/reguq-uq-data). For an account without internet attach its own
                 <user>/uq-offline-kit (wheels, pile data, best_params.pkl).
  --no-internet  enable_internet = false. The job then installs papermill (and the notebooks
                 install their packages) from the attached kit's wheels/ folder.

Without the options the job is built exactly as before (main account, internet on).
"""
import argparse
import base64
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("nb_path")
ap.add_argument("slug")
ap.add_argument("out_dir")
ap.add_argument("kernel_sources", nargs="*")
ap.add_argument("--user", default=os.environ.get("KAGGLE_USER", "daneshselwal"))
ap.add_argument("--datasets", default=os.environ.get("KAGGLE_DATASET", "daneshselwal/reguq-uq-data"))
ap.add_argument("--no-internet", action="store_true")
a = ap.parse_args()

os.makedirs(a.out_dir, exist_ok=True)
payload = base64.b64encode(open(a.nb_path, "rb").read()).decode()
name = os.path.basename(a.nb_path)

run_py = f'''# Runs "{name}" with papermill; the executed notebook is saved as an output file.
import base64, glob, os, subprocess, sys, time
NAME = {name!r}
os.chdir("/kaggle/working")
with open("/kaggle/tmp_input.ipynb", "wb") as f:
    f.write(base64.b64decode({payload!r}))
# offline kit attached (account without internet) -> install from its wheels/ folder, no index
_wh = [p for p in glob.glob("/kaggle/input/**/wheels", recursive=True) if glob.glob(p + "/*.whl")]
_src = ["--no-index", "--find-links", _wh[0]] if _wh else []
print("pip source:", _wh[0] if _wh else "PyPI")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *_src, "papermill"], check=True)
out = "/kaggle/working/EXECUTED_" + NAME
t0 = time.time()
r = subprocess.run([sys.executable, "-m", "papermill", "/kaggle/tmp_input.ipynb", out,
                    "--cwd", "/kaggle/working", "-k", "python3", "--log-output",
                    "--request-save-on-cell-execute", "--autosave-cell-every", "60"])
print(f"papermill exit code {{r.returncode}} after {{(time.time() - t0) / 60:.1f}} min")
print("RUN_STATUS:", "OK" if r.returncode == 0 else "FAILED (see EXECUTED notebook for the failing cell)")
'''
open(os.path.join(a.out_dir, "run.py"), "w").write(run_py)
meta = {
    "id": f"{a.user}/{a.slug}", "title": a.slug, "code_file": "run.py", "language": "python",
    "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False,
    "enable_internet": not a.no_internet,
    "dataset_sources": [d.strip() for d in a.datasets.split(",") if d.strip()],
    "kernel_sources": a.kernel_sources, "competition_sources": [],
}
json.dump(meta, open(os.path.join(a.out_dir, "kernel-metadata.json"), "w"), indent=1)
print("built", a.out_dir, "for", name, "| user", a.user, "| internet", not a.no_internet,
      "| datasets", meta["dataset_sources"])

# Kaggle job tools

Small helpers for running the pipeline notebooks as Kaggle jobs and getting the **executed
notebook** (every cell output) back through the Kaggle CLI.

Why: for a normal notebook job, `kaggle kernels output` returns only the files in
`/kaggle/working` and a log, not the notebook with its cell outputs. These tools wrap each
notebook in a small *script* job that runs it with **papermill** and writes
`EXECUTED_<notebook>.ipynb` into `/kaggle/working`, so it is downloaded with the Excel results.
Papermill saves the notebook after every cell, so a failing run still returns everything up to the
failing cell.

## `make_kernel.py`

```bash
python tools/kaggle/make_kernel.py <notebook.ipynb> <slug> <job_dir> [kernel_source ...]
# e.g.
python tools/kaggle/make_kernel.py \
  "examples/pile_uncertainty_analysis/quantile_regression/Quantile_Regression.ipynb" \
  pile-demo-nb2-quantile /tmp/kd_nb2 daneshselwal/pile-demo-params
```

Writes `<job_dir>/run.py` (the notebook embedded, plus the papermill call) and
`<job_dir>/kernel-metadata.json` (CPU, internet on, private, dataset `daneshselwal/reguq-uq-data`,
the given `kernel_sources`, e.g. the tuning job whose output contains `best_params.pkl`).
Environment overrides: `KAGGLE_USER`, `KAGGLE_DATASET`.

At the end of the job log, `RUN_STATUS: OK` or `RUN_STATUS: FAILED` tells whether every cell ran.

## `queue_push.sh`

```bash
tools/kaggle/queue_push.sh /tmp/kd_nb3 /tmp/kd_nb4 ...
```

**Run names.** Whatever slug `make_kernel.py` was given, `queue_push.sh` renames each job right before
its push to `REG<DD>_<MM>_<YYYY>_<HH>_<MM>` (local push time; URL slug `reg<DD>-<MM>-<YYYY>-<HH>-<MM>`)
and appends `code name, slug, notebook, job dir, time` to `docs/run_registry.tsv`. Use
`kaggle kernels output <user>/<slug> -p <dir>` with the slug from the registry to fetch the outputs.


Pushes each job folder in turn. If the account already has the maximum of 5 concurrent CPU
sessions, it waits 2 minutes and retries. The API token is read from `~/.kaggle/access_token` and is
never printed. `KAGGLE_BIN` overrides the path of the `kaggle` executable.

## Getting results

```bash
kaggle kernels status daneshselwal/<slug>
kaggle kernels output daneshselwal/<slug> -p <local_dir>
```

Record every job in [`docs/KAGGLE_RUNS.md`](../../docs/KAGGLE_RUNS.md) and commit outputs of **final** runs only to
`examples/<dataset>_uncertainty_analysis/<phase>/results/` (trial/demo runs stay on Kaggle).

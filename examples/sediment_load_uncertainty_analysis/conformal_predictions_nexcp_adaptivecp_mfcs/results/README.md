# Not run — out of compute

This notebook (NexCP, Adaptive CP, MFCS) was **not run** for sediment_load.

- sediment_load has 1095 training and 365 test rows, the largest dataset here.
- NexCP-Full refits the model for many candidate values at every test point. Even with FIX B15
  (edge scan + measured workers), the whole notebook is estimated at about 200 hours of CPU time;
  its slowest model alone would need about 48 hours. A Kaggle run stops after 12 hours.
- The online methods in the same notebook process the test points in order, so the run cannot be
  split into smaller jobs by test points.

All the other notebooks were run for sediment_load. The same notebook was run for every other dataset.

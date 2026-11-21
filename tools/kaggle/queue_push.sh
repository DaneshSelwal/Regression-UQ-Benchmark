#!/bin/zsh
# usage: queue_push.sh dir1 dir2 ...   pushes each Kaggle job folder, retrying while the 5-session limit is reached
# The token is read from ~/.kaggle/access_token (main account) and never printed.
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
KAGGLE_BIN="${KAGGLE_BIN:-$(command -v kaggle || echo /opt/anaconda3/bin/kaggle)}"
for d in "$@"; do
  while true; do
    out=$("$KAGGLE_BIN" kernels push -p "$d" 2>&1 | grep -v outdated)
    if echo "$out" | grep -q "successfully pushed"; then echo "$(date +%H:%M) $out"; break; fi
    if ! echo "$out" | grep -q "Maximum batch CPU session"; then echo "$(date +%H:%M) ERROR $d: $out"; break; fi
    sleep 120
  done
done
echo QUEUE_DONE

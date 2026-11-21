#!/bin/zsh
# usage: queue_push.sh dir1 dir2 ...
# Pushes each Kaggle job folder, retrying while the 5-session CPU limit is reached.
# Naming rule: every run is named REG<DD>_<MM>_<YYYY>_<HH>_<MM> = the local time it is pushed
# (Kaggle URL slug: reg<DD>-<MM>-<YYYY>-<HH>-<MM>). Each push is appended to the run registry.
# The token is read from ~/.kaggle/access_token (main account) and never printed.
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
KAGGLE_BIN="${KAGGLE_BIN:-$(command -v kaggle || echo /opt/anaconda3/bin/kaggle)}"
KUSER="${KAGGLE_USER:-daneshselwal}"
REG="${RUN_REGISTRY:-$(dirname "$0")/../../docs/run_registry.tsv}"
[[ -f "$REG" ]] || printf "code_name\tslug\tnotebook\tjob_dir\tpushed_at\n" > "$REG"
for d in "$@"; do
  nb=$(grep -m1 '^NAME = ' "$d/run.py" | sed "s/^NAME = '//; s/'$//")
  while true; do
    code="REG$(date +%d_%m_%Y_%H_%M)"
    # one name per minute: if this minute's name is taken, wait for the next minute
    while grep -q "^$code	" "$REG"; do sleep 15; code="REG$(date +%d_%m_%Y_%H_%M)"; done
    slug=$(echo "$code" | tr 'A-Z_' 'a-z-')
    python3 -I - "$d/kernel-metadata.json" "$KUSER/$slug" "$code" <<'PY'
import json, sys
p, kid, title = sys.argv[1:4]
m = json.load(open(p)); m["id"] = kid; m["title"] = title
json.dump(m, open(p, "w"), indent=1)
PY
    out=$("$KAGGLE_BIN" kernels push -p "$d" 2>&1 | grep -v outdated)
    if echo "$out" | grep -q "successfully pushed"; then
      printf "%s\t%s\t%s\t%s\t%s\n" "$code" "$KUSER/$slug" "$nb" "$(basename "$d")" "$(date '+%Y-%m-%d %H:%M')" >> "$REG"
      echo "$(date +%H:%M) $code ($nb) pushed: https://www.kaggle.com/code/$KUSER/$slug"; break
    fi
    if ! echo "$out" | grep -q "Maximum batch CPU session"; then echo "$(date +%H:%M) ERROR $d: $out"; break; fi
    sleep 120
  done
done
echo QUEUE_DONE

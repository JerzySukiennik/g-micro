#!/bin/bash
# Copy the view modules the desktop app and the website share.
#
# They are copied rather than imported across directories because Firebase
# Hosting serves one folder and cannot reach up into app/renderer. Copies drift;
# this script is the answer to that — run it after touching any renderer module,
# and the diff will show if the site was left behind.
set -euo pipefail
cd "$(dirname "$0")"
SRC=../app/renderer
for f in style.css chat.js format.js composer.js history.js; do
  cp "$SRC/$f" "./$f"
  echo "  $f"
done
echo "zsynchronizowane z $SRC"

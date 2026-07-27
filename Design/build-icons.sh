#!/bin/bash
# Assemble G-Micro.icns from the rendered PNGs.
#
# The PNGs come from rasterize.html, not from qlmanage. QuickLook composites
# SVG onto an opaque white background, so its "transparent" output was a white
# box — the white-on-white logo that showed up in the README. A browser canvas
# starts transparent, which is what an app icon needs.
#
# To re-render the PNGs:
#     python Design/raster-server.py 150 &
#     open http://127.0.0.1:8765/
#
# The page must be loaded *from* the server: Chrome blocks a file:// page from
# POSTing to localhost (private network access), and the upload silently fails.
set -euo pipefail
cd "$(dirname "$0")"

OUT=assets
ICONSET="$OUT/G-Micro.iconset"

for s in 16 32 64 128 256 512 1024; do
  [ -f "$OUT/icon-$s.png" ] || { echo "missing $OUT/icon-$s.png — render first"; exit 1; }
done

rm -rf "$ICONSET" && mkdir -p "$ICONSET"
cp "$OUT/icon-16.png"   "$ICONSET/icon_16x16.png"
cp "$OUT/icon-32.png"   "$ICONSET/icon_16x16@2x.png"
cp "$OUT/icon-32.png"   "$ICONSET/icon_32x32.png"
cp "$OUT/icon-64.png"   "$ICONSET/icon_32x32@2x.png"
cp "$OUT/icon-128.png"  "$ICONSET/icon_128x128.png"
cp "$OUT/icon-256.png"  "$ICONSET/icon_128x128@2x.png"
cp "$OUT/icon-256.png"  "$ICONSET/icon_256x256.png"
cp "$OUT/icon-512.png"  "$ICONSET/icon_256x256@2x.png"
cp "$OUT/icon-512.png"  "$ICONSET/icon_512x512.png"
cp "$OUT/icon-1024.png" "$ICONSET/icon_512x512@2x.png"

iconutil -c icns "$ICONSET" -o "$OUT/G-Micro.icns"

echo "built $OUT/G-Micro.icns"
ls -1sh "$OUT" | sed 's/^/  /'

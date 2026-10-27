#!/usr/bin/env bash
# Re-capture the real viewer footage (seed 45 SOLO vs SHARED MAP) and re-render the video.
# Run from anywhere: bash video/scripts/recapture-and-render.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VID="$ROOT/video"
TMP="$(mktemp -d)"
python3 -m http.server -d "$ROOT/viewer" 8765 >/dev/null 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true; rm -rf "$TMP"' EXIT
sleep 1
cd "$VID"
node scripts/capture-viewer.mjs \
  "http://127.0.0.1:8765/?left=seed45-solo-sr1-R20-cal-b0.05&right=seed45-share-sr1-R20-cal-b0.05&hiker=4&reveal=end&capture=1&play=1&speed=10" \
  "$TMP/frames" 600
# playback must end before frame ~550 (the Hero scene dims the clip at scene frame 612 = clip frame 550)
LAST=$(md5sum "$TMP"/frames/f*.jpg | awk '{print $1}' | uniq -c | awk '{n+=$1} END {print n-$1}')
echo "viewer playback stops at clip frame $LAST"
npx remotion ffmpeg -hide_banner -loglevel error -y -framerate 30 -i "$TMP/frames/f%05d.jpg" -frames:v 600 \
  -c:v libx264 -crf 16 -pix_fmt yuv420p -r 30 public/hero.mp4
# refresh the fogged hiker views used in the Hook/Idea scenes
for h in 0 1 2 3; do cp "$ROOT/viewer/data/frames/seed45-share-sr1-R20-cal-b0.05/0_$h.jpg" "public/view0_$h.jpg"; done
cp "$ROOT/viewer/data/frames/seed45-share-sr1-R20-cal-b0.05/80_0.jpg" public/forest_bg.jpg
npx remotion render src/Root.tsx MainVideo out/lost-together-demo.mp4 --log=error --crf=18
npx remotion still src/Root.tsx Hero out/thumbnail.png --frame=720 --log=error

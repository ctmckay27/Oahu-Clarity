#!/usr/bin/env bash
set -euo pipefail
: "${ACTUATOR_TOKEN:?ACTUATOR_TOKEN required}"
: "${VNC_PASSWORD:?VNC_PASSWORD required}"
export DISPLAY="${DISPLAY:-:99}"
export PROFILE_DIR="${PROFILE_DIR:-/tmp/mari404-chrome-profile}"
mkdir -p "$PROFILE_DIR" /tmp/.X11-unix
Xvfb "$DISPLAY" -screen 0 1440x900x24 -ac +extension RANDR >/tmp/xvfb.log 2>&1 &
for i in $(seq 1 30); do xdpyinfo -display "$DISPLAY" >/dev/null 2>&1 && break; sleep 0.2; done
x11vnc -storepasswd "$VNC_PASSWORD" /tmp/vnc.pass >/dev/null
x11vnc -display "$DISPLAY" -rfbport 5900 -rfbauth /tmp/vnc.pass -forever -shared -localhost -noxdamage >/tmp/x11vnc.log 2>&1 &
websockify 6080 localhost:5900 >/tmp/websockify.log 2>&1 &
exec node /app/server.js

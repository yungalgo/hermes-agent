#!/bin/sh
# shellcheck shell=sh
# Fly Machines entrypoint.
#
# Fly's own init runs as PID 1 (it hosts the SSH / api-proxy "hallpass"),
# so s6-overlay — which refuses to run as anything other than PID 1 — cannot
# be the container init on Fly. Instead of the s6 supervision tree we boot the
# gateway directly:
#
#   1. run the same root bootstrap cont-init normally does (volume perms/dirs),
#   2. drop to the hermes user and exec `hermes gateway run`.
#
# Selected by boot.sh only when FLY_MACHINE_ID is set; local `docker run`
# keeps the full s6 tree unchanged.
set -u

HERMES_HOME="${HERMES_HOME:-/opt/data}"
export HERMES_HOME

# stage2-hook.sh is plain /bin/sh and idempotent. On Fly (running as root, no
# HERMES_UID/PUID, no docker socket) its meaningful work is creating and
# chowning $HERMES_HOME so the dropped hermes user can write the volume, plus
# config/skills seeding. A non-zero exit here must not strand the machine in a
# reboot loop, so surface it and continue to the gateway rather than aborting.
if ! /opt/hermes/docker/stage2-hook.sh; then
    echo "[fly-entrypoint] WARNING: stage2-hook.sh exited non-zero; continuing to gateway" >&2
fi

# s6's main-wrapper resets HOME to the volume before dropping privileges so
# HOME-anchored state lands under $HERMES_HOME; mirror that. /command holds the
# s6 userland (s6-setuidgid); it is only on PATH when s6 runs, so add it here.
export HOME="$HERMES_HOME"
export PATH="/command:${PATH}"
cd "$HERMES_HOME"

# Bare `hermes` is the interactive TUI (exits immediately without a TTY); the
# gateway is `hermes gateway run` — the same command the docker provisioner
# passes as CMD. Drop to the hermes user exactly as the s6 services do.
exec s6-setuidgid hermes /opt/hermes/.venv/bin/hermes gateway run

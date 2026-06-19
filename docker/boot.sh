#!/bin/sh
# shellcheck shell=sh
# Container entrypoint dispatcher.
#
# On Fly Machines, Fly's init is PID 1 (it runs the SSH / api-proxy hallpass),
# so s6-overlay cannot run as PID 1 and the container reboot-loops with
# "s6-overlay-suexec: fatal: can only run as pid 1". Detect Fly via
# FLY_MACHINE_ID (Fly sets it on every machine) and boot the gateway directly.
#
# Everywhere else (docker run, Compose) keep the full s6 supervision tree,
# preserving the ENTRYPOINT+CMD arg-routing contract (chat, --tui, sleep,
# `gateway run`, …) by forwarding "$@" to the s6 main program unchanged.
if [ -n "${FLY_MACHINE_ID:-}" ]; then
    exec /opt/hermes/docker/fly-entrypoint.sh
fi
exec /init /opt/hermes/docker/main-wrapper.sh "$@"

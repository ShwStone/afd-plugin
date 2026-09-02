#!/bin/bash
# Minimal detached-spawn probe: writes /tmp/detached_probe.txt if the child survives.
setsid nohup /bin/bash -c "echo alive > /tmp/detached_probe.txt" < /dev/null &
echo probe-launched

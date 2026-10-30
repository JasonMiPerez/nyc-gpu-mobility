#!/usr/bin/env bash
# Run in your own Terminal so SSH can request your password normally.
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo 'Usage: bash scripts/connect_gpu.sh USER@LOGIN_SERVER USER@GPU_SERVER'
  exit 2
fi
socket_dir="/tmp/nyc-mobility-$(id -u)"
mkdir -p "$socket_dir"
chmod 700 "$socket_dir"
gateway="$1"
gpu="$2"
echo 'Connecting to the login server...'
if ! ssh -S "$socket_dir/gateway" -O check "$gateway" 2>/dev/null; then
  ssh -MNf -o ControlMaster=yes -o ControlPersist=4h -S "$socket_dir/gateway" "$gateway"
fi
echo 'Connecting to the GPU server through the login server...'
if ! ssh -S "$socket_dir/gpu" -O check "$gpu" 2>/dev/null; then
  ssh -MNf -o ControlMaster=yes -o ControlPersist=4h -S "$socket_dir/gpu" \
    -o "ProxyCommand=ssh -S $socket_dir/gateway -W %h:%p $gateway" "$gpu"
fi
echo "Connected. The reusable GPU SSH socket is $socket_dir/gpu"
ssh -S "$socket_dir/gpu" "$gpu" 'hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader'
echo 'Connections expire after four idle hours. No passwords were saved.'

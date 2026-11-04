#!/bin/bash
# Block until GPU 1 uses less than 1 GB, up to 40 minutes. Exit 1 if it never idles.
for i in $(seq 1 120); do
  u=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/GPU\[1\].*Used/{print $NF}')
  [ -n "$u" ] && [ "$u" -lt 1000000000 ] && exit 0
  sleep 20
done
exit 1

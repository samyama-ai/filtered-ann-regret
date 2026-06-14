#!/usr/bin/env bash
# Fetch ANN_SIFT1M (the real-geometry anchor, n=1e6, d=128): base 1M, queries 10k.
# ~168 MB tar from the canonical Jegou/IRISA mirror. Run from the repo root; lands in ./data/sift/.
set -euo pipefail
mkdir -p data && cd data
if [ ! -f sift/sift_base.fvecs ]; then
  echo "downloading ANN_SIFT1M (~168MB)..."
  curl -fsSL ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz -o sift.tar.gz \
    || curl -fsSL ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz --ftp-method nocwd -o sift.tar.gz
  tar xzf sift.tar.gz && rm -f sift.tar.gz
  echo "SIFT1M ready in data/sift/"
else
  echo "SIFT1M already present."
fi
ls -la sift/ | head

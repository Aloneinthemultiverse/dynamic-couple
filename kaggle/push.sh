#!/usr/bin/env bash
# Push the driver kernel to Kaggle (needs ~/.kaggle/kaggle.json).
set -euo pipefail
cd "$(dirname "$0")"
kaggle kernels push -p .
echo "pushed. Monitor: kaggle kernels status aloneinthemultiverse/dynamic-couple-driver"

#!/usr/bin/env bash
set -euo pipefail

# Push the notebook to Kaggle for training on a free P100 GPU.
#
# Prerequisites:
#   1. Install kaggle CLI: pip install kagglehub
#   2. Place kaggle.json in ~/.kaggle/ (get token from kaggle.com/settings)
#   3. Set secrets on Kaggle:
#      - HF_TOKEN, HF_REPO, GH_REPO
#      (Kaggle dataset page → Add-ons → Secrets)
#
# Usage:
#   bash kaggle_push.sh <kaggle-username>

if [ $# -lt 1 ]; then
  echo "Usage: $0 <kaggle-username>"
  echo "Example: $0 your-kaggle-username"
  exit 1
fi

KAGGLE_USER="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Update kernel-metadata.json with the username
sed -i "s/__KAGGLE_USER__/$KAGGLE_USER/g" "$SCRIPT_DIR/kernel-metadata.json"

cd "$SCRIPT_DIR"
kaggle kernels push

echo ""
echo "Done. View progress at: https://www.kaggle.com/$KAGGLE_USER/german-tutor-train"
echo ""
echo "After training completes, the adapter will be uploaded to Hugging Face Hub."
echo "Download it back: huggingface-cli download <HF_REPO> --local-dir qwen25_german_tutor_qlora"

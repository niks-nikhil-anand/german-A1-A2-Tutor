#!/usr/bin/env bash
set -euo pipefail

docker build -f output/Dockerfile.qwen_german_tutor_qlora -t qwen-german-tutor-qlora .
docker run --gpus all --rm -it   -v "$PWD":/workspace   -w /workspace   qwen-german-tutor-qlora   bash -lc "python3 train_qlora_german_tutor.py --config qlora_config_german_tutor.json"

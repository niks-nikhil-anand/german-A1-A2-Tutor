# Qwen2.5-3B German Tutor QLoRA

## Prerequisites
- Training/validation JSONL files in this directory
- GPU with at least 8 GB VRAM (or CPU — very slow)

## Train locally
```bash
python train_qlora_german_tutor.py --config qlora_config_german_tutor.json
```

Output goes to `qwen25_german_tutor_qlora/adapter/`.

## Train on Kaggle (free P100 GPU)

Push the notebook to Kaggle for free GPU training:

### One-time setup
1. Create a **Kaggle API token** at `kaggle.com/settings` → API → Create New Token. Save `kaggle.json`.
2. **Set secrets on Kaggle** for the notebook:
   - Go to the notebook on Kaggle → Add-ons → Secrets
   - Add `HF_TOKEN` (Hugging Face write token)
   - Add `HF_REPO` (e.g. `your-username/german-tutor-lora`)
   - Add `GH_REPO` (e.g. `your-org/Omni`)

### Push manually
```bash
pip install kagglehub
cp ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
bash kaggle_push.sh your-kaggle-username
```

### Automate via GitHub Actions
Add these **GitHub secrets** to your repository:
- `KAGGLE_USERNAME`
- `KAGGLE_KEY`

Then any push to `training/*.jsonl` or `training/*.py` will automatically push to Kaggle.

The workflow is at `.github/workflows/kaggle_train.yml`.

### After training
The adapter is uploaded to Hugging Face Hub. Download it back:
```bash
huggingface-cli download your-username/german-tutor-lora \
  --local-dir qwen25_german_tutor_qlora
```

## Evaluate
```bash
python eval_german_tutor.py \
  --base_model Qwen/Qwen2.5-3B-Instruct \
  --adapter_path qwen25_german_tutor_qlora/adapter \
  --eval_file eval.jsonl
```

## Merge adapter into full model
```bash
python merge_lora_german_tutor.py \
  --base_model Qwen/Qwen2.5-3B-Instruct \
  --adapter_path qwen25_german_tutor_qlora/adapter \
  --out_dir qwen25_german_tutor_merged
```

## Quick inference
```bash
python infer_german_tutor.py --message "Ich habe nicht Hunger."
```

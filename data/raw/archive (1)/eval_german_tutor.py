import argparse
import csv
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_model(base_model, adapter_path=None, merged_model_path=None, trust_remote_code=False):
    model_path = merged_model_path or base_model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map="auto", trust_remote_code=trust_remote_code)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, messages, max_new_tokens=220, temperature=0.2, top_p=0.9):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return text


def heuristic_checks(case, raw_output):
    result = {
        "valid_json": False,
        "has_question": False,
        "has_text_field": False,
        "no_direct_correction": True,
        "no_user_repeat": True,
    }
    lower = raw_output.lower()
    if "es heißt" in lower or "das ist falsch" in lower:
        result["no_direct_correction"] = False
    last_user = ""
    for msg in reversed(case["messages"]):
        if msg["role"] == "user":
            last_user = msg["content"].strip().lower()
            break
    try:
        payload = json.loads(raw_output)
        result["valid_json"] = True
        text = payload.get("text", "")
        result["has_text_field"] = isinstance(text, str)
        result["has_question"] = "?" in text
        if last_user and len(last_user) > 10 and last_user in text.lower():
            result["no_user_repeat"] = False
    except Exception:
        pass
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter_path", type=str, default="qwen25_german_tutor_qlora/adapter")
    parser.add_argument("--merged_model_path", type=str, default=None)
    parser.add_argument("--eval_file", type=str, default="eval.jsonl")
    parser.add_argument("--out_csv", type=str, default="eval_results.csv")
    parser.add_argument("--out_json", type=str, default="eval_summary.json")
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    cases = load_jsonl(args.eval_file)
    model, tokenizer = build_model(args.base_model, args.adapter_path, args.merged_model_path, args.trust_remote_code)

    rows = []
    summary = {"cases": len(cases), "valid_json": 0, "has_question": 0, "no_direct_correction": 0, "no_user_repeat": 0}
    for idx, case in enumerate(cases, start=1):
        raw = generate_response(model, tokenizer, case["messages"])
        checks = heuristic_checks(case, raw)
        for key in ["valid_json", "has_question", "no_direct_correction", "no_user_repeat"]:
            summary[key] += int(checks[key])
        rows.append({
            "id": idx,
            "tags": "|".join(case.get("tags", [])),
            "valid_json": checks["valid_json"],
            "has_question": checks["has_question"],
            "no_direct_correction": checks["no_direct_correction"],
            "no_user_repeat": checks["no_user_repeat"],
            "raw_output": raw,
        })

    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

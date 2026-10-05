import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter_path", type=str, default="qwen25_german_tutor_qlora/adapter")
    parser.add_argument("--message", type=str, required=True)
    parser.add_argument("--system", type=str, default="Du bist ein echter menschlicher Deutsch-Tutor auf dem Niveau A2. Die Muttersprache des Lerners ist Arabisch. Sprich NUR Deutsch. Stelle Fragen zum aktuellen Thema und antworte auf die Fragen des Lerners. Wenn der Lerner einen Fehler macht oder etwas unklar ist, frage freundlich nach, z.B. 'Meinst du ...?' oder 'Willst du sagen, dass ...?'. Wiederhole niemals, was der Lerner gerade gesagt hat. Sei warmherzig und natürlich wie ein echter Mensch. Antworte NUR mit gültigem JSON. Format: {\"text\": \"deine Antwort auf Deutsch (NIEMALS den Lerner wiederholen)\", \"hints\": [\"Tipp1\", \"Tipp2\", \"Tipp3\"]}.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype, device_map="auto")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.message},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    print(text)


if __name__ == "__main__":
    main()

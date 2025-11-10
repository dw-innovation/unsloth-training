<img width="1280" height="200" alt="Github-Banner_spot" src="https://github.com/user-attachments/assets/bec5a984-2f1f-44e7-b50d-cc6354d823cd" />

# 🏋️ SPOT Unsloth Training

This repository contains training scripts for **fine-tuning open-source LLMs** (e.g. LLaMA 3, Mistral) using the [Unsloth](https://github.com/unslothai/unsloth) library.  
It applies parameter-efficient tuning techniques such as **LoRA** on synthetic SPOT data.

---

## 🚀 Quickstart

To execute the python files in this repository, please use "scripts/train_unsloth.sh" and adjust the relevant parmeters.

---

## ⚙️ Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace token to download and push models. Required for accessing private weights or uploading to the Hub. |

> **Tip:** Store this token in a `.env` file or use an environment manager like `direnv`.

---

## 🔑 Features

- Fine-tunes SPOT-compatible LLMs on YAML ↔ sentence data
- Uses **LoRA adapters** for memory-efficient training
- Easily switchable model backbones (e.g., LLaMA 3, Mistral, Phi, Qwen)
- Compatible with data from [`datageneration`](https://github.com/dw-innovation/kid2-spot-datageneration)

---

## 🧩 Part of the SPOT System

This module is used to train the LLMs served by:
- [`central-nlp-api`](https://github.com/dw-innovation/kid2-spot-central-nlp-api) — for YAML generation from natural queries

It works in tandem with:
- [`spot-datageneration`](https://github.com/dw-innovation/kid2-spot-datageneration) — which produces training data

---

## 🧪 Notes

- Uses [Unsloth](https://github.com/unslothai/unsloth) for faster and lighter training
- Supports evaluation, early stopping, and adapter merging (if needed)
- Pre-configured for quantized base models

---

## 🔗 Related Docs

- [Main SPOT Repo](https://github.com/dw-innovation/kid2-spot)
- [SPOT Datageneration](https://github.com/dw-innovation/kid2-spot-datageneration)

---

## 🙌 Contributing

We welcome contributions of all kinds — from developers, journalists, mappers, and more!  
See [CONTRIBUTING.md](https://github.com/dw-innovation/kid2-spot/blob/main/CONTRIBUTING.md) for how to get started.
Also see our [Code of Conduct](https://github.com/dw-innovation/kid2-spot/blob/main/CODE_OF_CONDUCT.md).

---

## 📜 License

Licensed under [AGPLv3](../LICENSE).

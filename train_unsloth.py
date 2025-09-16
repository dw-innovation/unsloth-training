import torch
torch.set_float32_matmul_precision("high")
import os
import pandas as pd
from argparse import ArgumentParser
import jsonlines
from tqdm import tqdm
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, EarlyStoppingCallback, AutoTokenizer
from trl import SFTTrainer
from dotenv import load_dotenv
from peft import AutoPeftModelForCausalLM, PeftModel


"""
Script for fine-tuning and evaluating a language model (e.g., LLaMA) using the Unsloth framework and LoRA adapters.

This script supports:
- Fine-tuning a language model with LoRA on structured information extraction tasks.
- Saving the trained model locally or pushing it to HuggingFace Hub.
- Running inference (generation) on test data and saving predictions to disk.

Usage:
    python script.py --train --output_name ... [other args]
    python script.py --test --output_name ... [other args]
"""


final_prompt = """You are a structured geographic information extractor.
Your task is to read a natural sentence and convert it into a structured YAML representation of the area, entities, their properties, and their relations.

Always follow these rules:
- Only use information explicitly mentioned in the sentence.
- Do not invent data, values, or objects.
- Preserve exact measurements (units, numbers, formats).
- Do not explain or annotate the output - only produce the YAML.

Sentence:
{input}

YAML:
{output}"""


def train(output_name, model_name, train_path, dev_path, epochs, lora_r,lora_alpha, random_state, early_stopping,
          eval_steps, save_steps, auto_batch_size, batch_size, learning_rate, weight_decay, lr_scheduler,
          max_seq_length, dtype, load_in_4bit):
    """
    Fine-tunes a language model using LoRA adapters and the Unsloth framework.

    Args:
        output_name (str): Output directory name for saving model and logs.
        model_name (str): Name or path of the pretrained base model.
        train_path (str): Path to training dataset (TSV format).
        dev_path (str): Path to validation dataset (TSV format).
        epochs (int): Number of training epochs.
        lora_r (int): LoRA rank (r) value.
        lora_alpha (int): LoRA alpha value.
        random_state (int): Random seed for reproducibility.
        early_stopping (int): Number of evaluations with no improvement before early stopping.
        eval_steps (int): Evaluation interval in steps.
        save_steps (int): Model checkpoint saving interval in steps.
        auto_batch_size (bool): Whether to auto-tune batch size.
        batch_size (int): Batch size per device.
        learning_rate (float): Learning rate.
        weight_decay (float): Weight decay coefficient.
        lr_scheduler (str): Type of learning rate scheduler.
        max_seq_length (int): Maximum input sequence length.
        dtype (str or None): Data type (e.g., "bfloat16", "float16").
        load_in_4bit (bool): Whether to load the model in 4-bit quantized mode.

    Saves:
        - Fine-tuned model locally to `models/{output_name}_local`
        - Optionally pushes model/tokenizer to HuggingFace Hub under `{output_name}_lora`
        - Prints memory usage and training runtime.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        # trust_remote_code=True,
        attn_implementation="eager",
        device_map="auto",
        full_finetuning=False,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    """We now add LoRA adapters so we only need to update 1 to 10% of all parameters!"""
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=lora_alpha, # 8, 16, 32
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        # random_state=random_state,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

    def formatting_prompts_func(examples):
        inputs = examples["sentence"]
        outputs = examples["query"]
        texts = [
            final_prompt.format(input=inp.strip(), output=out.strip()) + EOS_TOKEN
            for inp, out in zip(inputs, outputs)
        ]
        return {"text": texts}

    # Load and preprocess training data
    train_ds = pd.read_csv(train_path, sep='\t')
    train_ds['sentence'] = train_ds['sentence'].str.lower()
    train_ds['query'] = train_ds['query'].str.lower()
    train_data = Dataset.from_pandas(train_ds)
    train_data = train_data.map(formatting_prompts_func, batched=True)

    # Load and preprocess validation data
    val_ds = pd.read_csv(dev_path, sep='\t')
    val_ds['sentence'] = val_ds['sentence'].str.lower()
    val_ds['query'] = val_ds['query'].str.lower()
    val_data = Dataset.from_pandas(val_ds)
    val_data = val_data.map(formatting_prompts_func, batched=True)

    # Set up trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=val_data,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping)],
            args=TrainingArguments(
                eval_strategy="steps",
                do_eval=True,
                save_strategy="steps",
                eval_steps=eval_steps,
                save_steps=save_steps,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                auto_find_batch_size=auto_batch_size,
                warmup_steps=int(len(train_ds) / 8),  # 5,
                # max_steps = 60,
                learning_rate=learning_rate,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=5,
                optim="adamw_bnb_8bit",
                gradient_accumulation_steps=4,
                dataloader_num_workers=2,
                logging_first_step=True,
                report_to="none",
                torch_compile=False,
                # optional speedups:
                tf32=True,
                weight_decay=weight_decay,
                lr_scheduler_type = lr_scheduler,
                # seed = random_state,
                output_dir=output_name,
                num_train_epochs=epochs,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
            ),
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')

    #model.save_pretrained_merged(f'models/{output_name}_merged_4bit', tokenizer, save_method = "merged_4bit_forced", token=hf_token)
    #model.push_to_hub_merged(fmodels/'{output_name}_merged_4bit', tokenizer, save_method="merged_4bit_forced", token=hf_token)

    model.save_pretrained(f'models/{output_name}_local')  # Local saving
    tokenizer.save_pretrained(f'models/{output_name}_local')

    model.push_to_hub(f'{output_name}_lora', token=hf_token)  # Online saving
    tokenizer.push_to_hub(f'{output_name}_lora', token=hf_token)  # Online saving

    # model.save_pretrained_merged(f'{output_name}_lora', tokenizer, save_method = "lora", token=hf_token)
    # model.push_to_hub_merged(f'{output_name}_lora', tokenizer, save_method = "lora", token=hf_token)

    # # # for cpu code
    # quant_methods = ["q2_k", "q3_k_m", "q4_k_m", "q5_k_m", "q6_k", "q8_0"]
    # cpu_output_name = f'{output_name}_cpu'
    # for quant in quant_methods:
    #     model.save_pretrained_gguf(cpu_output_name, tokenizer, quantization_method=quant)
    #     model.push_to_hub_gguf(cpu_output_name, tokenizer, quant, token=hf_token)


def test(output_name, max_seq_length, dtype, load_in_4bit):
    """
    Runs inference using a fine-tuned model on a test dataset.

    Args:
        output_name (str): Name of the fine-tuned model directory (under `models/`).
        max_seq_length (int): Maximum sequence length for tokenization.
        dtype (str or None): Data type to use during inference.
        load_in_4bit (bool): Whether to load the model in 4-bit quantized mode.

    Outputs:
        - Saves predictions to `test_results/{output_name}.jsonl`
    """
    if "gpt-oss" in output_name:
        adapters_dir = f'models/{output_name}_local'
        model = AutoPeftModelForCausalLM.from_pretrained(
            adapters_dir,
            load_in_4bit=True,
            torch_dtype="bfloat16",
            device_map="auto",
            attn_implementation="eager",
        )
        tokenizer = AutoTokenizer.from_pretrained(adapters_dir, use_fast=True)
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=f'models/{output_name}_local',  # YOUR MODEL YOU USED FOR TRAINING
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            # trust_remote_code=True,
            attn_implementation="eager",
            device_map="auto",
            # full_finetuning=False,
        )

        FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

    print("Has adapters:", hasattr(model, "peft_config"), getattr(model, "peft_config", None))

    # how many LoRA params?
    print("LoRA params:", sum("lora_" in n for n, _ in model.named_parameters()))

    test_sentences = pd.read_csv('data/sentences.txt', sep='\t')
    test_sentences = test_sentences['sentence'].tolist()

    with open(f'test_results/predictions.jsonl', 'a') as outfile:
        results = []
        for sentence in tqdm(test_sentences, total=len(test_sentences)):
            sentence = sentence.lower()
            inputs = tokenizer(
                [
                    final_prompt.format(
                        input=sentence.strip(), # input
                        output=""  # output - leave this blank for generation!
                    )
                ], return_tensors="pt").to("cuda")

            outputs = model.generate(
                **inputs,
                max_new_tokens=1048,
                use_cache=True,
                top_p=0.1,
                temperature=0.001,
            )
            outputs = tokenizer.batch_decode(outputs)[0]

            # input = outputs.split("### Input:")[1].split("### Response:")[0]
            respo = outputs.split("YAML:")[1].split("<|end_of_text|>")[0].strip()

            results.append({
                "sentence": sentence,
                "model_result": respo
            })

    with jsonlines.open(f'test_results/{output_name}.jsonl', mode='w') as writer:
        for sample in results:
            writer.write(sample)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--dev_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--lora_r', type=int, required=True)
    parser.add_argument('--lora_alpha', type=int, required=True)
    parser.add_argument('--random_state', type=int, required=True)
    parser.add_argument('--early_stopping', type=int, required=True)
    parser.add_argument('--eval_steps', type=int, required=True)
    parser.add_argument('--save_steps', type=int, required=True)
    parser.add_argument('--auto_batch_size', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--weight_decay', type=float, required=True)
    parser.add_argument('--lr_scheduler', type=str, required=True)
    parser.add_argument('--max_seq_length', type=int, default=2048)
    parser.add_argument('--dtype', type=str, default=None)
    parser.add_argument('--load_in_4bit', type=int, default=True)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    output_name = args.output_name
    model_name = args.model_name
    train_path = args.train_path
    dev_path = args.dev_path
    epochs = args.epochs
    lora_r = args.lora_r
    lora_alpha = args.lora_alpha
    random_state = args.random_state
    early_stopping = args.early_stopping
    eval_steps = args.eval_steps
    save_steps = args.save_steps
    auto_batch_size = args.auto_batch_size
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    lr_scheduler = args.lr_scheduler
    max_seq_length = args.max_seq_length
    dtype = args.dtype
    load_in_4bit = args.load_in_4bit
    _train = args.train
    _test = args.test

    # Convert CLI inputs to proper types
    if dtype == "-1":
        dtype = None
    if load_in_4bit == 1:
        load_in_4bit = True
    else:
        load_in_4bit = False
    if auto_batch_size == 1:
        auto_batch_size = True
    else:
        auto_batch_size = False

    if _train:
        train(output_name, model_name, train_path, dev_path, epochs, lora_r,lora_alpha, random_state, early_stopping,
              eval_steps, save_steps, auto_batch_size, batch_size, learning_rate, weight_decay, lr_scheduler,
              max_seq_length, dtype, load_in_4bit)
    if _test:
        test(output_name, max_seq_length, dtype, load_in_4bit)
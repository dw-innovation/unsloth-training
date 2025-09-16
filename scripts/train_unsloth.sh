# REQUIRED FOR GPT-OSS
export UNSLOTH_COMPILE_DISABLE=1          # disables torch.compile on Unsloth hooks
export TORCHDYNAMO_DISABLE=1              # belt & suspenders: fully bypass Dynamo
export UNSLOTH_DISABLE_FUSED_CE_LOSS=1    # if present, skips the fused CE kernel

DATE=14082025
PARAMETER_VERSION=7
MODEL="gpt-oss-20b-unsloth" # "gpt-oss-20b-unsloth" # Mistral-Small-24B-Base-2501-unsloth # llama-3-8b # Meta-Llama-3.1-8B # gemma-7b Phi-3-medium-4k-instruct Qwen2.5-14B Qwen2.5-32B
VERSION_NAME='v18_120fix_75k'
EPOCHS=5 # Default: 3
LORA_R=32
LORA_ALPHA=64
LORA_RANDOM_STATE=3407
EARLY_STOPPING=10
EVAL_STEPS=200
SAVE_STEPS=200
AUTO_BATCH_SIZE=1
BATCH_SIZE=8
LEARNING_RATE=1e-5
WEIGHT_DECAY=0.01
LR_SCHEDULER='cosine'
MAX_SEQ_LENGTH=2048  # Choose any! We auto support RoPE Scaling internally!
DTYPE="-1"  # Leave at -1 for auto-detection, set to Float16 for Tesla T4, V100, or Bfloat16 for Ampere+
LOAD_IN_4BIT=1  # Use 1 for True, 0 for False
PROMPT_VERSION=v2

OUTPUT_NAME="spot_${MODEL}_ep${EPOCHS}_training_ds_${VERSION_NAME}_param-${PARAMETER_VERSION}_prompt-${PROMPT_VERSION}"
#OUTPUT_NAME="spot_llama-3-8b_ep10_training_ds_v16_3-17_1-2_lora" # deployed model

MODEL_NAME=unsloth/${MODEL}-bnb-4bit
TRAIN_PATH=data/train_${VERSION_NAME}.tsv
DEV_PATH=data/dev_${VERSION_NAME}.tsv
PROMPT_FILE=data/zero_shot_cot_prompt_v2.txt

echo Training $OUTPUT_NAME
# \\CUDA_VISIBLE_DEVICES="1" python -m train_unsloth \
# CUDA_VISIBLE_DEVICES="1" screen -L -Logfile logs/${OUTPUT_NAME}_${DATE}.txt python -m train_unsloth \
CUDA_VISIBLE_DEVICES="1" python -m train_unsloth \
  --output_name $OUTPUT_NAME \
  --model_name $MODEL_NAME \
  --train_path $TRAIN_PATH \
  --dev_path $DEV_PATH \
  --epochs $EPOCHS \
  --lora_r $LORA_R \
  --lora_alpha $LORA_ALPHA \
  --random_state $LORA_RANDOM_STATE \
  --early_stopping $EARLY_STOPPING \
  --eval_steps $EVAL_STEPS \
  --save_steps $SAVE_STEPS \
  --auto_batch_size $AUTO_BATCH_SIZE \
  --batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --weight_decay $WEIGHT_DECAY \
  --lr_scheduler $LR_SCHEDULER \
  --max_seq_length $MAX_SEQ_LENGTH \
  --dtype $DTYPE \
  --load_in_4bit $LOAD_IN_4BIT \
  --test \
  2>&1 | tee full_log.txt
  #--train \
  #--prompt_file $PROMPT_FILE

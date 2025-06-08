from transformers import TrainingArguments
from peft import LoraConfig

def get_peft_config():
    """Get the LoRA configuration for parameter-efficient fine-tuning"""
    lora_target_modules = [
        "q_proj",
        "up_proj",
        "o_proj",
        "k_proj",
        "down_proj",
        "gate_proj",
        "v_proj",
    ]
    
    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules=lora_target_modules,
        task_type="CAUSAL_LM",
    )
    
    return peft_config

def get_training_args():
    """Get the training arguments for the trainer"""
    args = TrainingArguments(
        output_dir="Llama-3.2-1B-Instruct-MEdQuAD-v2",  # directory to save and repository id
        num_train_epochs=4,                             # number of training epochs
        per_device_train_batch_size=3,                  # batch size per device during training
        gradient_accumulation_steps=2,                  # number of steps before performing a backward/update pass
        gradient_checkpointing=True,                    # use gradient checkpointing to save memory
        optim="adamw_torch_fused",                      # use fused adamw optimizer
        logging_steps=500,                              # log every 500 steps
        save_strategy="epoch",                          # save checkpoint every epoch
        learning_rate=1e-4,                             # learning rate, based on QLoRA paper
        fp16=True,                                      # use fp16 precision
        max_grad_norm=0.3,                              # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,                              # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",                   # use constant learning rate scheduler
        push_to_hub=False,                              # push model to hub
        report_to="tensorboard",                        # report metrics to tensorboard
    )
    
    return args

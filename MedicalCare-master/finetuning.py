from huggingface_hub import login
from data_preprocessing import load_and_preprocess_data, prepare_datasets
from model_utils import setup_model_tokenizer, generate_response, generate_predictions
from evaluation import evaluate_model_performance
from training_config import get_training_args, get_peft_config

# Login to Hugging Face
login(token="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")

# Load and preprocess data
df_cleaned = load_and_preprocess_data()

# Prepare datasets
dataset_dict, tokenized_datasets = prepare_datasets(df_cleaned)

# Setup model and tokenizer
model, tokenizer, device = setup_model_tokenizer()

# Get training configurations
peft_config = get_peft_config()
training_args = get_training_args()

# Train the model
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    args=training_args,
    peft_config=peft_config,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    processing_class=tokenizer,
)

trainable = [name for name, param in trainer.model.named_parameters() if param.requires_grad]
print(f"Trainable parameters ({len(trainable)}):", trainable[:5])
trainer.train()
trainer.save_model()
trainer.push_to_hub()

# Generate predictions and evaluate model
predictions, references = generate_predictions(tokenized_datasets['test'], model, tokenizer, device)
evaluate_model_performance(predictions, references)
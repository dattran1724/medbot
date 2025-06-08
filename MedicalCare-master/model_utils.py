import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader

def setup_model_tokenizer():
    """Initialize and set up the model and tokenizer"""
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.add_special_tokens({'pad_token': '<pad>', 'bos_token': '<question>', 'eos_token': '<end>', 'sep_token': '<answer>'})
    
    # Resize token embeddings to match new tokenizer size
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.config.use_cache = False
    
    return model, tokenizer, device

def tokenize_function(examples, tokenizer):
    """Tokenize examples for training"""
    tokenized = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=1024)
    tokenized["question"] = examples["Question"]
    tokenized["answer"] = examples["Answer"]
    return tokenized

def prepare_tokenized_datasets(dataset_dict, tokenizer):
    """Prepare tokenized datasets for training"""
    return dataset_dict.map(
        lambda examples: tokenize_function(examples, tokenizer), 
        batched=True,
        num_proc=4, 
        remove_columns=['text']
    )

def generate_response(model, tokenizer, prompt, device):
    """Generate a response from the model for a given prompt"""
    model.to(device)
    model.eval()

    formatted_prompt = f"""
    You are a medical professional with extensive knowledge of diseases, symptoms, and treatments. Please answer the following questions with accurate and clear information.
    <question> {prompt} <answer>"""

    inputs = tokenizer.encode(formatted_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens=200,
            num_return_sequences=1,
            do_sample=True,
            num_beams=5,
            top_k=20,
            top_p=0.6,
            temperature=0.7,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_response = decoded_output.replace(formatted_prompt, "").strip()

    return generated_response

def generate_predictions(dataset, model, tokenizer, device, batch_size=16, cutoff=512):
    """Generate predictions for evaluation"""
    model.to(device)
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size)
    predictions = []
    references = []

    for batch in dataloader:
        # Tokenize the inputs
        inputs = tokenizer(batch['question'], return_tensors='pt', padding=True, truncation=True, max_length=cutoff)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        # Generate predictions without updating the model parameters
        with torch.no_grad():
            outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=200)
        
        # Decode the generated tokens into strings
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Extend the predictions and references lists
        refs = batch['answer']
        predictions.extend(preds)
        references.extend(refs)

    return predictions, references

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from evaluate import load

def evaluate_model_performance(predictions, references):
    """Evaluate model performance using multiple metrics"""
    if len(predictions) == 0 or len(references) == 0:
        print("No predictions or references to evaluate.")
        return
    
    # Calculate BLEU, ROUGE scores
    metrics = calculate_text_metrics(predictions, references)
    
    # Calculate BERTScore
    bertscore = load("bertscore")
    bert_results = bertscore.compute(predictions=predictions, references=references, lang="en")
    avg_bertscore = sum(bert_results['f1'])/len(bert_results['f1'])
    
    # Calculate METEOR score
    avg_meteor = calculate_meteor_score(predictions, references)
    
    # Print all results
    print(f"BLEU-1 Score: {metrics['bleu1']}")
    print(f"BLEU-4 Score: {metrics['bleu4']}")
    print(f"ROUGE-1 Score: {metrics['rouge1']}")
    print(f"ROUGE-2 Score: {metrics['rouge2']}")
    print(f"ROUGE-L Score: {metrics['rougeL']}")
    print(f"BERTScore (F1): {avg_bertscore:.4f}")
    print(f"METEOR Score: {avg_meteor}")

def calculate_text_metrics(predictions, references):
    """Calculate BLEU and ROUGE metrics"""
    # Initialize metrics
    smooth = SmoothingFunction().method4
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Score containers
    bleu1_scores = []
    bleu4_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for ref, pred in zip(references, predictions):
        # BLEU-1 and BLEU-4
        bleu1 = sentence_bleu([ref.split()], pred.split(), weights=(1, 0, 0, 0), smoothing_function=smooth)
        bleu4 = sentence_bleu([ref.split()], pred.split(), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
        bleu1_scores.append(bleu1)
        bleu4_scores.append(bleu4)

        # ROUGE-1, ROUGE-2, and ROUGE-L
        rouge_scores = rouge.score(ref, pred)
        rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
        rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
        rougeL_scores.append(rouge_scores['rougeL'].fmeasure)
    
    # Check if there are any scores to average
    if len(bleu1_scores) == 0:
        avg_bleu1 = avg_bleu4 = avg_rouge1 = avg_rouge2 = avg_rougeL = 0
    else:
        # Average the scores
        avg_bleu1 = sum(bleu1_scores) / len(bleu1_scores)
        avg_bleu4 = sum(bleu4_scores) / len(bleu4_scores)
        avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
        avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
        avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)
    
    return {
        'bleu1': avg_bleu1,
        'bleu4': avg_bleu4,
        'rouge1': avg_rouge1,
        'rouge2': avg_rouge2,
        'rougeL': avg_rougeL
    }

def calculate_meteor_score(predictions, references):
    """Calculate METEOR score"""
    # Tokenize the sentences
    tokenized_references = [ref.split() for ref in references]
    tokenized_hypotheses = [hyp.split() for hyp in predictions]
    
    # Calculate METEOR scores for each reference-hypothesis pair
    meteor_scores = [meteor_score([ref], hyp) for ref, hyp in zip(tokenized_references, tokenized_hypotheses)]
    
    # Average the scores
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0
    
    return avg_meteor

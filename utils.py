import evaluate
import numpy as np

# Загрузка метрик
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
# spice = evaluate.load("spice") # SPICE требует Java и скачивания зависимостей, раскомментируйте если среда позволяет

def compute_metrics(eval_pred, tokenizer):
    preds, labels = eval_pred
    
    # Декодируем предсказания
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # Заменяем -100 на pad_token_id для корректного декодирования меток
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Пост-обработка текста
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # Вычисление BLEU
    bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    
    # Вычисление ROUGE
    rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    
    # SPICE (может быть долгим и требовать памяти)
    # spice_score = spice.compute(predictions=decoded_preds, references=decoded_labels)

    result = {
        "bleu": bleu_score["bleu"],
        "rouge1": rouge_score["rouge1"],
        "rouge2": rouge_score["rouge2"],
        "rougeL": rouge_score["rougeL"],
        # "spice": spice_score["spice"]
    }
    
    return result

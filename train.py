import argparse
import torch
from datasets import load_dataset
from transformers import (
    VisionEncoderDecoderModel, 
    AutoTokenizer, 
    AutoImageProcessor, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    default_data_collator
)
from utils import compute_metrics
import os

def main(args):
    # 1. Настройка устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Загрузка датасета COCO
    print("Loading dataset (this may take a while)...")
    # Используем split='train[:5000]' для примера, чтобы не качать 20Гб сразу, 
    # если args.limit_data не задан, можно качать всё (но это долго для Colab)
    
    try:
        # Пытаемся загрузить с trust_remote_code=True (для datasets 2.15.0)
        dataset = load_dataset("HuggingFaceM4/COCO", script_version="2014", trust_remote_code=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Try downgrading datasets library: pip install datasets==2.15.0")
        return

    # Ограничение данных для теста (чтобы обучение не шло часами)
    if args.limit_data:
        print(f"Limiting dataset to first {args.limit_data} examples.")
        # COCO обычно загружается как DatasetDict
        if 'train' in dataset:
            dataset['train'] = dataset['train'].select(range(args.limit_data))
        if 'validation' in dataset:
            dataset['validation'] = dataset['validation'].select(range(args.limit_data // 5))

    # 3. Фильтрация черно-белых изображений (оставляем только RGB)
    def filter_grayscale(example):
        return example['image'].mode == 'RGB'

    print("Filtering non-RGB images...")
    dataset = dataset.filter(filter_grayscale)

    # 4. Инициализация Токенайзера и Процессора изображений
    # Encoder (Image)
    image_processor = AutoImageProcessor.from_pretrained(args.encoder_model)
    # Decoder (Text)
    tokenizer = AutoTokenizer.from_pretrained(args.decoder_model)
    
    # GPT2 требует установки pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 5. Создание модели VisionEncoderDecoder
    print("Initializing model...")
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        args.encoder_model, 
        args.decoder_model
    )
    
    # Настройка токенов модели
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Настройка параметров генерации
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.max_length = args.max_length
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    model.to(device)

    # 6. Препроцессинг данных
    def transforms(examples):
        images = [image.convert("RGB") for image in examples['image']]
        # COCO dataset structure: 'sentences': [{'raw': '...', ...}, ...]
        # Берем первую подпись
        captions = [s[0]['raw'] for s in examples['sentences']]
        
        inputs = image_processor(images=images, return_tensors="pt")
        targets = tokenizer(captions, padding="max_length", max_length=args.max_length, truncation=True)
        
        inputs["labels"] = targets["input_ids"]
        return inputs

    print("Processing dataset...")
    processed_dataset = dataset.map(
        transforms, 
        batched=True, 
        remove_columns=dataset['train'].column_names,
        batch_size=args.batch_size
    )
    
    # Установка формата Torch
    processed_dataset.set_format(type="torch")

    # 7. Настройка аргументов обучения
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        evaluation_strategy="epoch", # Оценка каждую эпоху
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="tensorboard", # Мониторинг
        fp16=torch.cuda.is_available(), # Использовать fp16 если есть GPU
        push_to_hub=False,
    )

    # Функция метрик с замыканием для доступа к токенайзеру
    def compute_metrics_closure(eval_pred):
        return compute_metrics(eval_pred, tokenizer)

    # 8. Инициализация Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=image_processor, # Здесь processor, так как он обрабатывает вход (картинки)
        args=training_args,
        train_dataset=processed_dataset['train'],
        eval_dataset=processed_dataset['validation'],
        data_collator=default_data_collator,
        compute_metrics=compute_metrics_closure,
    )

    # 9. Запуск обучения
    print("Starting training...")
    trainer.train()

    # 10. Сохранение модели
    print(f"Saving model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    image_processor.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Аргументы архитектуры
    parser.add_argument("--encoder_model", type=str, default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--decoder_model", type=str, default="gpt2")
    
    # Аргументы обучения
    parser.add_argument("--output_dir", type=str, default="./image_captioning_model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8) # Уменьшите, если не хватает памяти
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--limit_data", type=int, default=None, help="Set to e.g., 1000 for quick testing")

    args = parser.parse_args()
    main(args)

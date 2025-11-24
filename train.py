import argparse
import torch
from datasets import load_from_disk
from transformers import (
    VisionEncoderDecoderModel, 
    AutoTokenizer, 
    AutoImageProcessor, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    default_data_collator
)
from utils import compute_metrics
import warnings

warnings.filterwarnings("ignore")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Загрузка данных с диска
    print(f"Загрузка данных из {args.data_path}...")
    try:
        dataset = load_from_disk(args.data_path)
        print(f"Загружено: {dataset}")
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        return

    # Загрузка токенайзера (из той же папки, чтобы совпадал конфиг)
    print("Загрузка токенайзера/процессора...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.data_path)
        image_processor = AutoImageProcessor.from_pretrained(args.data_path)
    except:
        print("Внимание: Не удалось загрузить токенайзер...")
        tokenizer = AutoTokenizer.from_pretrained(args.decoder_model)
        image_processor = AutoImageProcessor.from_pretrained(args.encoder_model)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # Инициализация модели
    print("Инициализация модели...")
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        args.encoder_model, 
        args.decoder_model
    )
    
    # Конфигурация спец. токенов
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    
    # Параметры генерации
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.max_length = args.max_length
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    model.to(device)

    # Обучение
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=500,
        logging_steps=50,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="tensorboard",
        fp16=torch.cuda.is_available(),
    )

    def compute_metrics_closure(eval_pred):
        return compute_metrics(eval_pred, tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=image_processor, # Важно передать image_processor
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=default_data_collator,
        compute_metrics=compute_metrics_closure,
    )

    print("Начало обучения...")
    trainer.train()

    print(f"Сохранение модели в {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    image_processor.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Аргументы путей
    parser.add_argument("--data_path", type=str, default="./processed_data")
    parser.add_argument("--output_dir", type=str, default="./image_captioning_model")
    
    # Аргументы модели (нужны для init, даже если данные готовы)
    parser.add_argument("--encoder_model", type=str, default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--decoder_model", type=str, default="gpt2")
    
    # Гиперпараметры
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--max_length", type=int, default=32)

    args = parser.parse_args()
    main(args)

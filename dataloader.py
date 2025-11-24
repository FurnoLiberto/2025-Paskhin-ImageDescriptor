import argparse
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoImageProcessor
import warnings

warnings.filterwarnings("ignore")

def main(args):
    print(f" Подготовка данных")
    
    # Загрузка датасета (через локальный COCO.py)
    print("Загрузка датасета...")
    try:
        # Если скрипта нет, скачаем его
        if not os.path.exists("COCO.py"):
            print("Скачивание COCO.py...")
            os.system("wget https://huggingface.co/datasets/HuggingFaceM4/COCO/resolve/main/COCO.py -O COCO.py")
            
        dataset = load_dataset("./COCO.py", "2014")
    except Exception as e:
        print(f"Ошибка скачиванияt: {e}")
        return

    # Ограничение (для теста или быстрого пайплайна)
    if args.limit_data:
        print(f"Ограничение до {args.limit_data} примеров.")
        if 'train' in dataset:
            dataset['train'] = dataset['train'].select(range(args.limit_data))
        if 'validation' in dataset:
            dataset['validation'] = dataset['validation'].select(range(args.limit_data // 5))
        if 'test' in dataset: # Удаляем тест, чтобы не тратить место, если он не нужен
             del dataset['test']

    # Инициализация процессоров
    print(f"Загрузка инструментов: {args.encoder_model} & {args.decoder_model}")
    image_processor = AutoImageProcessor.from_pretrained(args.encoder_model)
    tokenizer = AutoTokenizer.from_pretrained(args.decoder_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3Фильтрация (RGB + Text check)
    def filter_corrupt_data(example):
        try:
            if example['image'].mode != 'RGB':
                return False
            
            sents = example['sentences']
            # Проверка на пустоту
            if isinstance(sents, list) and len(sents) == 0: return False
            if isinstance(sents, dict) and 'raw' in sents and len(sents['raw']) == 0: return False
            
            return True
        except:
            return False

    print("Фильрация изображение (оставить только RGB)...")
    dataset = dataset.filter(filter_corrupt_data)

    # Препроцессинг (Map)
    def transforms(examples):
        # Картинки
        images = [img.convert("RGB") for img in examples['image']]
        
        # Текст (с обработкой сложной структуры COCO)
        captions = []
        for sents in examples['sentences']:
            caption = ""
            try:
                # Если словарь {'raw': ...}
                if isinstance(sents, dict):
                    if 'raw' in sents:
                        val = sents['raw']
                        caption = val[0] if isinstance(val, list) and val else str(val)
                    else:
                        caption = str(list(sents.values())[0])
                # Если список [{'raw':...}]
                elif isinstance(sents, list):
                    if len(sents) > 0:
                        first = sents[0]
                        caption = first.get('raw', str(first)) if isinstance(first, dict) else str(first)
            except:
                caption = ""
            captions.append(caption)

        # Токенизация
        inputs = image_processor(images=images, return_tensors="pt")
        targets = tokenizer(captions, padding="max_length", max_length=args.max_length, truncation=True)
        
        inputs["labels"] = targets["input_ids"]
        return inputs

    print("Применение трансформации...")
    processed_dataset = dataset.map(
        transforms, 
        batched=True, 
        remove_columns=dataset['train'].column_names, # Удаляем сырые картинки
        batch_size=args.batch_size
    )
    
    # Устанавливаем формат
    processed_dataset.set_format(type="torch")

    # Сохранение на диск
    print(f"Сохранение обработанных данных в : {args.save_path}")
    processed_dataset.save_to_disk(args.save_path)
    
    # Сохраняем токенайзер туда же, чтобы при обучении использовать именно его
    print("Сохранение токенайзера и конфига...")
    tokenizer.save_pretrained(args.save_path)
    image_processor.save_pretrained(args.save_path)
    
    print("Готово")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_model", type=str, default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--decoder_model", type=str, default="gpt2")
    parser.add_argument("--limit_data", type=int, default=None)
    parser.add_argument("--save_path", type=str, default="./processed_data")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=32)
    
    args = parser.parse_args()
    main(args)

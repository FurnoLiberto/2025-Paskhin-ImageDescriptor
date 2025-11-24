# Обучение модели описания изображений

Проект по обучению модели описания изображений (Image Captioning) на датасете COCO 2014.

## Архитектура
Мы используем гибридную архитектуру на основе трансформеров:
- Encoder (Зрение): `google/vit-base-patch16-224-in21k` (Vision Transformer). Извлекает признаки из изображения.
- Decoder (Текст): `gpt2`. Генерирует текст на основе признаков от энкодера.

Библиотеки: `transformers`, `datasets`, `accelerate`.

## Данные
- Использован датасет COCO 2014.
- Применена фильтрация: оставлены только цветные (RGB) изображения.
- Для демонстрации работоспособности использована выборка из 2000 изображений.

## Результаты обучения
Обучение проводилось в течение 10 эпох.
- **Training Loss:** снизился с 3.81 до 0.9.
- **Eval ROUGE-2:** 0.005 (модель начала угадывать пары слова).
- **Eval BLEU:** ~0.0 (из-за малого размера выборки модель не выучила сложные грамматические конструкции для точного совпадения).

<img width="1119" height="392" alt="изображение" src="https://github.com/user-attachments/assets/df67679a-35b8-42b8-ab6e-e8fe5248db25" />

<img width="1180" height="394" alt="изображение" src="https://github.com/user-attachments/assets/9c530d85-b2af-42b3-9ab3-887421fcb847" />

<img width="1144" height="420" alt="изображение" src="https://github.com/user-attachments/assets/ed2a0b4f-06a3-49c6-a83a-1bb856391b48" />

<img width="357" height="357" alt="изображение" src="https://github.com/user-attachments/assets/32034338-c5f8-4eb2-897a-7780d70d0179" />



Графики обучения (Loss) доступны в папке `runs/` (TensorBoard).

## Как запустить

1. Подготовка данных:
  
   ```bash
   python dataloader.py --limit_data 50000 --save_path "./processed_data"

2. Обучение:
  
   ```bash
   python train.py --data_path "./processed_data" --epochs 10

3. Инференс (проверка):
  
   ```bash
   python inference.py --image_path "URL_КАРТИНКИ"


## Выводы
Эксперимент подтвердил работоспособность архитектуры ViT + GPT2.
Модель демонстрирует снижение функции потерь (Loss).

import argparse
import torch
from PIL import Image
import requests
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoImageProcessor

def predict_caption(image_path, model_path, max_length=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка модели и инструментов
    model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    image_processor = AutoImageProcessor.from_pretrained(model_path)

    # Загрузка изображения
    if image_path.startswith("http"):
        image = Image.open(requests.get(image_path, stream=True).raw)
    else:
        image = Image.open(image_path)
    
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Препроцессинг
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values.to(device)

    # Генерация
    output_ids = model.generate(
        pixel_values, 
        max_length=max_length, 
        num_beams=4, 
        return_dict_in_generate=True
    )
    
    preds = tokenizer.batch_decode(output_ids[0], skip_special_tokens=True)
    return preds[0].strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="URL or local path to image")
    parser.add_argument("--model_path", type=str, default="./image_captioning_model")
    
    args = parser.parse_args()
    
    caption = predict_caption(args.image_path, args.model_path)
    print(f"Generated Caption: {caption}")

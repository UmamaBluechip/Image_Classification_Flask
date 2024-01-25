from transformers import AutoConfig,ViTImageProcessor,ViTForImageClassification,AutoModel
from PIL import Image
import io

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')


def image_classifier(image_bytes):

    pil_image = Image.open(io.BytesIO(image_bytes))

    inputs = processor(images=pil_image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits

    logits_np = logits.detach().cpu().numpy()
    predicted_class_idx = logits_np.argsort()[0][-1]
    predicted_class = model.config.id2label[predicted_class_idx]

    result = {"class": predicted_class}

    return result

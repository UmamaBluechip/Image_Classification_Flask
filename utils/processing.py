from transformers import AutoConfig,ViTImageProcessor,ViTForImageClassification,AutoModel
import base64
import os

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

def image_classifier(image):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits

    logits_np = logits.detach().cpu().numpy()
    logits_args = logits_np.argsort()[0][-3:]

    prediction_classes = [model.config.id2label[predicted_class_idx] for predicted_class_idx in logits_args]
    
    result = {}
    for i,item in enumerate(prediction_classes):
        result[item] = logits_np[0][i]

    return result

import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

class AgePredictor:
    def __init__(self):
        # We use a lightweight pre-trained Vision Transformer model for age classification
        self.model_name = "nateraw/vit-age-classifier"
        self.processor = ViTImageProcessor.from_pretrained(self.model_name)
        self.model = ViTForImageClassification.from_pretrained(self.model_name)
        self.model.eval() # Set model to evaluation mode
        
    def predict(self, image: Image.Image) -> str:
        """
        Predicts the age range from a PIL Image.
        """
        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Preprocess the image
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Get the predicted class
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_age_range = self.model.config.id2label[predicted_class_idx]
        
        return predicted_age_range

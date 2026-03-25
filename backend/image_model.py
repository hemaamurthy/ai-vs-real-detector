import torch
from torchvision import models, transforms
from PIL import Image
import os

# correct path
model_path = os.path.join(os.path.dirname(__file__), "..", "image_model.pth")

# load model
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

classes = ["AI Generated", "Real Image"]

def detect_image(path):
    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)

    confidence, predicted = torch.max(probs, 1)

    return f"{classes[predicted.item()]} ({confidence.item()*100:.2f}%)"
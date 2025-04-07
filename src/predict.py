import torch
from deepfake_model import DeepfakeCNN
from utils import preprocess_image
import sys

def predict(image_path):
    model = DeepfakeCNN()
    model.load_state_dict(torch.load("model/deepfake_cnn.pth", map_location='cpu'))
    model.eval()

    image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image)
        prediction = "Real" if output.item() > 0.5 else "Fake"
        print(f"Prediction: {prediction}")

if __name__ == "__main__":
    predict(sys.argv[1])

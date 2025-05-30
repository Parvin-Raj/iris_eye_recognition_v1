import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from flask import Flask, request, jsonify
import cv2

app = Flask(__name__)  

class IrisEncoder(nn.Module):
    def __init__(self, weights_path=None, device='cpu'):
        super(IrisEncoder, self).__init__()
        # Initialize model architecture WITHOUT pretrained weights
        convnext = models.convnext_tiny(weights=None)  # No pretrained weights
        self.backbone = nn.Sequential(*list(convnext.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        if weights_path is not None:
            # Load your trained weights
            state_dict = torch.load(weights_path, map_location=device)
            # If your saved file is a dict with more than just state_dict,
            # adjust this line accordingly:
            # state_dict = state_dict['state_dict']
            self.load_state_dict(state_dict)
        
        self.to(device)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x

# Set device and load your trained model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = IrisEncoder(weights_path='iris_encoder_convnext.pth', device=device)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_embedding(model, iris_img, device='cpu'):
    model.eval()
    with torch.no_grad():
        x = transform(iris_img).unsqueeze(0).to(device)
        embedding = model(x)
        embedding = embedding.cpu().numpy().flatten()
        embedding = embedding / np.linalg.norm(embedding)
    return embedding


@app.route('/register', methods=['POST'])
def register():
    return jsonify({'message': 'Register endpoint'})

@app.route('/recognize', methods=['POST'])
def recognize():
    return jsonify({'message': 'Recognize endpoint'})

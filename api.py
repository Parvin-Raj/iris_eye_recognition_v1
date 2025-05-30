import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from flask import Flask, request, jsonify
import cv2
from faiss_index import FaissIndex  # your FaissIndex class file

app = Flask(__name__)

class IrisEncoder(nn.Module):
    def __init__(self):
        super(IrisEncoder, self).__init__()
        convnext = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(convnext.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = IrisEncoder().to(device)

# Initialize FaissIndex with embedding_dim=768 (ConvNeXt Tiny last layer output size)
faiss_index = FaissIndex(embedding_dim=768)  

@app.route('/register', methods=['POST'])
def register():
    if 'iris_image' not in request.files or 'user_id' not in request.form:
        return jsonify({'error': 'Missing iris_image file or user_id'}), 400
    
    user_id = request.form['user_id']
    file = request.files['iris_image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Invalid image'}), 400

    embedding = extract_embedding(model, img, device)

    # Add embedding with metadata (user_id)
    faiss_index.add(embedding, {'user_id': user_id})

    # Count how many images this user has registered so far
    count = sum(1 for meta in faiss_index.metadata if meta.get('user_id') == user_id)

    return jsonify({'message': f'User {user_id} registered successfully with {count} image(s)'})


@app.route('/recognize', methods=['POST'])
def recognize():
    if 'iris_image' not in request.files:
        return jsonify({'error': 'Missing iris_image file'}), 400
    
    file = request.files['iris_image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Invalid image'}), 400

    query_embedding = extract_embedding(model, img, device)

    if faiss_index.index.ntotal == 0:
        return jsonify({'error': 'No users registered'}), 400

    distances, results = faiss_index.search(query_embedding, top_k=1)

    threshold = 0.8  # similarity threshold, tune this for your case
    if results and distances[0] < (1 - threshold):  # L2 distance, so smaller is better
        user_id = results[0]['user_id']
        similarity = 1 - distances[0]  # convert distance to similarity approx
        return jsonify({'user_id': user_id, 'similarity': float(similarity), 'message': 'User recognized'})
    else:
        return jsonify({'message': 'No matching user found', 'similarity': float(1 - distances[0])}), 404

if __name__ == '__main__':
    app.run(debug=True)


## File Structure

IRIS_RECOGNITION/
├── __pycache__/
├── app/
│   ├── __pycache__/
│   ├── api.py
│   ├── capture.py
│   ├── encoder.py
│   ├── normalize.py
│   └── segment.py
├── data/
│   ├── iris_dataset/
│   ├── embeddings.npy
│   ├── faiss.index
│   └── metadata.json
├── env/
├── models/
│   ├── convnext_tiny-983f1562.pth
│   ├── iris_encoder_convnext.pth
│   └── faiss_index.py
├── requirements.txt
├── run.py
└── train_convnext_tiny.py

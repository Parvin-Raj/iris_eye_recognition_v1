import faiss
import numpy as np
import json
import os

class FaissIndex:
    def __init__(self, embedding_dim=128,
                 index_path='data/faiss.index',
                 metadata_path='data/metadata.json'):
        self.embedding_dim = embedding_dim
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.metadata = []

        # Initialize index - try to load from disk, else create new
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = faiss.IndexFlatL2(self.embedding_dim)

        # Load metadata or create empty list
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                try:
                    self.metadata = json.load(f)
                except json.JSONDecodeError:
                    self.metadata = []
        else:
            self.metadata = []

    def add(self, embedding, metadata):
        print("Adding metadata:", metadata)  # <-- debug print here
        embedding = np.array([embedding]).astype('float32')
        self.index.add(embedding)
        self.metadata.append(metadata)
        self.save()


    def search(self, embedding, top_k=1):
        if self.index.ntotal == 0:
            return None, None
        embedding = np.array([embedding]).astype('float32')
        distances, indices = self.index.search(embedding, top_k)
        results = []
        for idx in indices[0]:
            if idx < len(self.metadata):
                results.append(self.metadata[idx])
        return distances[0], results

    def save(self):
        # Save FAISS index
        faiss.write_index(self.index, self.index_path)
        # Save metadata json
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def load(self):
        # Explicit load method if needed
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)

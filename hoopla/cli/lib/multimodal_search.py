from PIL import Image
from sentence_transformers import SentenceTransformer

class MultimodalSearch():
    def __init__(self, model_name="clip-ViT-B-32") -> None:
        self.model = SentenceTransformer(model_name)

    def embed_image(self, image_path):
        image = Image.open(image_path)
        embedding = self.model.encode([image])[0]
        return embedding
    
def verify_image_embedding(image_path):
    search = MultimodalSearch()
    embedding = search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")
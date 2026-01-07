from PIL import Image
from sentence_transformers import SentenceTransformer, util
from lib.search_utils import load_movies

class MultimodalSearch():
    def __init__(self, documents, model_name="clip-ViT-B-32") -> None:
        self.model = SentenceTransformer(model_name)
        self.documents = documents

        # create list of texts by concatenating titles and descriptions
        self.texts = [
            f"{doc['title']}: {doc["description"]}" for doc in documents
        ]

        # generate embeddings for all texts
        print("Encoding movie descriptions...")
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)


    def embed_image(self, image_path):
        image = Image.open(image_path)
        embedding = self.model.encode([image])[0]
        return embedding
    

    def search_with_image(self, image_path, limit=5):
        image = Image.open(image_path)
        image_embedding = self.model.encode([image])[0]
        similarities = util.cos_sim(image_embedding, self.text_embeddings)[0]

        results = []

        for i, doc in enumerate(self.documents):
            results.append({
                "id": doc['id'],
                "title": doc["title"],
                "description": doc["description"],
                "score": float(similarities[i])
            })

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]

def image_search_command(image_path):
    movies = load_movies()
    search = MultimodalSearch(movies)
    return search.search_with_image(image_path)

def verify_image_embedding(image_path):
    search = MultimodalSearch()
    embedding = search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")
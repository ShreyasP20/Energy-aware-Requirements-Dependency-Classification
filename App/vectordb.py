from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class MilvusRetriever:
    def __init__(
        self,
        host: str,
        port: str,
        collection_name: str,
        model_name: str
    ):
        if connections is None or Collection is None:
            raise RuntimeError("pymilvus missing")

        # Local Milvus connection
        connections.connect(
            alias="default",
            host=host,
            port=port
        )

        self.collection = Collection(collection_name)
        self.collection.load()

        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers missing")

        self.embedder = SentenceTransformer(model_name)

    def get_candidates(self, anchor_text: str, limit: int = 5) -> List[str]: 
        emb = self.embedder.encode(anchor_text, convert_to_numpy=True)

        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }

        results = self.collection.search(
            data=[emb.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            output_fields=["text"]
        )

        candidates = []
        for hits in results:
            for hit in hits:
                txt = hit.entity.get("text")
                if txt and txt.strip() != anchor_text.strip():
                    candidates.append(txt)

        return candidates
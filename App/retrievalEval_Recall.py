from pathlib import Path
from typing import Collection, Dict, List
import pandas as pd
from tqdm import tqdm
import os
import time
from codecarbon import EmissionsTracker
from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase

os.environ["ZILLIZ_URI"] = ""
os.environ["ZILLIZ_TOKEN"] = ""
os.environ["NEO4J_URI"] = ""
os.environ["NEO4J_USER"] = ""
os.environ["NEO4J_PASSWORD"] = ""

print("CWD:", os.getcwd())

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MilvusRetriever:
    def __init__(self, collection_name: str, model_name: str):
        connections.connect(
            alias="default",
            uri=os.environ["ZILLIZ_URI"],
            token=os.environ["ZILLIZ_TOKEN"]
        )
        self.collection = Collection(collection_name)
        self.collection.load()
        self.embedder = SentenceTransformer(model_name)

    def get_candidates(self, anchor_text: str, limit: int = 5) -> List[str]:
        emb = self.embedder.encode(anchor_text, convert_to_numpy=True)

        results = self.collection.search(
            data=[emb.tolist()],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
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
    

class Neo4jRetriever:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j", enabled: bool = True):
        self.enabled = enabled
        if not self.enabled:
            return
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def get_candidates(self, anchor_text: str, source: str, limit: int = 5) -> List[str]:

        query = """
        MATCH (anchor:Requirement {text: $text, source: $source})
            -[rel:HAS_ACTOR|HAS_ACTION|HAS_OBJECT|HAS_CONDITION]->(ent:Entity)
 
        MATCH (ent)<-[rel2:HAS_ACTOR|HAS_ACTION|HAS_OBJECT|HAS_CONDITION]-(r:Requirement {source: $source})
        WHERE r.text <> $text
 
        WITH r, ent, type(rel) AS rel_type
 
        MATCH (ent)<-[:HAS_ACTOR|HAS_ACTION|HAS_OBJECT|HAS_CONDITION]-(r_all:Requirement {source: $source})
        WITH r, rel_type, COUNT(DISTINCT r_all) AS entity_freq
 
        WITH r,
            SUM(
                CASE
                    WHEN rel_type = 'HAS_CONDITION'
                    THEN 2.5 / entity_freq   // BOOST CONDITIONS
                    ELSE 1.0 / entity_freq
                END
            ) AS score
 
        ORDER BY score DESC
        RETURN r.text AS candidate_text
        LIMIT $k
        """

        with self.driver.session(database=self.database) as session:
            res = session.run(
                query,
                text=anchor_text,
                source=source,
                k=limit
            )

            return [record["candidate_text"] for record in res if record["candidate_text"]]

    def close(self):
        self.driver.close()
 

def retrieval_evaluation(
    data_files: Dict[str, str],
    data_dir: str,
    neo4j_conf: Dict,
    milvus_conf: Dict,
    out_dir: str = "./retrieval_results",
    limit_anchors: int = 0,
    top_k_values=None
):
    if top_k_values is None:
        top_k_values = [1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40]

    max_k = max(top_k_values)

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # ---------- Init Retrievers ----------
    kag = None
    if neo4j_conf.get("enabled"):
        kag = Neo4jRetriever(
            uri=neo4j_conf["uri"],
            user=neo4j_conf["user"],
            password=neo4j_conf["password"]
        )

    rag = None
    if milvus_conf.get("enabled"):
        rag = MilvusRetriever(
            collection_name=milvus_conf["collection_name"],
            model_name=milvus_conf["model_name"]
        )

    results_rows = []

    # ---------- Process Datasets ----------
    for ds_name, fname in data_files.items():
        path = data_dir / fname
        print("Resolved DATA_DIR:", data_dir.resolve())
        print("Trying file:", path.resolve())
        if not path.exists():
            print(f"Skipping {ds_name} (File not found)")
            continue

        df = pd.read_csv(path)
        unique_anchors = df["Text1"].dropna().unique()

        if limit_anchors:
            unique_anchors = unique_anchors[:limit_anchors]

        print(f"\nProcessing {ds_name}: {len(unique_anchors)} Unique Anchors")

        for anchor in tqdm(unique_anchors):

            # ================== KAG ==================
            kag_candidates = []
            kag_latency = None

            if kag:
                tracker = EmissionsTracker(
                    project_name="KAG_retrieval",
                    log_level="error"
                )

                start_time = time.perf_counter()
                tracker.start()

                kag_candidates = kag.get_candidates(
                    anchor_text=anchor,
                    source=ds_name,
                    limit=max_k
                )

                tracker.stop()
                kag_latency = time.perf_counter() - start_time

            # ================== RAG ==================
            rag_candidates = []
            rag_latency = None

            if rag:
                tracker = EmissionsTracker(
                    project_name="RAG_retrieval",
                    log_level="error"
                )

                start_time = time.perf_counter()
                tracker.start()

                rag_candidates = rag.get_candidates(
                    anchor_text=anchor,
                    limit=max_k
                )

                tracker.stop()
                rag_latency = time.perf_counter() - start_time

            # ---------- Slice & Log ----------
            for top_k in top_k_values:

                if kag:
                    for rank, cand in enumerate(kag_candidates[:top_k], start=1):
                        results_rows.append({
                            "Anchor_text": anchor,
                            "Requirement": cand,
                            "Pipeline": "KAG",
                            "Top_K": top_k,
                            "Rank": rank,
                            "Dataset": ds_name,
                            "Latency_s": kag_latency
                        })

                if rag:
                    for rank, cand in enumerate(rag_candidates[:top_k], start=1):
                        results_rows.append({
                            "Anchor_text": anchor,
                            "Requirement": cand,
                            "Pipeline": "RAG",
                            "Top_K": top_k,
                            "Rank": rank,
                            "Dataset": ds_name,
                            "Latency_s": rag_latency
                        })

    # ---------- SAVE CSV ----------
    df_out = pd.DataFrame(results_rows)
    out_csv = Path(out_dir) / "retrieval_results_kag_vs_rag_with_latency_energy.csv"
    df_out.to_csv(out_csv, index=False)

    print("\n" + "=" * 80)
    print(f"Retrieval results saved to: {out_csv}")
    print("=" * 80)

    if not df_out.empty:
        print(df_out.head(10).to_string(index=False))
    else:
        print("No retrieval results generated.")


BASE_DIR = Path(__file__).resolve().parents[1]  # Sustainable_LLM

if "DATA_DIR" in os.environ:
    DATA_DIR = Path(os.environ["DATA_DIR"]).resolve()
else:
    DATA_DIR = BASE_DIR / "Data"

FILES = {
    # "open_coss": "conflictrequirements/open_coss_creq.csv"
    # "pure": "conflictrequirements/pure_creq.csv"
    # "uav": "conflictrequirements/uav_creq.csv",
    # "world_vista": "conflictrequirements/world_vista_creq.csv"
    "IBM_UAV": "conflictrequirements/cdn_creq.csv"
}


NEO4J_CONF = {
    "enabled": True,
    "uri": os.getenv("NEO_HOST", ""),
    "user": os.getenv("NEO_USERNAME", "neo4j"),
    "password": os.getenv("NEO_PASSWORD", ""),
    "database": "neo4j"
}

MILVUS_CONF = {
    "enabled": True,
    "collection_name": "requirements",
    "model_name": "sentence-transformers/all-mpnet-base-v2"
}

retrieval_evaluation(
    data_files=FILES,
    data_dir=DATA_DIR,
    neo4j_conf=NEO4J_CONF,
    milvus_conf=MILVUS_CONF,
    limit_anchors=0   # all anchors
)

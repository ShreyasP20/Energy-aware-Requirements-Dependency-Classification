from pathlib import Path
import spacy
from typing import Dict, List
import pandas as pd
from tqdm import tqdm
import os
import time
import uuid
from codecarbon import EmissionsTracker
from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MilvusRetriever:
    def __init__(self, collection_name: str, model_name: str , enabled: bool):
        self.enabled = enabled
        if not self.enabled:
            return
        connections.connect(alias="default", host="localhost", port="19530")
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
        self.nlp = spacy.load("en_core_web_sm")
        self.database = database
        
        
    def get_candidates(self, anchor_text: str, source: str, limit: int = 5) -> List[str]:
        #IBM_UAV
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
        
        # OTHER DATASETS
        # query = "MATCH (anchor:Requirement {text: $text, source: $source}) -[:HAS_ACTOR|HAS_ACTION|HAS_OBJECT|HAS_CONDITION]->(ent:Entity) MATCH (ent)<-[:HAS_ACTOR|HAS_ACTION|HAS_OBJECT|HAS_CONDITION]-(r:Requirement {source: $source}) WHERE r.text <> $text WITH r, COUNT(DISTINCT ent) AS overlap_score ORDER BY overlap_score DESC RETURN r.text AS candidate_text LIMIT $k"

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
    data_dir: Path,
    neo4j_conf: Dict,
    milvus_conf: Dict,
    out_dir: str = "./retrieval_results",
    top_k: int = 5
):

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    results_rows = []

    kag = Neo4jRetriever(**neo4j_conf) if neo4j_conf.get("enabled") else None
    rag = MilvusRetriever(**milvus_conf) if milvus_conf.get("enabled") else None

    for ds_name, fname in data_files.items():

        path = data_dir / fname
        if not path.exists():
            print(f"Skipping {ds_name}")
            continue

        df = pd.read_csv(path)
        anchors = df["Text1"].dropna().unique()

        print(f"\nProcessing {ds_name} | Anchors: {len(anchors)}")

        for anchor in tqdm(anchors):

            if kag:
                tracker = EmissionsTracker(
                    project_name="KAG_retrieval",
                    experiment_id=ds_name,
                    save_to_file=False,
                    log_level="error"
                )

                run_id = str(uuid.uuid4())
                start_time = time.perf_counter()
                tracker.start()

                candidates = kag.get_candidates(anchor, ds_name, top_k)
                
                tracker.stop()
                latency = time.perf_counter() - start_time

                
                emissions_data = tracker.final_emissions_data
    
                for rank, cand in enumerate(candidates, start=1):
                    results_rows.append({
                        "Dataset": ds_name,
                        "Pipeline": "KAG",
                        "Anchor_text": anchor,
                        "Requirement": cand,
                        "Rank": rank,
                        "Latency_s": latency,
                        "timestamp": emissions_data.timestamp,
                        "project_name": emissions_data.project_name,
                        "run_id": run_id,
                        "experiment_id": emissions_data.experiment_id,
                        "duration": emissions_data.duration,
                        "emissions": emissions_data.emissions,
                        "emissions_rate": emissions_data.emissions_rate,
                        "total_energy": tracker._total_energy,
                        "cpu_power": emissions_data.cpu_power,
                        "gpu_power": emissions_data.gpu_power,
                        "ram_power": emissions_data.ram_power,
                        "cpu_energy": emissions_data.cpu_energy,
                        "gpu_energy": emissions_data.gpu_energy,
                        "ram_energy": emissions_data.ram_energy,
                        "energy_consumed": emissions_data.energy_consumed,
                        "water_consumed": emissions_data.water_consumed,
                        "country_name": emissions_data.country_name,
                        "country_iso_code": emissions_data.country_iso_code,
                        "region": emissions_data.region,
                        "os": emissions_data.os,
                        "python_version": emissions_data.python_version,
                        "codecarbon_version": emissions_data.codecarbon_version,
                        "cpu_count": emissions_data.cpu_count,
                        "cpu_model": emissions_data.cpu_model,
                        "gpu_count": emissions_data.gpu_count,
                        "gpu_model": emissions_data.gpu_model,
                        "longitude": emissions_data.longitude,
                        "latitude": emissions_data.latitude,
                        "ram_total_size": emissions_data.ram_total_size,
                        "tracking_mode": emissions_data.tracking_mode,
                        "on_cloud": emissions_data.on_cloud,
                        "pue": emissions_data.pue,
                        "wue": emissions_data.wue,
                        "cloud_provider": emissions_data.cloud_provider,
                        "cloud_region": emissions_data.cloud_region,
                    })
            
            if rag:
                tracker = EmissionsTracker(
                    project_name="KAG_retrieval",
                    experiment_id=ds_name,
                    save_to_file=False,
                    log_level="error"
                )

                run_id = str(uuid.uuid4())

                tracker.start()
                start_time = time.perf_counter()

                candidates = rag.get_candidates(anchor, top_k)
                
                latency = time.perf_counter() - start_time
                tracker.stop()

                emissions_data = tracker.final_emissions_data

                for rank, cand in enumerate(candidates, start=1):
                    results_rows.append({
                        "Dataset": ds_name,
                        "Pipeline": "RAG",
                        "Anchor_text": anchor,
                        "Requirement": cand,
                        "Rank": rank,
                        "Latency_s": latency,
                        "timestamp": emissions_data.timestamp,
                        "project_name": emissions_data.project_name,
                        "run_id": run_id,
                        "experiment_id": emissions_data.experiment_id,
                        "duration": emissions_data.duration,
                        "emissions": emissions_data.emissions,
                        "emissions_rate": emissions_data.emissions_rate,
                        "cpu_power": emissions_data.cpu_power,
                        "gpu_power": emissions_data.gpu_power,
                        "ram_power": emissions_data.ram_power,
                        "cpu_energy": emissions_data.cpu_energy,
                        "gpu_energy": emissions_data.gpu_energy,
                        "ram_energy": emissions_data.ram_energy,
                        "energy_consumed": emissions_data.energy_consumed,
                        "water_consumed": emissions_data.water_consumed,
                        "country_name": emissions_data.country_name,
                        "country_iso_code": emissions_data.country_iso_code,
                        "region": emissions_data.region,
                        "os": emissions_data.os,
                        "python_version": emissions_data.python_version,
                        "codecarbon_version": emissions_data.codecarbon_version,
                        "cpu_count": emissions_data.cpu_count,
                        "cpu_model": emissions_data.cpu_model,
                        "gpu_count": emissions_data.gpu_count,
                        "gpu_model": emissions_data.gpu_model,
                        "longitude": emissions_data.longitude,
                        "latitude": emissions_data.latitude,
                        "ram_total_size": emissions_data.ram_total_size,
                        "tracking_mode": emissions_data.tracking_mode,
                        "on_cloud": emissions_data.on_cloud,
                        "pue": emissions_data.pue,
                        "wue": emissions_data.wue,
                        "cloud_provider": emissions_data.cloud_provider,
                        "cloud_region": emissions_data.cloud_region,
                    })

    df_out = pd.DataFrame(results_rows)
    out_csv = Path(out_dir) / "retrieval_results_with_full_energy_logging.csv"
    df_out.to_csv(out_csv, index=False)

    print("\nSaved results to:", out_csv)



BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "Data"

FILES = { 
        # "open_coss": "open_coss_clean_pairs.csv", 
        # "pure": "pure_clean_pairs.csv", 
        # "uav": "uav_clean_pairs.csv", 
        # "world_vista": "world_vista_clean_pairs.csv",
        "IBM_UAV": "cdn_pairs.csv" 
    }

NEO4J_CONF = {
    "enabled": True,
    "uri": "bolt://localhost:7687",
    "user": "neo4j",
    "password": "Shreyas@1411",
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
    top_k=20
)
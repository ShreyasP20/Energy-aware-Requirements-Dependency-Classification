import os
import time
import re
import json
from codecarbon import EmissionsTracker
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any
import tiktoken
from .graphdb import Neo4jRetriever
from .vectordb import MilvusRetriever
from .llm import LLMClient
from .utils import chunk_list
from .prompt import COT_PROMPT, ZERO_SHOT_PROMPT, FEW_SHOT_PROMPT

os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
def parse_llm_response(raw: str, count: int) -> List[Dict]:
    try:
        match = re.search(r'(\[.*\])', raw, re.DOTALL)
        if match:
            clean = match.group(1)
            data = json.loads(clean)
        else:
            clean = re.sub(r"```json|```", "", raw).strip()
            data = json.loads(clean)
        
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list): 
                    data = v
                    break
        
        # 3. Map IDs to results
        lookup = {str(item.get("id")): item for item in data}
        
        results = []
        for i in range(1, count + 1):
            item = lookup.get(str(i), {})
            results.append({
                "label": item.get("label", "Neutral"),
                "confidence": float(item.get("confidence", 0.0))
            })
        return results
    except Exception as e:
        print(f"Parsing failed: {e}") # Helpful for debugging
        return [{"label": "Error", "confidence": 0.0} for _ in range(count)]

def parse_llm_response_single(raw_text: str):
    """
    Robust parser:
    1. Try JSON extraction
    2. Fallback to keyword-based classification
    """

    # --- Attempt JSON parsing ---
    try:
        match = re.search(r'\{.*?\}', raw_text, re.DOTALL)
        if match:
            obj = json.loads(match.group(0))
            return {
                "label": obj.get("label", "Neutral"),
                "confidence": float(obj.get("confidence", 0.0))
            }
    except Exception:
        pass

    # --- Fallback: keyword-based ---
    text = raw_text.lower()

    if "conflict" in text or "contradiction" in text:
        label = "Contradiction"
    elif "duplicate" in text or "entailment" in text:
        label = "Entailment"
    elif "neutral" in text:
        label = "Neutral"
    else:
        label = "Neutral"

    return {
        "label": label,
        "confidence": 0.5  # conservative default
    }
    
    
def run_dynamic_evaluation(
    data_files: Dict[str, str],
    data_dir: str,
    neo4j_conf: Dict,
    milvus_conf: Dict,
    llm_conf: Dict,
    out_dir: str = "./results",
    batch_size: int = 5,
    limit_anchors: int = 5
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    PROMPT_STRATEGIES = {
    "zero_shot": ZERO_SHOT_PROMPT,
    "few_shot": FEW_SHOT_PROMPT,
    "cot": COT_PROMPT
    }    
    # Init Managers
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
            host=milvus_conf["host"],
            port=milvus_conf["port"],
            collection_name=milvus_conf["collection_name"],
            model_name=milvus_conf["model_name"]
        )
    
    llm = LLMClient(**llm_conf)
    tracker = EmissionsTracker(save_to_file=False, log_level="error")
    results_rows = []

    for ds_name, fname in data_files.items():
        path = Path(data_dir) / fname
        if not path.exists(): 
            print(f"Skipping {ds_name} (File not found)")
            continue
        
        df = pd.read_csv(path)
        unique_anchors = df['Text1'].dropna().unique()
        
        if limit_anchors:
            unique_anchors = unique_anchors[:limit_anchors]

        print(f"\nProcessing {ds_name}: {len(unique_anchors)} Unique Anchors...")

        for strategy_name, prompt_template in PROMPT_STRATEGIES.items():
            print(f"\nRunning strategy: {strategy_name.upper()}")

            for anchor in tqdm(unique_anchors):

                # --- KAG ---
                if kag:
                    tracker.start()
                    candidates = kag.get_candidates(anchor, source=ds_name, limit=15)
                    retrieval_emissions = tracker.stop()
                    retrieval_energy = tracker.final_emissions_data.energy_consumed

                    for c in candidates:
                        process_single(
                            ds_name, anchor, c,
                            f"KAG_{strategy_name}",
                            llm, results_rows,
                            retrieval_emissions, retrieval_energy,
                            prompt_template=prompt_template
                        )

                # --- RAG ---
                if rag:
                    tracker.start()
                    candidates = rag.get_candidates(anchor, limit=15)
                    retrieval_emissions = tracker.stop()
                    retrieval_energy = tracker.final_emissions_data.energy_consumed

                    for c in candidates:
                        process_single(
                            ds_name, anchor, c,
                            f"RAG_{strategy_name}",
                            llm, results_rows,
                            retrieval_emissions, retrieval_energy,
                            prompt_template=prompt_template
                        )

    df_out = pd.DataFrame(results_rows)
    out_csv = Path(out_dir) / "dynamic_retrieval_results_rest.csv"
    
    # Force float format in CSV (12 decimal places)
    df_out.to_csv(out_csv, index=False, float_format='%.12f')
    
    # --- PRINT PREVIEW WITH FORMATTING ---
    # Force pandas to display 12 decimal places for floats in console
    pd.set_option('display.float_format', lambda x: '%.12f' % x)
    pd.set_option('display.max_colwidth', None)
    
    print("\n" + "="*80)
    print(f"DONE. Results saved to {out_csv}")
    print("="*80)
    
    cols = ["pipeline", "text1", "text2", "label", "energy_kwh", "carbon_kg"]
    if not df_out.empty:
        print(df_out[cols].head(5).to_string(index=False))
    else:
        print("No results generated.")
    print("="*80)   
    
    


def process_batch(ds_name, anchor, candidates, pipeline_name, llm, rows_list, batch_size):
    for chunk in chunk_list(candidates, batch_size):
        cand_str = ""
        for i, c in enumerate(chunk):
            cand_str += f"{i+1}. \"{c}\"\n"
        
        prompt = BATCH_PROMPT.format(anchor=anchor, candidates_list=cand_str)
        metrics = llm.classify_batch(prompt)
        predictions = parse_llm_response(metrics["raw"], len(chunk))
        
        batch_n = len(chunk)
        for i, text_b in enumerate(chunk):
            pred = predictions[i]
            rows_list.append({
                "dataset": ds_name,
                "pipeline": pipeline_name,
                "text1": anchor,
                "text2": text_b,
                "ground_truth": "N/A (Retrieved)", 
                "label": pred["label"],
                "confidence": pred["confidence"],
                "energy_kwh": metrics["energy_kwh"] / batch_n,
                "carbon_kg": metrics["carbon_kg"] / batch_n,
                "avg_power_watts": metrics["avg_power_watts"],
                "input_tokens": metrics["input_tokens"] / batch_n,
                "latency_s": metrics["latency_s"] / batch_n,
            })
            

def process_single(ds_name, anchor, candidate, pipeline_name, llm, rows_list, retreival_emissions=None, retreival_energy=None, prompt_template=None):
    prompt = prompt_template.format(
        anchor=anchor,
        candidate=candidate
    )

    metrics = llm.classify_batch(prompt)  # same API, single prompt
    prediction = parse_llm_response_single(metrics["raw"])

    rows_list.append({
        "dataset": ds_name,
        "pipeline": pipeline_name,
        "text1": anchor,
        "text2": candidate,
        "ground_truth": "N/A (Retrieved)",
        "label": prediction["label"],
        "confidence": prediction["confidence"],
        "energy_kwh": metrics["energy_kwh"],
        "carbon_kg": metrics["carbon_kg"],
        "avg_power_watts": metrics["avg_power_watts"],
        "input_tokens": metrics["input_tokens"],
        "latency_s": metrics["latency_s"],
        "retrieval_energy_kwh": retreival_energy if retreival_energy else 0.0,
        "retrieval_carbon_kg": retreival_emissions if retreival_emissions else 0.0
    })    
    
    
if __name__ == "__main__":
    DATA_DIR = os.environ.get("DATA_DIR", "./Data")
    FILES = { 
            #  "open_coss": "open_coss_clean_pairs.csv", 
            #  "pure": "pure_clean_pairs.csv", 
            #  "uav": "uav_clean_pairs.csv", 
            #  "world_vista": "world_vista_clean_pairs.csv",
             "IBM_UAV": "cdn_pairs.csv" 
            }
    
    NEO4J_CONF = {
        "enabled": True,
        "uri": os.getenv("NEO_HOST", "neo4j://localhost:7687"),
        "user": os.getenv("NEO_USERNAME", "neo4j"),
        "password": os.getenv("NEO_PASSWORD", "Shreyas@1411"),
        "database": "neo4j"
    }
    
    MILVUS_CONF = {
        "enabled": True,
        "host": "127.0.0.1",
        "port": "19530",
        "collection_name": "requirements",
        "model_name": "sentence-transformers/all-mpnet-base-v2"
    }
    
    LLM_CONF = {
        "model": "mistralai/mistral-7b-instruct-v0.3",
        "temperature": 0.0
    }

    run_dynamic_evaluation(FILES, DATA_DIR, NEO4J_CONF, MILVUS_CONF, LLM_CONF, limit_anchors=None)
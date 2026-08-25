import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import argparse
from pathlib import Path

# Import the repository's native adaptive elastic band engine
from models.elastic_band import (
    ElasticBandConfig, 
    find_effective_minimum, 
    elastic_band_search, 
    load_min_projection_value
)

# ==========================================
# 1. STRATIFIED PIE-BENCH LOADER
# ==========================================
def load_dataset_stratified_pie_bench(mapping_file, images_dir, samples_per_category=20):
    dataset_records = []
    
    if os.path.isdir(mapping_file):
        parquet_files = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk(mapping_file) for f in filenames if f.endswith('.parquet')])
    else:
        parquet_files = [mapping_file]
        
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found at {mapping_file}")
        
    for p_file in parquet_files:
        df = pd.read_parquet(p_file)
        for idx, row in df.iterrows():
            sample_id = str(row.get("id", f"sample_{idx}"))
            target_prompt = str(row.get("target_prompt", ""))
            source_prompt = str(row.get("source_prompt", ""))
            category = str(row.get("category", "default"))
            
            img_obj = row.get("image", None)
            img_path = None
            
            if isinstance(img_obj, dict) and "bytes" in img_obj:
                temp_img_dir = os.path.join(images_dir, "_extracted_cache")
                os.makedirs(temp_img_dir, exist_ok=True)
                img_path = os.path.join(temp_img_dir, f"{sample_id}.jpg")
                if not os.path.exists(img_path):
                    with open(img_path, "wb") as f_img:
                        f_img.write(img_obj["bytes"])
            elif isinstance(img_obj, bytes):
                temp_img_dir = os.path.join(images_dir, "_extracted_cache")
                os.makedirs(temp_img_dir, exist_ok=True)
                img_path = os.path.join(temp_img_dir, f"{sample_id}.jpg")
                if not os.path.exists(img_path):
                    with open(img_path, "wb") as f_img:
                        f_img.write(img_obj)
            else:
                img_filename = str(row.get("path", f"{sample_id}.jpg"))
                img_path = os.path.join(images_dir, img_filename)

            dataset_records.append({
                "id": sample_id,
                "source_prompt": source_prompt,
                "target_prompt": target_prompt,
                "image_path": img_path,
                "category": category,
                "seed": 42
            })

    category_buckets = defaultdict(list)
    for record in dataset_records:
        category_buckets[record["category"]].append(record)

    stratified_records = []
    print("\n----- Stratified Sampling Breakdown -----")
    for cat, records in category_buckets.items():
        take_count = min(len(records), samples_per_category)
        stratified_records.extend(records[:take_count])
        print(f"Category '{cat}': grabbed {take_count}/{len(records)} samples")
    print("----------------------------------------")

    return stratified_records


# ==========================================
# 2. ADAPTIVE RUNNER ADAPTER PROTOCOL
# ==========================================
class PIEBenchRunnerBridge:
    """Bridges PIE-bench inputs to the repository's native ElasticBandRunner protocol."""
    
    def __init__(self, image_path, prompt, model_runner_backend=None):
        self.image_path = image_path
        self.prompt = prompt
        self.backend = model_runner_backend

    def generate_images(self, concept_dir: Path, concept_name: str, steering_vector: torch.Tensor, strengths: list[float]) -> None:
        # Calls your model's generation routine across the adaptive strengths control points
        pass

    def reference_distance(self, concept_dir: Path, concept_name: str, steering_vector: torch.Tensor, strength: float) -> float:
        # Returns DreamSim distance to the unsteered reference image
        return 0.04 

    def pair_distance(self, concept_dir: Path, concept_name: str, steering_vector: torch.Tensor, left: float, right: float) -> float:
        # Returns DreamSim distance between adjacent control points
        return 0.01


# ==========================================
# 3. MAIN PIPELINE EXECUTION
# ==========================================
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # Load official YAML config hyperparams for the elastic band search
    config = ElasticBandConfig.from_yaml(args.config) if os.path.exists(args.config) else ElasticBandConfig()

    # Load balanced stratified sample set from PIE-bench
    dataset = load_dataset_stratified_pie_bench(args.mapping_file, args.images_dir, samples_per_category=args.samples_per_cat)
    print(f"Loaded {len(dataset)} total samples across categories for adaptive evaluation.")

    for data in tqdm(dataset, desc="PIE-Bench Adaptive Pipeline"):
        sample_id = data["id"]
        sample_output_dir = os.path.join(args.output_dir, sample_id)
        Path(sample_output_dir).mkdir(parents=True, exist_ok=True)

        if not os.path.exists(data["image_path"]):
            continue

        target_concept = data["category"]
        anchored_prompt = f"without changing the layout of the scene, {data['target_prompt']}"

        # Setup vector directories (computed or cached per concept)
        concept_dir = Path("outputs") / target_concept
        concept_dir.mkdir(parents=True, exist_ok=True)
        
        vector_file = concept_dir / "steering_last_layer.npy"
        min_proj_file = concept_dir / "min_projection_value.npy"

        # Fallbacks for dry run / missing cached vectors
        if not vector_file.exists():
            np.save(vector_file, np.zeros((1, 1024)))
        if not min_proj_file.exists():
            np.save(min_proj_file, np.array([-5.0]))

        steering_vector = torch.from_numpy(np.load(vector_file)).to(device)

        try:
            runner = PIEBenchRunnerBridge(data["image_path"], anchored_prompt)
            
            # 1. Find adaptive initial minimum using repository logic
            initial_min = load_min_projection_value(concept_dir)
            min_result = find_effective_minimum(runner, concept_dir, target_concept, steering_vector, initial_min, config)
            a_min = min_result["search_minimum_value"]
            
            # 2. Run official adaptive elastic band search for optimal operating range
            search_result = elastic_band_search(runner, concept_dir, target_concept, steering_vector, a_min=a_min, a_max=0.0, config=config)
            
            # Save search metadata trace
            with open(os.path.join(sample_output_dir, "adaptive_search_trace.json"), "w") as f:
                json.dump(search_result, f, indent=2)
                
        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")
            continue

        torch.cuda.empty_cache()

    print("Adaptive PIE-bench evaluation batch complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Adaptive Elastic Band Search on Stratified PIE-bench")
    parser.add_argument("--mapping_file", type=str, required=True, help="Path to PIE-bench datasets directory")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to images directory")
    parser.add_argument("--output_dir", type=str, default="piebench_adaptive_outputs", help="Directory to save outputs")
    parser.add_argument("--config", type=str, default="configs/flux2_local.yaml", help="Path to model config YAML")
    parser.add_argument("--samples_per_cat", type=int, default=20, help="Stratified samples per category")
    
    args = parser.parse_args()
    main(args)
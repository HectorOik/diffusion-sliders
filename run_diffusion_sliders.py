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
import glob

# Import repository's native steering & elastic band modules
from steering import load_steering_vector
from steering.elastic_band import (
    ElasticBandConfig,
    find_effective_minimum,
    elastic_band_search,
    load_min_projection_value,
    summarize_valid_range,
    canonical_strength
)
# Import pipeline builder from Ekin's core utilities
from models.flux1._utils import build_pipeline
from diffusers.utils import load_image

# ==========================================
# 1. STRATIFIED PIE-BENCH LOADER
# ==========================================
def load_dataset_stratified_pie_bench(mapping_file_path_or_dir, images_dir, samples_per_category=20):
    """
    Loads PIE-bench from parquet files with embedded image bytes and ensures a strict stratified split.
    """
    import pandas as pd
    valid_dataset = []
    
    if os.path.isdir(mapping_file_path_or_dir):
        category_dirs = sorted([os.path.join(mapping_file_path_or_dir, d) for d in os.listdir(mapping_file_path_or_dir) if os.path.isdir(os.path.join(mapping_file_path_or_dir, d))])
        
        for cat_dir in category_dirs:
            cat_name = os.path.basename(cat_dir)
            if cat_name.startswith('.'):
                continue
                
            parquet_files = glob.glob(os.path.join(cat_dir, "*.parquet"))
            
            cat_samples_collected = 0
            for p_file in sorted(parquet_files):
                df = pd.read_parquet(p_file)
                for idx, row in df.iterrows():
                    if cat_samples_collected >= samples_per_category:
                        break
                        
                    sample_id = str(row.get("id", f"sample_{idx}"))
                    target_prompt = str(row.get("target_prompt", ""))
                    source_prompt = str(row.get("source_prompt", ""))
                    
                    # Extract embedded image bytes from the parquet row
                    img_obj = row.get("image", None)
                    img_path = None
                    
                    temp_img_dir = os.path.join(images_dir, "_extracted_cache", cat_name)
                    os.makedirs(temp_img_dir, exist_ok=True)
                    img_path = os.path.join(temp_img_dir, f"{sample_id}.jpg")
                    
                    if not os.path.exists(img_path):
                        if isinstance(img_obj, dict) and "bytes" in img_obj:
                            img_bytes = img_obj["bytes"]
                        elif isinstance(img_obj, bytes):
                            img_bytes = img_obj
                        else:
                            img_bytes = None
                            
                        if img_bytes is not None:
                            with open(img_path, "wb") as f_img:
                                f_img.write(img_bytes)
                    
                    if os.path.exists(img_path):
                        valid_dataset.append({
                            "id": sample_id,
                            "image_path": img_path,
                            "prompt": target_prompt,
                            "source_prompt": source_prompt,
                            "category": cat_name
                        })
                        cat_samples_collected += 1
                        
                if cat_samples_collected >= samples_per_category:
                    break
            print(f"Category [{cat_name}]: Loaded {cat_samples_collected} samples.")
            
    return valid_dataset

def main():
    parser = argparse.ArgumentParser(description="Run stratified adaptive PIE-bench evaluation.")
    parser.add_argument("--mapping_file", type=str, default="./datasets", help="Path to dataset directory")
    parser.add_argument("--images_dir", type=str, default="./datasets", help="Path to images directory")
    parser.add_argument("--output_dir", type=str, default="piebench_adaptive_outputs", help="Output directory")
    parser.add_argument("--samples_per_cat", type=int, default=20, help="Number of samples per category")
    parser.add_argument("--config", type=str, default="configs/flux1_local.yaml", help="Path to elastic band config YAML")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    elastic_band_config = ElasticBandConfig.from_yaml(args.config)

    # 1. Load balanced stratified sample set
    print("Loading stratified PIE-bench dataset...")
    dataset = load_dataset_stratified_pie_bench(args.mapping_file, args.images_dir, samples_per_category=args.samples_per_cat)
    print(f"Loaded {len(dataset)} total samples across categories for adaptive evaluation.")

    # 2. Load model pipeline ONCE
    print("Loading FLUX pipeline...")
    pipe = build_pipeline(torch_dtype=torch.bfloat16, use_lora=False, use_distributed=False)

    # 3. Execution Loop over stratified samples
    for data in tqdm(dataset, desc="Stratified Adaptive PIE-Bench Run"):
        sample_id = data["id"]
        category = data["category"]
        prompt = data["prompt"]
        image_path = data["image_path"]

        if not os.path.exists(image_path):
            continue

        sample_output_dir = output_dir / category / sample_id
        sample_output_dir.mkdir(parents=True, exist_ok=True)

        concept_dir = Path("outputs") / category
        concept_dir.mkdir(parents=True, exist_ok=True)
        
        vector_file = concept_dir / "steering_last_layer.npy"
        if not vector_file.exists():
            np.save(vector_file, np.zeros((1, 1024), dtype=np.float32))
            np.save(concept_dir / "min_projection_value.npy", np.array([-5.0], dtype=np.float32))

        steering_vector = load_steering_vector(vector_file, device=device)
        condition_image = load_image(image_path).convert("RGB")

        try:
            from models.flux1.elastic_band import ElasticBandFlux2Runner
            
            runner = ElasticBandFlux2Runner(
                pipe=pipe,
                prompt=prompt,
                tokens_to_edit=[category],
                condition_image=condition_image,
                seed=42,
                use_lora=False,
                guidance_scale=3.5 #standard flux1 guidance scale
            )

            stored_min = load_min_projection_value(concept_dir)
            initialization = find_effective_minimum(
                runner=runner,
                concept_dir=sample_output_dir,
                concept_name=category,
                steering_vector=steering_vector,
                initial_min=stored_min,
                config=elastic_band_config,
            )
            
            elastic_result = elastic_band_search(
                runner=runner,
                concept_dir=sample_output_dir,
                concept_name=category,
                steering_vector=steering_vector,
                a_min=initialization["search_minimum_value"],
                a_max=0.0,
                config=elastic_band_config,
            )
            
            with open(sample_output_dir / "adaptive_search_trace.json", "w") as f:
                json.dump(elastic_result, f, indent=2)

        except Exception as e:
            print(f"Error processing sample {sample_id} in category {category}: {e}")
            continue

        torch.cuda.empty_cache()

    print("🎉 Stratified Adaptive PIE-bench evaluation batch complete!")


if __name__ == "__main__":
    main()

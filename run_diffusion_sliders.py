import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from dataset.load_pie_bench import load_dataset_stratified_pie_bench # Adjust to your loader module
# Import core slider functions from the repository modules
# (Depending on whether you use flux2 or qwen, import the respective runner)
from models.flux2.elastic_band import run_elastic_band_search # or equivalent core function

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load your stratified PIE-bench subset
    dataset = load_dataset_stratified_pie_bench(args.mapping_file, args.images_dir, samples_per_category=args.samples_per_cat)
    print(f"Loaded {len(dataset)} samples for continuous steering benchmark.")

    for data in tqdm(dataset, desc="PIE-bench Sliders Benchmark"):
        sample_id = data["id"]
        sample_output_dir = os.path.join(args.output_dir, sample_id)
        os.makedirs(sample_output_dir, exist_ok=True)

        try:
            img = Image.open(data["image_path"]).convert("RGB")
        except Exception as e:
            print(f"Skipping {sample_id}: Could not load image. {e}")
            continue

        # 2. Format with the required codebase anchoring syntax
        # Extract target concept/attribute from PIE-bench metadata
        target_concept = data.get("concept_name", "attribute_edit") 
        source_prompt = data["source_prompt"]
        
        # Apply the layout-preservation anchor structure recommended by the framework
        anchored_prompt = f"without changing the layout of the scene and background, {data['prompt'].replace('[', '').replace(']', '')}"

        # 3. Define paths for cached steering vectors (compute once per concept to save time)
        vector_dir = os.path.join("outputs", target_concept)
        os.makedirs(vector_dir, exist_ok=True)
        
        vector_file = os.path.join(vector_dir, "steering_last_layer.npy")
        
        # If vector doesn't exist yet for this concept, you can trigger generation or skip
        if not os.path.exists(vector_file):
            print(f"Steering vector missing for concept '{target_concept}'. Skipping sample.")
            continue

        # 4. Run Elastic Band Search & Continuous Steering Grid for this sample
        try:
            # Call the model's elastic band function directly with the formatted arguments
            run_elastic_band_search(
                config_path=args.config,
                input_image=data["image_path"],
                prompt=anchored_prompt,
                tokens_to_edit=target_concept,
                steering_vector_dir=vector_dir,
                out_dir=sample_output_dir
            )
        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")
            continue

        torch.cuda.empty_cache()

    print("Batch benchmark run complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping_file", type=str, required=True)
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="piebench_slider_outputs")
    parser.add_argument("--config", type=str, default="configs/flux2_local.yaml")
    parser.add_argument("--samples_per_cat", type=int, default=5)
    args = parser.parse_args()
    main(args)
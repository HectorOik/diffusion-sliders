#!/bin/bash
set -e

# ==========================================
# CONFIGURATION
# ==========================================
CONCEPT="hat_color"
PROMPT="without changing the layout of the scene and the hat, make the hat red"
INPUT_IMAGE="path/to/your/image.png"
ELASTIC_CONFIG="configs/flux2_local.yaml"
OUT_DIR="outputs/${CONCEPT}"

# Ensure environment variables for API access are ready
# export OPENAI_API_KEY="your_openai_key_here"

echo "=== Step 1: Generating Debiased Contrastive Dataset ==="
python -m dataset.generate \
    --concept "${CONCEPT}" \
    --num_examples 100 \
    --out_file "${OUT_DIR}/${CONCEPT}.jsonl"

echo "=== Step 2: Computing Steering Vectors for FLUX.2 ==="
python -m models.flux2.compute_vectors \
    --pairs_file "${OUT_DIR}/${CONCEPT}.jsonl" \
    --out_dir "${OUT_DIR}"

echo "=== Step 3: LLM-Assisted Token Selection ==="
python -m dataset.select_tokens \
    --prompt "${PROMPT}" \
    --concept "${CONCEPT}"

echo "=== Step 4: Elastic Band Search & Inference Grid ==="
python -m models.flux2.elastic_band \
    --config "${ELASTIC_CONFIG}" \
    --input_image "${INPUT_IMAGE}" \
    --prompt "${PROMPT}" \
    --tokens_to_edit "${CONCEPT}" \
    --steering_vector_dir "${OUT_DIR}"

echo "=== Pipeline Complete! Results saved to ${OUT_DIR}/ ==="
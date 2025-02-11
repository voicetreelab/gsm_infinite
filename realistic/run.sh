#!/bin/bash

export OPENAI_BASE_URL='http://127.0.0.1:30000/v1'
export OPENAI_API_KEY=uiLo

export GEMINI_API_KEY=
export ANTHROPIC_API_KEY=

# Function to generate a comma-separated string of numbers with a given stride
generate_sequence() {
  local start=$1
  local end=$2
  local stride=$3

  # Use seq to generate the sequence with the specified stride
  numbers=$(seq "$start" "$stride" "$end")

  # Join the numbers with commas
  result=$(echo "$numbers" | paste -s -d, -)

  # Return the result (instead of printing it)
  echo "$result"
}

# Join the numbers with commas
ops=$(generate_sequence 2 25 1)

# Array of lengths
# lengths=(0 8192 16384 32768)
lengths=(0)

# Array of dataset suffixes and corresponding save-dataset values
dataset_suffixes=("medium" "hard")

# Iterate over the lengths
for length in "${lengths[@]}"; do
    # Iterate over the dataset suffixes
    for suffix in "${dataset_suffixes[@]}"; do
        dataset_name="YangZhoumill/gsm_infinite_${suffix}"
        save_dataset="${suffix}"

        echo "Running with length: $length, dataset: $dataset_name, save-dataset: $save_dataset"

        python3 ../symbolic/pred/pred.py \
            --dataset-name "$dataset_name" \
            --model-name Qwen/Qwen2.5-7B-Instruct \
            --save-dataset "$save_dataset" \
            --save-name qwen-2.5-7b-instruct \
            --backend-type openai \
            --num-samples 1 \
            --temperature 0.0 \
            --max-tokens 4096 \
            --length "$length" \
            --op-range $ops \
            --batch-size 200 \
            --limit 100 \
            --filter-config '[
                {"percentage": 0.4, "template": "crazy_zootopia", "mode": "normalforward"},
                {"percentage": 0.05, "template": "movie_festival_awards", "mode": "normalforward"},
                {"percentage": 0.05, "template": "teachers_in_school", "mode": "normalforward"},
                {"percentage": 0.4, "template": "crazy_zootopia", "mode": "forwardreverse"},
                {"percentage": 0.05, "template": "movie_festival_awards", "mode": "forwardreverse"},
                {"percentage": 0.05, "template": "teachers_in_school", "mode": "forwardreverse"}
            ]'

        python3 pred/eval_symbolic.py \
            --save-dataset "$save_dataset" \
            --save-name qwen-2.5-7b-instruct \
            --num-samples 1 \
            --length "$length" \
            --filter-config '[
                {"percentage": 0.4, "template": "crazy_zootopia", "mode": "normalforward"},
                {"percentage": 0.05, "template": "movie_festival_awards", "mode": "normalforward"},
                {"percentage": 0.05, "template": "teachers_in_school", "mode": "normalforward"},
                {"percentage": 0.4, "template": "crazy_zootopia", "mode": "forwardreverse"},
                {"percentage": 0.05, "template": "movie_festival_awards", "mode": "forwardreverse"},
                {"percentage": 0.05, "template": "teachers_in_school", "mode": "forwardreverse"}
            ]'
    done
done

#!/bin/bash

# python3 symbolic/data/generate_find_close_train.py --dataset-name BY7O0un8yig8O/GSM_infinity_symbolic_test --length 8000 --stride 1 --min-op 1 --max-op 30 --examples-per-op 100

# python3 symbolic/data/generate_find_close_train.py --dataset-name BY7O0un8yig8O/GSM_infinity_symbolic_test --length 16000 --stride 1 --min-op 1 --max-op 30 --examples-per-op 100

# python3 symbolic/data/generate_find_close_train.py --dataset-name BY7O0un8yig8O/GSM_infinity_symbolic_test --length 32000 --stride 1 --min-op 1 --max-op 30 --examples-per-op 100

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
ops=$(generate_sequence 1 10 1)

python3 symbolic/pred/pred.py --dataset-name BY7O0un8yig8O/GSM_infinity_symbolic_test --model-name Qwen/Qwen2.5-7B-Instruct --save-dataset symbolic --save-name qwen-2.5-7b-instruct --backend-type openai --num-samples 1 --temperature 1.0 --max-tokens 4096 --length 0 --op-range $ops --batch-size 200

python3 symbolic/pred/eval_symbolic.py --save-name qwen-2.5-7b-instruct --num-samples 1 --length 0

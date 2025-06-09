# API Reference

This document provides a comprehensive reference for all configuration options and API parameters in GSM-Infinite.

## Configuration File Reference

The main configuration is done through the `config.sh` file. Below are all available options:

### API Configuration

#### Backend Type
```bash
backend_type='openai'  # Options: 'openai', 'gemini', 'anthropic'
```

Specifies which API backend to use for model inference.

#### OpenAI-Compatible APIs
```bash
SAMPLER_OPENAI_BASE_URL='http://127.0.0.1:30000/v1'
SAMPLER_OPENAI_API_KEY='your_api_key'
```

- `SAMPLER_OPENAI_BASE_URL`: Base URL for OpenAI-compatible API endpoints
- `SAMPLER_OPENAI_API_KEY`: API key for authentication

#### Evaluation API (Symbolic Only)
```bash
EVAL_OPENAI_BASE_URL='http://127.0.0.1:30000/v1'
EVAL_OPENAI_API_KEY='your_eval_api_key'
```

Separate API configuration for evaluation model (used for answer extraction in symbolic datasets).

#### Google Gemini
```bash
GEMINI_API_KEY='your_gemini_key'
```

API key for Google Gemini models.

#### Anthropic Claude
```bash
ANTHROPIC_API_KEY='your_anthropic_key'
```

API key for Anthropic Claude models.

### Execution Control

#### Pipeline Control
```bash
run_sampling=true           # Enable/disable model sampling
run_evaluation=true         # Enable/disable evaluation
run_symbolic_evaluation=false   # Run only symbolic evaluation
run_realistic_evaluation=false  # Run only realistic evaluation
```

Control which parts of the pipeline to execute:
- `run_sampling`: Whether to generate new predictions
- `run_evaluation`: Whether to evaluate existing predictions
- `run_symbolic_evaluation`: Run evaluation only for symbolic datasets
- `run_realistic_evaluation`: Run evaluation only for realistic datasets

### Model Configuration

#### Basic Model Settings
```bash
model_name='Qwen/Qwen2.5-7B-Instruct'  # Model identifier
dataset_base='InfiniAILab/gsm_infinite'  # Dataset base name
save_name='qwen-2.5-7b-instruct'        # Save identifier
```

- `model_name`: Model name as recognized by the API
- `dataset_base`: Base name for Hugging Face datasets
- `save_name`: Identifier used for saving results (should be filesystem-safe)

#### Sampling Parameters
```bash
num_samples=1              # Number of samples per problem
temperature_symbolic=1.0   # Temperature for symbolic datasets
temperature_realistic=0.0  # Temperature for realistic datasets
max_tokens=4096           # Maximum tokens in response
```

- `num_samples`: How many responses to generate per problem
- `temperature_symbolic`: Sampling temperature for symbolic problems (higher for exploration)
- `temperature_realistic`: Sampling temperature for realistic problems (lower for consistency)
- `max_tokens`: Maximum length of model responses

#### Processing Parameters
```bash
batch_size=200        # Batch size for API requests
limit_symbolic=100    # Examples per operation for symbolic
limit_realistic=200   # Examples per operation for realistic
```

- `batch_size`: Number of problems to process in each batch
- `limit_symbolic`: Maximum examples per operation count for symbolic datasets
- `limit_realistic`: Maximum examples per operation count for realistic datasets

### Dataset Configuration

#### Context Lengths
```bash
lengths=( 
    "0"     # Zero noise (no additional context)
    "8k"    # 8K tokens context
    "16k"   # 16K tokens context
    "32k"   # 32K tokens context
)
```

Array of context lengths to evaluate. Use:
- `"0"` for zero noise (no additional context)
- `"8k"`, `"16k"`, `"32k"` for specific token counts

#### Dataset Types
```bash
dataset_suffixes=( 
    "symbolic"  # Abstract mathematical operations
    "medium"    # Realistic word problems
    "hard"      # Complex multi-step problems
)
```

Array of dataset types to evaluate:
- `symbolic`: Abstract mathematical operations using computational graphs
- `medium`: Realistic word problems with moderate complexity
- `hard`: Complex multi-step problems requiring deep reasoning

#### Operation Range Configuration
```bash
declare -A ops_config

# Format: ops_config["<length>_<suffix>"]='{"start": X, "end": Y, "stride": Z}'
ops_config["0_symbolic"]='{"start": 1, "end": 50, "stride": 1}'
ops_config["8k_medium"]='{"start": 2, "end": 30, "stride": 1}'
```

Associative array defining operation ranges for each dataset/length combination:
- `start`: Minimum number of operations
- `end`: Maximum number of operations  
- `stride`: Step size between operation counts

If a combination is not defined, it will be skipped.

#### Filter Configuration (Realistic Only)
```bash
filter_config='[
    {"percentage": 0.4, "template": "crazy_zootopia", "mode": "normalforward"},
    {"percentage": 0.05, "template": "movie_festival_awards", "mode": "normalforward"},
    {"percentage": 0.4, "template": "crazy_zootopia", "mode": "forwardreverse"}
]'
```

JSON array defining problem template distribution for realistic datasets:
- `percentage`: Fraction of problems using this template
- `template`: Template name (e.g., "crazy_zootopia", "movie_festival_awards", "teachers_in_school")
- `mode`: Generation mode ("normalforward" or "forwardreverse")

## Command Line Interface

### Prediction Script (`pred/pred.py`)

```bash
python3 pred/pred.py \
    --dataset-name "InfiniAILab/gsm_infinite_symbolic" \
    --model-name "Qwen/Qwen2.5-7B-Instruct" \
    --save-dataset "symbolic" \
    --save-name "qwen-2.5-7b-instruct" \
    --backend-type "openai" \
    --num-samples 1 \
    --temperature 1.0 \
    --max-tokens 4096 \
    --length "0" \
    --op-range "1,2,3,4,5" \
    --batch-size 200 \
    --limit 100 \
    --filter-config '...'  # For realistic datasets only
```

#### Parameters

- `--dataset-name`: Hugging Face dataset name
- `--model-name`: Model identifier for API
- `--save-dataset`: Local dataset identifier for saving
- `--save-name`: Model identifier for saving
- `--backend-type`: API backend ('openai', 'gemini', 'anthropic')
- `--num-samples`: Number of samples per problem
- `--temperature`: Sampling temperature
- `--max-tokens`: Maximum response length
- `--length`: Context length ("0", "8k", "16k", "32k")
- `--op-range`: Comma-separated list of operation counts
- `--batch-size`: Batch size for processing
- `--limit`: Maximum examples per operation count
- `--filter-config`: JSON filter configuration (realistic only)

### Evaluation Scripts

#### Symbolic Evaluation (`pred/eval_symbolic.py`)
```bash
python3 pred/eval_symbolic.py \
    --save-name "qwen-2.5-7b-instruct" \
    --num-samples 1 \
    --length "0"
```

#### Realistic Evaluation (`pred/eval_realistic.py`)
```bash
python3 pred/eval_realistic.py \
    --save-dataset "medium" \
    --save-name "qwen-2.5-7b-instruct" \
    --num-samples 1 \
    --length "0" \
    --filter-config '...'
```

## Data Generation API

### Symbolic Dataset Generation

```python
from gsm_infinite.data.symbolic.generate_symbolic import generate_symbolic_dataset

dataset = generate_symbolic_dataset(
    num_ops=10,           # Number of operations
    context_length=1000,  # Target context length
    num_examples=100,     # Number of examples
    seed=42              # Random seed
)
```

### Realistic Dataset Generation

```python
from gsm_infinite.data.realistic.forward_generator import ForwardGenerator

generator = ForwardGenerator(
    template="crazy_zootopia",
    mode="normalforward",
    num_ops=15,
    context_length=8000
)

problems = generator.generate_batch(num_examples=50)
```

## Output Formats

### Prediction Files

Predictions are saved as JSON files with this structure:

```json
{
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "dataset": "symbolic",
    "length": "0",
    "num_samples": 1,
    "predictions": [
        {
            "problem_id": "symbolic_ops_5_example_1",
            "problem": "Calculate the result of...",
            "prediction": "The answer is 42",
            "ground_truth": "42",
            "metadata": {
                "ops": 5,
                "template": "basic_arithmetic",
                "context_length": 1024
            }
        }
    ],
    "generation_config": {
        "temperature": 1.0,
        "max_tokens": 4096,
        "backend_type": "openai"
    }
}
```

### Evaluation Results

Evaluation results are saved with this structure:

```json
{
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "dataset": "symbolic",
    "length": "0",
    "overall_accuracy": 0.85,
    "accuracy_by_ops": {
        "1": 0.95,
        "2": 0.90,
        "5": 0.85,
        "10": 0.70
    },
    "detailed_results": [
        {
            "problem_id": "symbolic_ops_5_example_1",
            "correct": true,
            "predicted": "42",
            "ground_truth": "42",
            "ops": 5
        }
    ],
    "metadata": {
        "total_problems": 1000,
        "evaluation_time": "2025-02-26T12:00:00Z",
        "evaluator_model": "Qwen/Qwen2.5-7B-Instruct"
    }
}
```

## Environment Variables

You can override configuration using environment variables:

```bash
# API Configuration
export OPENAI_API_KEY="your_key"
export OPENAI_BASE_URL="your_url"
export GEMINI_API_KEY="your_key"
export ANTHROPIC_API_KEY="your_key"

# Model Configuration
export GSM_MODEL_NAME="your_model"
export GSM_SAVE_NAME="your_save_name"

# Processing Configuration
export GSM_BATCH_SIZE="100"
export GSM_MAX_TOKENS="2048"
```

## Error Handling

### Common Error Codes

- `API_KEY_MISSING`: API key not provided
- `MODEL_NOT_FOUND`: Model name not recognized by API
- `RATE_LIMIT_EXCEEDED`: API rate limit hit
- `INVALID_CONFIG`: Configuration validation failed
- `DATASET_NOT_FOUND`: Specified dataset not available

### Retry Behavior

The system automatically retries failed requests with exponential backoff:
- Initial delay: 1 second
- Maximum delay: 60 seconds
- Maximum retries: 5

## Performance Optimization

### Batch Size Tuning

Optimal batch sizes depend on your API limits:
- OpenAI: 50-200 requests per batch
- Gemini: 10-50 requests per batch
- Anthropic: 20-100 requests per batch

### Memory Management

For large evaluations:
- Use streaming for dataset loading
- Process results incrementally
- Clear intermediate results regularly

### API Rate Limiting

To avoid rate limits:
- Monitor your API usage
- Use multiple API keys if available
- Implement custom delay strategies

---

For additional technical details, see the source code documentation and our [GitHub repository](https://github.com/Infini-AI-Lab/gsm_infinite).


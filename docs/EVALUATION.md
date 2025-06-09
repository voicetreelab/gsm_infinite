# Evaluation Guide

This guide provides comprehensive instructions for evaluating language models on GSM-Infinite benchmarks and understanding the evaluation methodology.

## Overview

GSM-Infinite uses a two-stage evaluation process:
1. **Prediction Generation**: Models generate responses to benchmark problems
2. **Answer Extraction & Scoring**: Automated evaluation extracts and scores answers

## Evaluation Methodology

### 1. Problem Types

#### Symbolic Dataset
- **Abstract mathematical operations** using computational graphs
- **Evaluation**: Exact match against computed ground truth
- **Complexity**: Scales with number of operations (1-50+)
- **Context**: Variable length with high information density

#### Realistic Datasets (Medium/Hard)
- **Medium**: Realistic word problems with moderate complexity
- **Hard**: Complex multi-step problems requiring deep reasoning
- **Evaluation**: Answer extraction followed by numerical comparison
- **Templates**: Multiple problem templates (zootopia, movie_festival, teachers_in_school)

### 2. Evaluation Metrics

#### Primary Metrics
- **Overall Accuracy**: Percentage of correctly solved problems
- **Accuracy by Operations**: Performance vs. problem complexity
- **Context Length Scaling**: Performance vs. input length

#### Advanced Metrics
- **First Failure Point**: Operation count where accuracy drops below thresholds
- **Average Accuracy**: Weighted performance across operation ranges
- **Template Performance**: Accuracy breakdown by problem template

## Running Evaluations

### 1. Basic Evaluation

```bash
cd gsm-infinite

# Configure your model in config.sh
# Set run_sampling=true and run_evaluation=true

# Run complete evaluation
bash run.sh
```

### 2. Evaluation-Only Mode

If you already have predictions and want to re-evaluate:

```bash
# Edit config.sh
run_sampling=false
run_evaluation=true

# Run evaluation only
bash run.sh
```

### 3. Subset-Specific Evaluation

```bash
# Symbolic only
run_symbolic_evaluation=true
run_realistic_evaluation=false

# Realistic only (Medium/Hard)
run_symbolic_evaluation=false
run_realistic_evaluation=true
```

## Evaluation Scripts

### Symbolic Evaluation (`pred/eval_symbolic.py`)

```bash
python3 pred/eval_symbolic.py \
    --save-name "your_model_name" \
    --num-samples 1 \
    --length "0"
```

**Process:**
1. Loads model predictions from `datasets/symbolic/`
2. Uses evaluation model to extract numerical answers
3. Compares against ground truth with exact match
4. Generates accuracy metrics by operation count

**Requirements:**
- Evaluation model configured in `EVAL_OPENAI_*` variables
- Predictions must be in correct JSON format

### Realistic Evaluation (`pred/eval_realistic.py`)

```bash
python3 pred/eval_realistic.py \
    --save-dataset "medium" \
    --save-name "your_model_name" \
    --num-samples 1 \
    --length "0" \
    --filter-config '...'
```

**Process:**
1. Loads model predictions from `datasets/medium/` or `datasets/hard/`
2. Extracts numerical answers using regex patterns
3. Compares against ground truth with numerical tolerance
4. Aggregates results across templates and operation counts

## Understanding Results

### Output Files

#### Prediction Files
```
datasets/
├── symbolic/
│   └── your_model_name_length_0_samples_1.json
├── medium/
│   └── your_model_name_length_0_samples_1.json
└── hard/
    └── your_model_name_length_0_samples_1.json
```

#### Evaluation Results
```
results/
├── symbolic/
│   └── your_model_name_length_0_samples_1_eval.json
├── medium/
│   └── your_model_name_length_0_samples_1_eval.json
└── hard/
    └── your_model_name_length_0_samples_1_eval.json
```

### Result Format

```json
{
    "model_name": "your_model_name",
    "dataset": "symbolic",
    "length": "0",
    "overall_accuracy": 0.85,
    "accuracy_by_ops": {
        "1": 0.95,
        "2": 0.90,
        "5": 0.85,
        "10": 0.70,
        "20": 0.45
    },
    "first_failure_50": 15,
    "first_failure_10": 25,
    "avg_accuracy_op30": 0.72,
    "total_problems": 1000,
    "correct_problems": 850
}
```

### Key Metrics Explained

- **overall_accuracy**: Total percentage of correct answers
- **accuracy_by_ops**: Accuracy for each operation count
- **first_failure_50**: First operation count where accuracy < 50%
- **first_failure_10**: First operation count where accuracy < 10%
- **avg_accuracy_op30**: Average accuracy for operations ≤ 30

## Evaluation Best Practices

### 1. Model Configuration

#### Temperature Settings
```bash
# Symbolic: Higher temperature for exploration
temperature_symbolic=1.0

# Realistic: Lower temperature for consistency  
temperature_realistic=0.0
```

#### Sampling Parameters
```bash
# Multiple samples for robust statistics
num_samples=3

# Sufficient token limit for complete responses
max_tokens=4096
```

### 2. Evaluation Model Selection

For symbolic evaluation, use a capable model for answer extraction:
```bash
# Good choices for evaluation
EVAL_MODEL="Qwen/Qwen2.5-7B-Instruct"
EVAL_MODEL="gpt-3.5-turbo"
EVAL_MODEL="claude-3-haiku"
```

### 3. Batch Processing

```bash
# Optimize batch size for your API limits
batch_size=200  # OpenAI
batch_size=50   # Gemini
batch_size=100  # Anthropic
```

## Advanced Evaluation

### 1. Custom Operation Ranges

```bash
# Test specific complexity ranges
ops_config["0_symbolic"]='{"start": 10, "end": 30, "stride": 2}'
ops_config["0_medium"]='{"start": 5, "end": 15, "stride": 1}'
```

### 2. Template-Specific Analysis

For realistic datasets, analyze performance by template:

```python
# Custom analysis script
import json

def analyze_by_template(results_file):
    with open(results_file) as f:
        results = json.load(f)
    
    template_accuracy = {}
    for result in results['detailed_results']:
        template = result['metadata']['template']
        if template not in template_accuracy:
            template_accuracy[template] = []
        template_accuracy[template].append(result['correct'])
    
    for template, scores in template_accuracy.items():
        accuracy = sum(scores) / len(scores)
        print(f"{template}: {accuracy:.3f}")
```

### 3. Context Length Analysis

```bash
# Evaluate across multiple context lengths
lengths=("0" "8k" "16k" "32k")

# Analyze performance degradation
python3 analyze_context_scaling.py --model your_model_name
```

## Troubleshooting Evaluation

### Common Issues

#### 1. Answer Extraction Failures
**Problem**: Evaluation model fails to extract answers
**Solutions**:
- Use a more capable evaluation model
- Check prediction format and quality
- Verify evaluation model API configuration

#### 2. Inconsistent Results
**Problem**: Results vary between runs
**Solutions**:
- Use consistent temperature settings
- Increase number of samples
- Check for API rate limiting issues

#### 3. Memory/Performance Issues
**Problem**: Evaluation runs slowly or fails
**Solutions**:
- Reduce batch size
- Process subsets incrementally
- Use streaming for large datasets

### Debugging Commands

```bash
# Check prediction file format
python3 -c "import json; print(json.load(open('datasets/symbolic/model_length_0_samples_1.json'))['predictions'][0])"

# Verify evaluation model connectivity
python3 -c "import openai; client = openai.OpenAI(); print(client.models.list())"

# Test answer extraction
python3 pred/test_answer_extraction.py --prediction "The answer is 42"
```

## Evaluation Validation

### 1. Sanity Checks

Before submitting results:
- Verify accuracy trends (should decrease with complexity)
- Check for reasonable absolute performance levels
- Ensure consistent formatting across all outputs

### 2. Reproducibility

```bash
# Set consistent seeds
export PYTHONHASHSEED=42

# Use deterministic settings
temperature=0.0
num_samples=1
```


## Performance Benchmarks

### Expected Runtimes

| Dataset | Size | Prediction Time | Evaluation Time |
|---------|------|----------------|-----------------|
| Symbolic (ops 1-30) | ~3000 problems | 30-60 min | 5-10 min |
| Medium (ops 2-30) | ~5000 problems | 60-120 min | 10-15 min |
| Hard (ops 2-30) | ~5000 problems | 60-120 min | 10-15 min |

*Times assume batch_size=200, API latency ~1s per request*

### Resource Requirements

- **Memory**: 2-4 GB for dataset loading
- **Storage**: 1-2 GB for prediction files
- **API Calls**: ~15,000 calls for complete evaluation

## Integration with Analysis Tools

### 1. Streamlit Dashboard

```bash
# View results interactively
streamlit run app.py
```

### 2. Custom Analysis

```python
# Load and analyze results
import pandas as pd

# Load processed results
df = pd.read_csv('results/processed_results.csv')

# Custom analysis
model_comparison = df.groupby('model_name')['accuracy'].mean()
print(model_comparison.sort_values(ascending=False))
```

### 3. Export Options

```bash
# Export to CSV
python3 export_results.py --format csv --output results.csv

# Export to LaTeX table
python3 export_results.py --format latex --output table.tex
```

## Contributing Evaluation Improvements

### 1. New Metrics

To add new evaluation metrics:
1. Modify evaluation scripts in `pred/`
2. Update result format documentation
3. Add visualization support in dashboard
4. Submit pull request with tests

### 2. Evaluation Models

To add support for new evaluation models:
1. Implement answer extraction logic
2. Add configuration options
3. Test on existing benchmarks
4. Document performance characteristics

---

For questions about evaluation methodology or results interpretation, please refer to our [paper](https://arxiv.org/abs/2502.05252) or contact [yangzho6@andrew.cmu.edu](mailto:yangzho6@andrew.cmu.edu).


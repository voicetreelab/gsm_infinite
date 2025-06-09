<div align="center">
<h1><img src="static/images/facinfinity.webp" height="30px" align="top"/> GSM-Infinite</h1>
<p><em>Infinitely Scalable Reasoning Benchmark for Large Language Models</em></p>
</div>

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2502.05252)
[![Blog](https://img.shields.io/badge/Blog-Website-blue)](https://infini-ai-lab.github.io/gsm_infinite/)
[![Leaderboard](https://img.shields.io/badge/🤗-Leaderboard-yellow)](https://infiniailab-gsm-infinite-leaderboard.hf.space)
[![Datasets](https://img.shields.io/badge/🤗-Datasets-green)](https://huggingface.co/collections/InfiniAILab/gsm-infinite-67aa7b323eb5c4d9c693fe6a)
[![License](https://img.shields.io/badge/License-MIT-purple)](LICENSE)

</div>

<div align="center">
<b><a href="https://github.com/YangZhou08">Yang Zhou*</a></b><sup>1</sup>,
<b><a href="">Hongyi Liu*</a></b><sup>1</sup>,
<b><a href="https://github.com/dreaming-panda">Zhuoming Chen</a></b><sup>1</sup>,
<b><a href="">Yuandong Tian</a></b><sup>2</sup>,
<b><a href="https://github.com/keroro824">Beidi Chen</a></b><sup>1</sup>
<br>
<sup>*</sup>Equal Contributions | <sup>1</sup>Carnegie Mellon University | <sup>2</sup>Meta AI
</div>

---

## Overview

GSM-Infinite is a **completely synthetic reasoning benchmark** that generates problems with infinitely scalable context length and reasoning complexity. Unlike existing benchmarks that rely on text retrieval or summarization, GSM-Infinite creates high information density tasks that can only be solved by long-context LLMs, not by RAG systems.

### Key Features

- 🔄 **Infinitely Scalable**: Generate problems of any context length and reasoning complexity
- 🧮 **High Information Density**: Every token matters - RAG systems cannot solve these problems
- 🎯 **Three Difficulty Levels**: Symbolic, Medium, and Hard subsets
- 📊 **Comprehensive Evaluation**: Built-in evaluation scripts and leaderboards
- 🔬 **Synthetic Generation**: No LLMs in the loop, ensuring unbiased benchmarks

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Infini-AI-Lab/gsm_infinite.git
cd gsm_infinite

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install gsm-infinite
```

### Basic Usage

1. **Configure your setup** by editing `gsm-infinite/config.sh`:
   ```bash
   # Set your API configuration
   backend_type='openai'  # or 'gemini', 'anthropic'
   SAMPLER_OPENAI_BASE_URL='your_api_url'
   SAMPLER_OPENAI_API_KEY='your_api_key'
   
   # Configure model and dataset
   model_name='your_model_name'
   save_name='your_save_name'
   ```

2. **Run evaluation**:
   ```bash
   cd gsm-infinite
   bash run.sh
   ```

3. **View results** with the interactive dashboard:
   ```bash
   streamlit run app.py
   ```

## Project Structure

```
gsm_infinite/
├── gsm-infinite/           # Main package
│   ├── app.py             # Streamlit results viewer
│   ├── config.sh          # Configuration file
│   ├── run.sh             # Main execution script
│   ├── preprocess.py      # Data preprocessing
│   ├── data/              # Data generation modules
│   │   ├── symbolic/      # Symbolic dataset generation
│   │   └── realistic/     # Medium/Hard dataset generation
│   └── pred/              # Prediction and evaluation scripts
├── docs/                  # Detailed documentation
├── static/                # Web assets and images
├── requirements.txt       # Python dependencies
└── pyproject.toml        # Package configuration
```

## Dataset Information

GSM-Infinite provides three types of datasets:

| Dataset | Description | Context Length |
|---------|-------------|----------------|
| **Symbolic** | Abstract mathematical operations | 0-32K+ tokens |
| **Medium** | Realistic problems with at most 2-entity implicit relationship | 0-32K+ tokens |
| **Hard** | Realistic problems with at most 3-entity implicit relationship | 0-32K+ tokens |

## Documentation

For detailed information, please refer to our comprehensive documentation:

- 📖 **[Installation Guide](docs/INSTALLATION.md)** - Detailed setup instructions
- 🚀 **[Usage Guide](docs/USAGE.md)** - Complete usage examples
- 🔧 **[Data Generation](docs/DATA_GENERATION.md)** - Generate custom datasets
- 📊 **[Evaluation Guide](docs/EVALUATION.md)** - Evaluate your models
- 🏆 **[Leaderboards](docs/LEADERBOARDS.md)** - Current model rankings
- 🔍 **[API Reference](docs/API_REFERENCE.md)** - Configuration options
<!-- - 🤝 **[Contributing](docs/CONTRIBUTING.md)** - How to contribute -->

## Results

Our benchmark reveals significant differences in long-context reasoning capabilities across models. See our [leaderboards](https://infiniailab-gsm-infinite-leaderboard.hf.space) for the latest results.

**Top performers on Zero Noise tasks:**
- DeepSeek-R1: 8534.88 average score
- GPT-o3-mini: 6931.88 average score  
- GPT-o1-mini: 4951.11 average score

For complete results and analysis, visit our [paper](https://arxiv.org/abs/2502.05252) and [leaderboard](docs/LEADERBOARDS.md).

## Why GSM-Infinite?

<div align="center">
<img src="static/images/rag22.png" width="600"/>
<p><em>RAG systems fail on GSM-Infinite due to high information density</em></p>
</div>

Traditional long-context benchmarks can often be solved by RAG systems, making them insufficient for evaluating true long-context reasoning. GSM-Infinite addresses this by:

1. **High Information Density**: Every part of the context is essential
2. **Reasoning Complexity**: Requires multi-step mathematical reasoning
3. **Infinite Scalability**: Generate unlimited test cases at any difficulty

## Citation

If you use GSM-Infinite in your research, please cite our paper:

```bibtex
@misc{zhou2025gsminfinitellmsbehaveinfinitely,
    title={GSM-Infinite: How Do Your LLMs Behave over Infinitely Increasing Context Length and Reasoning Complexity?}, 
    author={Yang Zhou and Hongyi Liu and Zhuoming Chen and Yuandong Tian and Beidi Chen},
    year={2025},
    eprint={2502.05252},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2502.05252}, 
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/Infini-AI-Lab/gsm_infinite/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Infini-AI-Lab/gsm_infinite/discussions)
- 📧 **Contact**: [yangzho6@andrew.cmu.edu](mailto:yangzho6@andrew.cmu.edu)

---

<div align="center">
Made with ❤️ by the Infini-AI Lab team
</div>


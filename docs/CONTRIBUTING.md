# Contributing to GSM-Infinite

We welcome contributions to GSM-Infinite! This guide will help you get started with contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Types of Contributions](#types-of-contributions)
- [Submitting Changes](#submitting-changes)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Familiarity with the GSM-Infinite benchmark (read our [paper](https://arxiv.org/abs/2502.05252))

### Development Setup

1. **Fork the repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/gsm_infinite.git
   cd gsm_infinite
   ```

2. **Set up development environment**
   ```bash
   # Create virtual environment
   python -m venv gsm_infinite_dev
   source gsm_infinite_dev/bin/activate  # On Windows: gsm_infinite_dev\Scripts\activate
   
   # Install in development mode
   pip install -e .
   
   # Install development dependencies
   pip install pytest black flake8 mypy
   ```

3. **Set up pre-commit hooks** (optional but recommended)
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Contributing Guidelines

### Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Follow our community guidelines

### Before You Start

1. **Check existing issues** - Look for existing issues or discussions related to your contribution
2. **Create an issue** - For significant changes, create an issue to discuss your approach
3. **Get feedback** - Engage with maintainers and community members early

## Types of Contributions

### 🐛 Bug Fixes

- Fix issues in data generation
- Resolve evaluation inconsistencies
- Improve error handling
- Fix documentation errors

### ✨ New Features

- Add support for new model APIs
- Implement new dataset generation methods
- Create new evaluation metrics
- Add visualization tools

### 📚 Documentation

- Improve existing documentation
- Add examples and tutorials
- Create API documentation
- Write blog posts or guides

### 🧪 Testing

- Add unit tests
- Create integration tests
- Improve test coverage
- Add performance benchmarks

### 🎨 UI/UX Improvements

- Enhance the Streamlit dashboard
- Improve visualization components
- Add interactive features
- Optimize user experience

## Submitting Changes

### Pull Request Process

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make your changes**
   - Write clean, well-documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   # Run tests
   pytest tests/
   
   # Check code style
   black gsm-infinite/
   flake8 gsm-infinite/
   
   # Type checking
   mypy gsm-infinite/
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add support for new model API"
   # Use conventional commit format
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   # Create pull request on GitHub
   ```

### Commit Message Format

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

Examples:
```
feat(data): add support for custom templates
fix(eval): resolve accuracy calculation bug
docs(api): update configuration reference
```

## Code Style

### Python Code Style

We follow [PEP 8](https://pep8.org/) with some modifications:

- Line length: 88 characters (Black default)
- Use type hints for function signatures
- Use docstrings for all public functions and classes

### Code Formatting

We use [Black](https://black.readthedocs.io/) for code formatting:

```bash
# Format all code
black gsm-infinite/

# Check formatting
black --check gsm-infinite/
```

### Linting

We use [flake8](https://flake8.pycqa.org/) for linting:

```bash
flake8 gsm-infinite/
```

### Type Checking

We use [mypy](https://mypy.readthedocs.io/) for type checking:

```bash
mypy gsm-infinite/
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_symbolic.py

# Run with coverage
pytest --cov=gsm-infinite tests/
```

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names
- Test both success and failure cases
- Mock external API calls

Example test:
```python
import pytest
from gsm_infinite.data.symbolic.utils import generate_expression

def test_generate_expression_basic():
    """Test basic expression generation."""
    expr = generate_expression(num_ops=3)
    assert expr is not None
    assert len(expr.split()) >= 3

def test_generate_expression_invalid_ops():
    """Test expression generation with invalid parameters."""
    with pytest.raises(ValueError):
        generate_expression(num_ops=0)
```

### Test Categories

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows
4. **Performance Tests**: Test performance characteristics

## Documentation

### Documentation Standards

- Use clear, concise language
- Include code examples
- Add type hints and docstrings
- Update relevant documentation for changes

### Docstring Format

We use Google-style docstrings:

```python
def generate_dataset(num_examples: int, difficulty: str) -> List[Dict]:
    """Generate a dataset with specified parameters.
    
    Args:
        num_examples: Number of examples to generate
        difficulty: Difficulty level ('easy', 'medium', 'hard')
        
    Returns:
        List of generated examples as dictionaries
        
    Raises:
        ValueError: If difficulty is not recognized
        
    Example:
        >>> dataset = generate_dataset(100, 'medium')
        >>> len(dataset)
        100
    """
```

### Building Documentation

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Build documentation
cd docs/
make html
```

## Specific Contribution Areas

### Adding New Model Support

To add support for a new model API:

1. Create a new backend in `pred/backends/`
2. Implement the required interface
3. Add configuration options
4. Update documentation
5. Add tests

### Creating New Dataset Types

To add a new dataset type:

1. Create generation scripts in `data/`
2. Implement evaluation logic
3. Add configuration options
4. Update the main pipeline
5. Add documentation and examples

### Improving Evaluation

To improve evaluation methods:

1. Analyze current evaluation limitations
2. Propose new metrics or methods
3. Implement changes with backward compatibility
4. Validate against existing results
5. Update documentation

## Getting Help

### Communication Channels

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: [yangzho6@andrew.cmu.edu](mailto:yangzho6@andrew.cmu.edu) for direct contact

### Resources

- [Project Documentation](../README.md)
- [API Reference](API_REFERENCE.md)
- [Usage Guide](USAGE.md)
- [Paper](https://arxiv.org/abs/2502.05252)

## Recognition

Contributors will be recognized in:
- GitHub contributors list
- Release notes for significant contributions
- Academic papers for substantial research contributions

## License

By contributing to GSM-Infinite, you agree that your contributions will be licensed under the same license as the project (MIT License).

---

Thank you for contributing to GSM-Infinite! Your contributions help advance the field of long-context language model evaluation.


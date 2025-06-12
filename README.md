# LLM Sweep

A Python script that queries free Large Language Models (LLMs) via OpenRouter.ai API.

[![Python Tests](https://github.com/spencerahill/llm-sweep/actions/workflows/tests.yml/badge.svg)](https://github.com/spencerahill/llm-sweep/actions/workflows/tests.yml)

## Features

- Query 100% free LLMs available through OpenRouter
- Simple command-line interface
- Support for multiple free models including:
  - Meta Llama 3.1/3.2 models
  - Google Gemma models
  - Mistral models
  - Qwen models
  - Microsoft Phi models
- Comprehensive error handling
- Full test coverage with pytest

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Get your free OpenRouter API key:
   - Visit [OpenRouter.ai](https://openrouter.ai)
   - Sign up for a free account
   - Generate an API key

3. Set your API key as an environment variable:
```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

## Usage

### Basic Usage
```bash
python llm_sweep.py "What is the capital of France?"
```

### List Available Models
```bash
python llm_sweep.py --list-models
```

### Use a Specific Model
```bash
python llm_sweep.py "Explain quantum computing" --model meta-llama/llama-3.1-8b-instruct:free
```

### Limit Response Length
```bash
python llm_sweep.py "Write a haiku about programming" --max-tokens 100
```

### Verbose Output
```bash
python llm_sweep.py "Hello, how are you?" --verbose
```

## Command Line Options

- `prompt` - The text prompt to send to the LLM (required unless using --list-models)
- `--model MODEL` - Specific model to use (default: first available free model)
- `--max-tokens N` - Maximum number of tokens in response (default: 512)
- `--list-models` - List available free models and exit
- `--verbose` - Enable verbose output showing model info and token usage
- `--help` - Show help message

## Available Free Models

The script includes the following free models:
- `meta-llama/llama-3.1-8b-instruct:free`
- `meta-llama/llama-3.2-3b-instruct:free`
- `meta-llama/llama-3.2-1b-instruct:free`
- `google/gemma-2-9b-it:free`
- `mistralai/mistral-7b-instruct:free`
- `qwen/qwen-2.5-7b-instruct:free`
- `microsoft/phi-3-mini-128k-instruct:free`

## Testing

Run the comprehensive test suite:
```bash
python -m pytest test_llm_sweep.py -v
```

The tests cover:
- API client functionality
- Error handling
- Command line interface
- Integration workflows
- Mock API responses

## Error Handling

The script handles various error conditions:
- Missing API key
- Network errors
- Invalid API responses
- Rate limiting
- Invalid command line arguments

## Files

- `llm_sweep.py` - Main script
- `test_llm_sweep.py` - Comprehensive test suite
- `requirements.txt` - Python dependencies
- `README.md` - This documentation

## Dependencies

- `requests>=2.31.0` - HTTP client for API calls
- `pytest>=7.4.0` - Testing framework
- `pytest-mock>=3.11.1` - Mock utilities for tests

## License

This project is provided as-is for educational and research purposes. 
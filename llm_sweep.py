#!/usr/bin/env python3
"""
LLM Sweep - A script to query free LLMs via OpenRouter API

This script accepts a prompt as a command line argument and sends it to a free LLM
available through OpenRouter.ai. The API key should be stored in the OPENROUTER_API_KEY
environment variable.
"""

import os
import sys
import argparse
import requests
import json
import time
import platform
from typing import Dict, Any, Optional


class OpenRouterClient:
    """Client for interacting with OpenRouter API"""
    
    BASE_URL = "https://openrouter.ai/api/v1"
    
    # List of free models available on OpenRouter (as of 2024)
    FREE_MODELS = [
        "meta-llama/llama-3.1-8b-instruct:free",
        "meta-llama/llama-3.2-3b-instruct:free", 
        "meta-llama/llama-3.2-1b-instruct:free",
        "google/gemma-2-9b-it:free",
        "mistralai/mistral-7b-instruct:free",
        "qwen/qwen-2.5-7b-instruct:free",
        "microsoft/phi-3-mini-128k-instruct:free"
    ]
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenRouter client
        
        Args:
            api_key: OpenRouter API key. If None, will try to get from environment
        """
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable.")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo",  # Optional: for tracking
            "X-Title": "LLM Sweep Script"  # Optional: for tracking
        }
    
    def query_llm(self, prompt: str, model: Optional[str] = None, max_tokens: int = 512, seed: Optional[int] = None, temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        Send a query to the specified LLM model
        
        Args:
            prompt: The text prompt to send to the model
            model: The model to use. If None, uses the first available free model
            max_tokens: Maximum number of tokens in the response
            seed: Random seed for reproducible responses. If None, no seed is used
            temperature: Controls randomness (0.0-2.0). If None, uses the model's default temperature
            
        Returns:
            Dictionary containing the response from the API
            
        Raises:
            requests.RequestException: If the API request fails
            ValueError: If the response format is unexpected
        """
        if model is None:
            model = self.FREE_MODELS[0]  # Default to first free model
        
        # Pre-validate model name to give better error messages
        if model and model not in self.FREE_MODELS:
            # Check for common mistakes
            if '/' in model and model.count('/') > 1:
                # Model looks malformed (like "qwen/google/gemma-2-9b-it:free")
                raise ValueError(f"Model name '{model}' appears to be malformed. Check the format - it should be 'provider/model-name:free'.")
            else:
                raise ValueError(f"Model '{model}' is not in the list of available free models. Use --list-models to see available options.")

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens
        }

        # Add temperature only if explicitly provided
        if temperature is not None:
            payload["temperature"] = temperature

        # Add seed if provided for reproducible responses
        if seed is not None:
            payload["seed"] = seed
        
        try:
            response = requests.post(
                f"{self.BASE_URL}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Validate response format
            if 'choices' not in data or not data['choices']:
                raise ValueError(f"Unexpected response format: {data}")
            
            return data
            
        except requests.HTTPError as e:
            # Handle 400 errors specially to provide better model error messages
            if hasattr(e, 'response') and e.response and e.response.status_code == 400:
                try:
                    error_data = e.response.json()
                    error_message = error_data.get('error', {}).get('message', str(e))
                    
                    # Check if it's likely a model-related error
                    if any(keyword in error_message.lower() for keyword in ['model', 'not found', 'invalid']):
                        # Create a custom exception with model context
                        raise ValueError(f"Invalid model specification: {error_message}")
                    else:
                        raise ValueError(f"API error: {error_message}")
                except (ValueError, KeyError):
                    # If we can't parse the error response, provide a generic helpful message
                    if model and model not in self.FREE_MODELS:
                        raise ValueError(f"Model '{model}' may not be available. Use --list-models to see available options.")
                    raise ValueError(f"API request failed with 400 error: {e}")
            else:
                raise requests.RequestException(f"API request failed: {e}")
        except requests.RequestException as e:
            raise requests.RequestException(f"API request failed: {e}")
    
    def get_response_text(self, response_data: Dict[str, Any]) -> str:
        """
        Extract the text response from the API response
        
        Args:
            response_data: The response dictionary from query_llm
            
        Returns:
            The text content of the response
        """
        try:
            return response_data['choices'][0]['message']['content']
        except (KeyError, IndexError) as e:
            raise ValueError(f"Could not extract response text: {e}")
    
    def list_available_models(self) -> list:
        """
        Get list of available free models
        
        Returns:
            List of available free model names
        """
        return self.FREE_MODELS.copy()


def create_response_record(args, response_data, response_text, start_time, end_time, prompt_text, repetition=1, total_repetitions=1):
    """Create a structured JSON record for a single response"""
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime(start_time)),
        "prompt": prompt_text,
        "model": response_data.get('model', args.model or 'meta-llama/llama-3.1-8b-instruct:free'),
        "parameters": {
            "max_tokens": args.max_tokens,
            "repetition": repetition,
            "total_repetitions": total_repetitions,
            "seed": args.seed,
            "temperature": args.temperature
        },
        "response": {
            "text": response_text
        },
        "performance": {
            "response_time_ms": int((end_time - start_time) * 1000)
        },
        "system_info": {
            "platform": platform.system(),
            "python_version": platform.python_version()
        }
    }
    
    # Add usage information if available
    if 'usage' in response_data:
        usage = response_data['usage']
        record["performance"].update({
            "total_tokens": usage.get('total_tokens', 0),
            "prompt_tokens": usage.get('prompt_tokens', 0),
            "completion_tokens": usage.get('completion_tokens', 0)
        })
    
    return record


def write_json_output(records, output_file, append=True):
    """Write records to a JSON file.
    
    Args:
        records: List of record dictionaries to write
        output_file: Path to the output file
        append: If True, append to existing file. If False, overwrite.
               Defaults to True.
    """
    # Ensure all records have timestamps
    current_time = time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime())
    for record in records:
        if "timestamp" not in record:
            record["timestamp"] = current_time
    
    try:
        if append and os.path.exists(output_file):
            # Read existing data
            with open(output_file, 'r') as f:
                try:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = [existing_data]
                except json.JSONDecodeError:
                    existing_data = []
            
            # Append new records
            existing_data.extend(records)
            
            # Write back to file
            with open(output_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
        else:
            # Write new file
            with open(output_file, 'w') as f:
                json.dump(records, f, indent=2)
    except Exception as e:
        raise RuntimeError(f"Error writing to output file: {str(e)}")


def main():
    """Main function to handle command line arguments and execute the query"""
    parser = argparse.ArgumentParser(
        description="Query free LLMs via OpenRouter API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python llm_sweep.py "What is the capital of France?"
  python llm_sweep.py "Explain quantum computing" --model meta-llama/llama-3.1-8b-instruct:free
  python llm_sweep.py "Write a haiku about programming" --max-tokens 100
  python llm_sweep.py "Generate a creative story" --repetitions 3
  python llm_sweep.py "What's your favorite color?" --repetitions 5 --verbose
  python llm_sweep.py "Tell me a joke" --seed 42
  python llm_sweep.py "Random story" --repetitions 3 --seed 123 --verbose
  python llm_sweep.py "Write creatively" --temperature 1.2
  python llm_sweep.py "Be precise" --temperature 0.1 --seed 42 --verbose
  python llm_sweep.py "Analyze this" --output-file results.json
  python llm_sweep.py "Compare models" --repetitions 5 --output-file experiment.json --verbose
  python llm_sweep.py --prompt-file my_prompt.txt
  python llm_sweep.py --prompt-file long_essay.txt --repetitions 3 --seed 42 --verbose
  python llm_sweep.py --prompt-file research_query.txt --output-file analysis.json
        """
    )
    
    parser.add_argument(
        "prompt",
        nargs='?',  # Make prompt optional
        help="The prompt to send to the LLM"
    )
    
    parser.add_argument(
        "--model",
        help="Specific model to use (default: first available free model)",
        default=None
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens in response (default: 512)"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available free models and exit"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of times to repeat the prompt (default: 1)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible responses"
    )
    
    def temperature_type(value):
        """Validate temperature is a float between 0.0 and 2.0"""
        try:
            fval = float(value)
            if fval < 0.0 or fval > 2.0:
                raise argparse.ArgumentTypeError(f"Temperature must be between 0.0 and 2.0, got {fval}")
            return fval
        except ValueError:
            raise argparse.ArgumentTypeError(f"Temperature must be a number, got '{value}'")
    
    parser.add_argument(
        "--temperature",
        type=temperature_type,
        help="Controls randomness (0.0-2.0, default: 0.7)"
    )
    
    def output_format_type(value):
        """Validate output format is supported"""
        valid_formats = ['json']
        if value.lower() not in valid_formats:
            raise argparse.ArgumentTypeError(f"Output format must be one of {valid_formats}, got '{value}'")
        return value.lower()
    
    parser.add_argument(
        "--output-format",
        type=output_format_type,
        help="Output format for structured data (json)"
    )
    
    parser.add_argument(
        "--output-file",
        help="File to write structured output to (implies --output-format json if format not specified)"
    )
    
    parser.add_argument(
        "--prompt-file",
        help="Read prompt from a text file (mutually exclusive with positional prompt argument)"
    )
    
    args = parser.parse_args()
    
    # Validate output arguments
    if args.output_format and not args.output_file:
        parser.error("--output-format requires --output-file")
    
    # Default to JSON format if output file is specified without format
    if args.output_file and not args.output_format:
        args.output_format = 'json'
    
    # Validate prompt arguments - exactly one of prompt or prompt_file must be provided
    if args.prompt and args.prompt_file:
        parser.error("Cannot specify both prompt argument and --prompt-file option")
    
    # Read prompt from file if specified
    prompt_text = None
    if args.prompt_file:
        try:
            with open(args.prompt_file, 'r', encoding='utf-8') as f:
                prompt_text = f.read()
            
            # Check if file is empty or contains only whitespace
            if not prompt_text.strip():
                print("Error: prompt file is empty", file=sys.stderr)
                return 1
                
        except FileNotFoundError:
            print(f"Error reading prompt file '{args.prompt_file}': File not found", file=sys.stderr)
            return 1
        except PermissionError:
            print(f"Error reading prompt file '{args.prompt_file}': Permission denied", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Error reading prompt file '{args.prompt_file}': {e}", file=sys.stderr)
            return 1
    else:
        prompt_text = args.prompt
    
    try:
        client = OpenRouterClient()
        
        if args.list_models:
            print("Available free models:")
            for model in client.list_available_models():
                print(f"  - {model}")
            return 0
        
        # Validate that prompt is provided when not listing models
        if not prompt_text:
            print("Error: prompt is required when not using --list-models", file=sys.stderr)
            return 1
        
        # Validate repetitions count
        if args.repetitions < 1:
            print("Error: repetitions must be a positive integer", file=sys.stderr)
            return 1
        
        if args.verbose:
            print(f"Sending prompt to model: {args.model or client.FREE_MODELS[0]}")
            if args.prompt_file:
                print(f"Prompt file: {args.prompt_file}")
            print(f"Prompt: {prompt_text}")
            print(f"Repetitions: {args.repetitions}")
            if args.seed is not None:
                print(f"Seed: {args.seed}")
            if args.temperature is not None:
                print(f"Temperature: {args.temperature}")
            print("-" * 50)
        
        total_tokens = 0
        json_records = []
        
        for i in range(args.repetitions):
            if args.repetitions > 1 and not args.output_file:
                print(f"=== Response {i + 1} ===")
            
            start_time = time.time()
            response_data = client.query_llm(
                prompt=prompt_text,
                model=args.model,
                max_tokens=args.max_tokens,
                seed=args.seed,
                temperature=args.temperature
            )
            end_time = time.time()
            
            response_text = client.get_response_text(response_data)
            
            # Collect data for JSON output if needed
            if args.output_file:
                record = create_response_record(
                    args, response_data, response_text, 
                    start_time, end_time, prompt_text,
                    repetition=i + 1, 
                    total_repetitions=args.repetitions
                )
                json_records.append(record)
            
            # Console output (unless writing to file only)
            if not args.output_file:
                print(response_text)
            elif args.verbose:
                # Still show response in verbose mode even when writing to file
                if args.repetitions > 1:
                    print(f"=== Response {i + 1} ===")
                print(response_text)
            
            if args.verbose:
                print("-" * 30)
                print(f"Model used: {response_data.get('model', 'Unknown')}")
                if 'usage' in response_data:
                    usage = response_data['usage']
                    tokens = usage.get('total_tokens', 0)
                    print(f"Tokens used: {tokens}")
                    total_tokens += tokens
                print("-" * 30)
            
            # Add spacing between responses (except for the last one)
            if args.repetitions > 1 and i < args.repetitions - 1:
                if not args.output_file or args.verbose:
                    print()
        
        if args.verbose and args.repetitions > 1:
            print("-" * 50)
            print(f"Total repetitions: {args.repetitions}")
            print(f"Total tokens used: {total_tokens}")
            print("-" * 50)
        
        # Write JSON output if requested
        if args.output_file:
            # For single response, write object; for multiple, write array
            output_data = json_records[0] if args.repetitions == 1 else json_records
            try:
                write_json_output(output_data if isinstance(output_data, list) else [output_data], args.output_file, append=True)
            except Exception as e:
                print(f"Error writing JSON output to {args.output_file}: {e}", file=sys.stderr)
                return 1
        
        return 0
        
    except ValueError as e:
        error_msg = str(e)
        print(f"Error: {error_msg}", file=sys.stderr)
        
        # Provide additional help for model-related errors
        if "model" in error_msg.lower() and args.model:
            print(f"\nYou specified model: '{args.model}'", file=sys.stderr)
            print("Available free models:", file=sys.stderr)
            try:
                temp_client = OpenRouterClient()
                for model in temp_client.list_available_models():
                    print(f"  - {model}", file=sys.stderr)
            except:
                # Fallback to hardcoded list if client creation fails
                fallback_models = [
                    "meta-llama/llama-3.1-8b-instruct:free",
                    "microsoft/phi-3-mini-128k-instruct:free", 
                    "microsoft/phi-3-medium-128k-instruct:free",
                    "google/gemma-2-9b-it:free",
                    "mistralai/mistral-7b-instruct:free",
                    "huggingfaceh4/zephyr-7b-beta:free",
                    "openchat/openchat-7b:free"
                ]
                for model in fallback_models:
                    print(f"  - {model}", file=sys.stderr)
            print("\nUse --list-models to see all available options.", file=sys.stderr)
        
        return 1
    except requests.RequestException as e:
        print(f"Network error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

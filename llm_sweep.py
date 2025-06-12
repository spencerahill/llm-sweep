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
    
    def query_llm(self, prompt: str, model: Optional[str] = None, max_tokens: int = 512, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Send a query to the specified LLM model
        
        Args:
            prompt: The text prompt to send to the model
            model: The model to use. If None, uses the first available free model
            max_tokens: Maximum number of tokens in the response
            seed: Random seed for reproducible responses. If None, no seed is used
            
        Returns:
            Dictionary containing the response from the API
            
        Raises:
            requests.RequestException: If the API request fails
            ValueError: If the response format is unexpected
        """
        if model is None:
            model = self.FREE_MODELS[0]  # Default to first free model
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
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
    
    args = parser.parse_args()
    
    try:
        client = OpenRouterClient()
        
        if args.list_models:
            print("Available free models:")
            for model in client.list_available_models():
                print(f"  - {model}")
            return 0
        
        # Validate that prompt is provided when not listing models
        if not args.prompt:
            print("Error: prompt is required when not using --list-models", file=sys.stderr)
            return 1
        
        # Validate repetitions count
        if args.repetitions < 1:
            print("Error: repetitions must be a positive integer", file=sys.stderr)
            return 1
        
        if args.verbose:
            print(f"Sending prompt to model: {args.model or client.FREE_MODELS[0]}")
            print(f"Prompt: {args.prompt}")
            print(f"Repetitions: {args.repetitions}")
            if args.seed is not None:
                print(f"Seed: {args.seed}")
            print("-" * 50)
        
        total_tokens = 0
        
        for i in range(args.repetitions):
            if args.repetitions > 1:
                print(f"=== Response {i + 1} ===")
            
            response_data = client.query_llm(
                prompt=args.prompt,
                model=args.model,
                max_tokens=args.max_tokens,
                seed=args.seed
            )
            
            response_text = client.get_response_text(response_data)
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
                print()
        
        if args.verbose and args.repetitions > 1:
            print("-" * 50)
            print(f"Total repetitions: {args.repetitions}")
            print(f"Total tokens used: {total_tokens}")
            print("-" * 50)
        
        return 0
        
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
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

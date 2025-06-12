#!/usr/bin/env python3
"""
Unit tests for llm-sweep.py

This module contains comprehensive tests for the OpenRouter LLM client,
including mocked API responses, error handling, and command line interface testing.
"""

import pytest
import requests
import json
import os
from unittest.mock import Mock, patch, MagicMock, mock_open
from llm_sweep import OpenRouterClient, main, create_response_record, write_json_output
import sys
from io import StringIO
import argparse


class TestOpenRouterClient:
    """Test cases for the OpenRouterClient class"""
    
    def test_init_with_api_key(self):
        """Test client initialization with provided API key"""
        client = OpenRouterClient(api_key="test-key")
        assert client.api_key == "test-key"
        assert "Bearer test-key" in client.headers["Authorization"]
    
    def test_init_with_env_var(self):
        """Test client initialization using environment variable"""
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'env-key'}):
            client = OpenRouterClient()
            assert client.api_key == "env-key"
    
    def test_init_no_api_key(self):
        """Test client initialization fails without API key"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OpenRouter API key not found"):
                OpenRouterClient()
    
    def test_list_available_models(self):
        """Test listing available models"""
        client = OpenRouterClient(api_key="test-key")
        models = client.list_available_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert "meta-llama/llama-3.1-8b-instruct:free" in models
    
    @patch('requests.post')
    def test_query_llm_success(self, mock_post):
        """Test successful LLM query"""
        # Mock successful API response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "This is a test response from the LLM."
                    }
                }
            ],
            "model": "meta-llama/llama-3.1-8b-instruct:free",
            "usage": {
                "total_tokens": 25,
                "prompt_tokens": 10,
                "completion_tokens": 15
            }
        }
        mock_post.return_value = mock_response
        
        client = OpenRouterClient(api_key="test-key")
        result = client.query_llm("Test prompt")
        
        # Verify the request was made correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "https://openrouter.ai/api/v1/chat/completions"
        
        # Verify request payload
        payload = call_args[1]['json']
        assert payload['model'] == "meta-llama/llama-3.1-8b-instruct:free"
        assert payload['messages'][0]['content'] == "Test prompt"
        assert payload['max_tokens'] == 512
        
        # Verify response
        assert result['choices'][0]['message']['content'] == "This is a test response from the LLM."
    
    @patch('requests.post')
    def test_query_llm_with_custom_model(self, mock_post):
        """Test LLM query with custom model"""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_post.return_value = mock_response
        
        client = OpenRouterClient(api_key="test-key")
        client.query_llm("Test prompt", model="google/gemma-2-9b-it:free")
        
        # Verify custom model was used
        payload = mock_post.call_args[1]['json']
        assert payload['model'] == "google/gemma-2-9b-it:free"
    
    @patch('requests.post')
    def test_query_llm_with_seed(self, mock_post):
        """Test LLM query with seed parameter for reproducibility"""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Seeded response"}}]
        }
        mock_post.return_value = mock_response
        
        client = OpenRouterClient(api_key="test-key")
        client.query_llm("Test prompt", seed=42)
        
        # Verify seed was included in the payload
        payload = mock_post.call_args[1]['json']
        assert payload['seed'] == 42
        assert payload['messages'][0]['content'] == "Test prompt"
    
    @patch('requests.post')
    def test_query_llm_without_seed(self, mock_post):
        """Test LLM query without seed parameter (default behavior)"""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Non-seeded response"}}]
        }
        mock_post.return_value = mock_response
        
        client = OpenRouterClient(api_key="test-key")
        client.query_llm("Test prompt")
        
        # Verify seed is not in the payload when not specified
        payload = mock_post.call_args[1]['json']
        assert 'seed' not in payload
        assert payload['messages'][0]['content'] == "Test prompt"
    
    @patch('requests.post')
    def test_query_llm_with_temperature(self, mock_post):
        """Test LLM query with custom temperature parameter"""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Temperature controlled response"}}]
        }
        mock_post.return_value = mock_response
        
        client = OpenRouterClient(api_key="test-key")
        client.query_llm("Test prompt", temperature=0.2)
        
        # Verify temperature was included in the payload
        payload = mock_post.call_args[1]['json']
        assert payload['temperature'] == 0.2
        assert payload['messages'][0]['content'] == "Test prompt"
    
    @patch('requests.post')
    def test_query_llm_without_temperature(self, mock_post):
        """Test LLM query uses default temperature when not specified"""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Default temperature response"}}]
        }
        mock_post.return_value = mock_response
        
        client = OpenRouterClient(api_key="test-key")
        client.query_llm("Test prompt")
        
        payload = mock_post.call_args[1]['json']
        assert 'temperature' not in payload
        assert payload['messages'][0]['content'] == "Test prompt"
    
    @patch('requests.post')
    def test_query_llm_with_temperature_and_seed(self, mock_post):
        """Test LLM query with both temperature and seed parameters"""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Controlled response"}}]
        }
        mock_post.return_value = mock_response
        
        client = OpenRouterClient(api_key="test-key")
        client.query_llm("Test prompt", temperature=0.1, seed=42)
        
        # Verify both temperature and seed are in the payload
        payload = mock_post.call_args[1]['json']
        assert payload['temperature'] == 0.1
        assert payload['seed'] == 42
        assert payload['messages'][0]['content'] == "Test prompt"
    
    @patch('requests.post')
    def test_query_llm_request_exception(self, mock_post):
        """Test handling of request exceptions"""
        mock_post.side_effect = requests.RequestException("Network error")
        
        client = OpenRouterClient(api_key="test-key")
        with pytest.raises(requests.RequestException, match="API request failed"):
            client.query_llm("Test prompt")
    
    @patch('requests.post')
    def test_query_llm_http_error(self, mock_post):
        """Test handling of HTTP errors"""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_post.return_value = mock_response
        
        client = OpenRouterClient(api_key="test-key")
        with pytest.raises(requests.RequestException):
            client.query_llm("Test prompt")
    
    @patch('requests.post')
    def test_query_llm_invalid_response(self, mock_post):
        """Test handling of invalid API response format"""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"error": "Invalid request"}
        mock_post.return_value = mock_response
        
        client = OpenRouterClient(api_key="test-key")
        with pytest.raises(ValueError, match="Unexpected response format"):
            client.query_llm("Test prompt")
    
    def test_get_response_text_success(self):
        """Test extracting response text from valid response"""
        client = OpenRouterClient(api_key="test-key")
        response_data = {
            "choices": [
                {
                    "message": {
                        "content": "This is the LLM response."
                    }
                }
            ]
        }
        
        text = client.get_response_text(response_data)
        assert text == "This is the LLM response."
    
    def test_get_response_text_invalid_format(self):
        """Test handling of invalid response format when extracting text"""
        client = OpenRouterClient(api_key="test-key")
        
        # Test missing choices
        with pytest.raises(ValueError, match="Could not extract response text"):
            client.get_response_text({})
        
        # Test empty choices
        with pytest.raises(ValueError, match="Could not extract response text"):
            client.get_response_text({"choices": []})
        
        # Test missing message content
        with pytest.raises(ValueError, match="Could not extract response text"):
            client.get_response_text({"choices": [{"message": {}}]})


class TestMainFunction:
    """Test cases for the main function and command line interface"""
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm-sweep.py', 'Test prompt'])
    def test_main_basic_usage(self, mock_client_class):
        """Test basic usage of main function"""
        # Mock the client
        mock_client = Mock()
        mock_client.query_llm.return_value = {
            "choices": [{"message": {"content": "LLM response"}}],
            "model": "test-model"
        }
        mock_client.get_response_text.return_value = "LLM response"
        mock_client_class.return_value = mock_client
        
        # Capture stdout
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            result = main()
        
        assert result == 0
        assert "LLM response" in captured_output.getvalue()
        mock_client.query_llm.assert_called_once_with(
            prompt="Test prompt",
            model=None,
            max_tokens=512,
            seed=None,
            temperature=None
        )
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm-sweep.py', '--list-models'])
    def test_main_list_models(self, mock_client_class):
        """Test listing models functionality"""
        mock_client = Mock()
        mock_client.list_available_models.return_value = [
            "model1:free",
            "model2:free"
        ]
        mock_client_class.return_value = mock_client
        
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            result = main()
        
        assert result == 0
        output = captured_output.getvalue()
        assert "Available free models:" in output
        assert "model1:free" in output
        assert "model2:free" in output
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm-sweep.py', 'Test prompt', '--model', 'custom-model', '--max-tokens', '256'])
    def test_main_with_options(self, mock_client_class):
        """Test main function with custom options"""
        mock_client = Mock()
        mock_client.query_llm.return_value = {
            "choices": [{"message": {"content": "Custom response"}}]
        }
        mock_client.get_response_text.return_value = "Custom response"
        mock_client_class.return_value = mock_client
        
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            result = main()
        
        assert result == 0
        mock_client.query_llm.assert_called_once_with(
            prompt="Test prompt",
            model="custom-model",
            max_tokens=256,
            seed=None,
            temperature=None
        )
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm-sweep.py', 'Test prompt', '--verbose'])
    def test_main_verbose_mode(self, mock_client_class):
        """Test verbose output mode"""
        mock_client = Mock()
        mock_client.FREE_MODELS = ["test-model:free"]
        mock_client.query_llm.return_value = {
            "choices": [{"message": {"content": "Verbose response"}}],
            "model": "test-model:free",
            "usage": {"total_tokens": 42}
        }
        mock_client.get_response_text.return_value = "Verbose response"
        mock_client_class.return_value = mock_client
        
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            result = main()
        
        assert result == 0
        output = captured_output.getvalue()
        assert "Sending prompt to model:" in output
        assert "test-model:free" in output
        assert "Tokens used: 42" in output
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm-sweep.py', 'Test prompt'])
    def test_main_value_error(self, mock_client_class):
        """Test handling of ValueError in main function"""
        mock_client_class.side_effect = ValueError("Test error")
        
        captured_error = StringIO()
        with patch('sys.stderr', captured_error):
            result = main()
        
        assert result == 1
        assert "Error: Test error" in captured_error.getvalue()
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm-sweep.py', 'Test prompt'])
    def test_main_request_exception(self, mock_client_class):
        """Test handling of RequestException in main function"""
        mock_client = Mock()
        mock_client.query_llm.side_effect = requests.RequestException("Network error")
        mock_client_class.return_value = mock_client
        
        captured_error = StringIO()
        with patch('sys.stderr', captured_error):
            result = main()
        
        assert result == 1
        assert "Network error:" in captured_error.getvalue()
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm-sweep.py', 'Test prompt'])
    def test_main_keyboard_interrupt(self, mock_client_class):
        """Test handling of KeyboardInterrupt in main function"""
        mock_client = Mock()
        mock_client.query_llm.side_effect = KeyboardInterrupt()
        mock_client_class.return_value = mock_client
        
        captured_error = StringIO()
        with patch('sys.stderr', captured_error):
            result = main()
        
        assert result == 1
        assert "Operation cancelled by user" in captured_error.getvalue()
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm-sweep.py', 'Test prompt'])
    def test_main_unexpected_error(self, mock_client_class):
        """Test handling of unexpected errors in main function"""
        mock_client = Mock()
        mock_client.query_llm.side_effect = RuntimeError("Unexpected error")
        mock_client_class.return_value = mock_client
        
        captured_error = StringIO()
        with patch('sys.stderr', captured_error):
            result = main()
        
        assert result == 1
        assert "Unexpected error:" in captured_error.getvalue()
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm_sweep.py', 'Test prompt', '--repetitions', '3'])
    def test_main_multiple_repetitions(self, mock_client_class):
        """Test multiple repetitions functionality"""
        mock_client = Mock()
        mock_client.query_llm.return_value = {
            "choices": [{"message": {"content": "Response"}}],
            "model": "test-model"
        }
        mock_client.get_response_text.return_value = "Response"
        mock_client_class.return_value = mock_client
        
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            result = main()
        
        assert result == 0
        output = captured_output.getvalue()
        
        # Should contain response separators for multiple responses
        assert "=== Response 1 ===" in output
        assert "=== Response 2 ===" in output
        assert "=== Response 3 ===" in output
        
        # Should call query_llm 3 times
        assert mock_client.query_llm.call_count == 3
        assert mock_client.get_response_text.call_count == 3
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm_sweep.py', 'Test prompt', '--repetitions', '2', '--verbose'])
    def test_main_repetitions_verbose(self, mock_client_class):
        """Test repetitions with verbose output"""
        mock_client = Mock()
        mock_client.FREE_MODELS = ["test-model:free"]
        mock_client.query_llm.return_value = {
            "choices": [{"message": {"content": "Verbose response"}}],
            "model": "test-model:free",
            "usage": {"total_tokens": 10}
        }
        mock_client.get_response_text.return_value = "Verbose response"
        mock_client_class.return_value = mock_client
        
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            result = main()
        
        assert result == 0
        output = captured_output.getvalue()
        
        # Should show repetitions info
        assert "Repetitions: 2" in output
        assert "Total repetitions: 2" in output
        assert "Total tokens used: 20" in output
        
        # Should have response separators
        assert "=== Response 1 ===" in output
        assert "=== Response 2 ===" in output
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm_sweep.py', 'Test prompt', '--repetitions', '0'])
    def test_main_invalid_repetitions(self, mock_client_class):
        """Test handling of invalid repetitions count"""
        mock_client_class.return_value = Mock()
        
        captured_error = StringIO()
        with patch('sys.stderr', captured_error):
            result = main()
        
        assert result == 1
        assert "repetitions must be a positive integer" in captured_error.getvalue()
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm_sweep.py', 'Test prompt', '--repetitions', '1'])
    def test_main_single_repetition_no_separator(self, mock_client_class):
        """Test that single repetition doesn't show response separators"""
        mock_client = Mock()
        mock_client.query_llm.return_value = {
            "choices": [{"message": {"content": "Single response"}}],
            "model": "test-model"
        }
        mock_client.get_response_text.return_value = "Single response"
        mock_client_class.return_value = mock_client
        
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            result = main()
        
        assert result == 0
        output = captured_output.getvalue()
        
        # Should not contain response separators for single response
        assert "=== Response 1 ===" not in output
        assert "Single response" in output
        
        # Should call query_llm only once
        assert mock_client.query_llm.call_count == 1
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm_sweep.py', 'Test prompt', '--seed', '42'])
    def test_main_with_seed(self, mock_client_class):
        """Test that seed parameter is passed to the LLM query"""
        mock_client = Mock()
        mock_client.query_llm.return_value = {
            "choices": [{"message": {"content": "Seeded response"}}],
            "model": "test-model"
        }
        mock_client.get_response_text.return_value = "Seeded response"
        mock_client_class.return_value = mock_client
        
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            result = main()
        
        assert result == 0
        assert "Seeded response" in captured_output.getvalue()
        
        # Verify that query_llm was called with the seed parameter
        mock_client.query_llm.assert_called_once_with(
            prompt="Test prompt",
            model=None,
            max_tokens=512,
            seed=42,
            temperature=None
        )
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm_sweep.py', 'Test prompt', '--seed', '123', '--verbose'])
    def test_main_with_seed_verbose(self, mock_client_class):
        """Test that seed is shown in verbose output"""
        mock_client = Mock()
        mock_client.FREE_MODELS = ["test-model:free"]
        mock_client.query_llm.return_value = {
            "choices": [{"message": {"content": "Seeded verbose response"}}],
            "model": "test-model:free",
            "usage": {"total_tokens": 15}
        }
        mock_client.get_response_text.return_value = "Seeded verbose response"
        mock_client_class.return_value = mock_client
        
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            result = main()
        
        assert result == 0
        output = captured_output.getvalue()
        
        # Should show seed in verbose output
        assert "Seed: 123" in output
        assert "Seeded verbose response" in output
        
        # Verify query_llm was called with seed
        mock_client.query_llm.assert_called_once_with(
            prompt="Test prompt",
            model=None,
            max_tokens=512,
            seed=123,
            temperature=None
        )
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm_sweep.py', 'Test prompt', '--repetitions', '2', '--seed', '999'])
    def test_main_seed_with_repetitions(self, mock_client_class):
        """Test that same seed is used for all repetitions"""
        mock_client = Mock()
        mock_client.query_llm.return_value = {
            "choices": [{"message": {"content": "Consistent response"}}],
            "model": "test-model"
        }
        mock_client.get_response_text.return_value = "Consistent response"
        mock_client_class.return_value = mock_client
        
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            result = main()
        
        assert result == 0
        output = captured_output.getvalue()
        
        # Should have multiple responses
        assert "=== Response 1 ===" in output
        assert "=== Response 2 ===" in output
        
        # Verify query_llm was called twice with the same seed
        assert mock_client.query_llm.call_count == 2
        for call in mock_client.query_llm.call_args_list:
            args, kwargs = call
            # Check that seed=999 was passed in kwargs or as positional arg
            assert kwargs.get('seed') == 999 or (len(args) > 3 and args[3] == 999)
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm_sweep.py', 'Test prompt', '--seed', 'invalid'])
    def test_main_invalid_seed(self, mock_client_class):
        """Test handling of invalid seed values"""
        # This should fail at argument parsing level
        with pytest.raises(SystemExit):
            main()
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm_sweep.py', 'Test prompt', '--temperature', '0.2'])
    def test_main_with_temperature(self, mock_client_class):
        """Test that temperature parameter is passed to the LLM query"""
        mock_client = Mock()
        mock_client.query_llm.return_value = {
            "choices": [{"message": {"content": "Temperature controlled response"}}],
            "model": "test-model"
        }
        mock_client.get_response_text.return_value = "Temperature controlled response"
        mock_client_class.return_value = mock_client
        
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            result = main()
        
        assert result == 0
        assert "Temperature controlled response" in captured_output.getvalue()
        
        # Verify that query_llm was called with the temperature parameter
        mock_client.query_llm.assert_called_once_with(
            prompt="Test prompt",
            model=None,
            max_tokens=512,
            seed=None,
            temperature=0.2
        )
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm_sweep.py', 'Test prompt', '--temperature', '0.1', '--verbose'])
    def test_main_with_temperature_verbose(self, mock_client_class):
        """Test that temperature is shown in verbose output"""
        mock_client = Mock()
        mock_client.FREE_MODELS = ["test-model:free"]
        mock_client.query_llm.return_value = {
            "choices": [{"message": {"content": "Low temperature response"}}],
            "model": "test-model:free",
            "usage": {"total_tokens": 20}
        }
        mock_client.get_response_text.return_value = "Low temperature response"
        mock_client_class.return_value = mock_client
        
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            result = main()
        
        assert result == 0
        output = captured_output.getvalue()
        
        # Should show temperature in verbose output
        assert "Temperature: 0.1" in output
        assert "Low temperature response" in output
        
        # Verify query_llm was called with temperature
        mock_client.query_llm.assert_called_once_with(
            prompt="Test prompt",
            model=None,
            max_tokens=512,
            seed=None,
            temperature=0.1
        )
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm_sweep.py', 'Test prompt', '--repetitions', '2', '--temperature', '0.9'])
    def test_main_temperature_with_repetitions(self, mock_client_class):
        """Test that temperature is used for all repetitions"""
        mock_client = Mock()
        mock_client.query_llm.return_value = {
            "choices": [{"message": {"content": "High temperature response"}}],
            "model": "test-model"
        }
        mock_client.get_response_text.return_value = "High temperature response"
        mock_client_class.return_value = mock_client
        
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            result = main()
        
        assert result == 0
        output = captured_output.getvalue()
        
        # Should have multiple responses
        assert "=== Response 1 ===" in output
        assert "=== Response 2 ===" in output
        
        # Verify query_llm was called twice with the same temperature
        assert mock_client.query_llm.call_count == 2
        for call in mock_client.query_llm.call_args_list:
            args, kwargs = call
            # Check that temperature=0.9 was passed in kwargs
            assert kwargs.get('temperature') == 0.9
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm_sweep.py', 'Test prompt', '--temperature', '--0.5'])
    def test_main_invalid_temperature_negative(self, mock_client_class):
        """Test handling of invalid negative temperature values"""
        # This should fail at argument parsing level
        with pytest.raises(SystemExit):
            main()
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm_sweep.py', 'Test prompt', '--temperature', '2.5'])
    def test_main_invalid_temperature_too_high(self, mock_client_class):
        """Test handling of temperature values above valid range"""
        # This should fail at argument parsing level or validation
        with pytest.raises(SystemExit):
            main()
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm_sweep.py', 'Test prompt', '--temperature', 'invalid'])
    def test_main_invalid_temperature_not_float(self, mock_client_class):
        """Test handling of non-numeric temperature values"""
        # This should fail at argument parsing level
        with pytest.raises(SystemExit):
            main()
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm_sweep.py', 'Test prompt', '--temperature', '0.3', '--seed', '123'])
    def test_main_temperature_and_seed_combined(self, mock_client_class):
        """Test that temperature and seed can be used together"""
        mock_client = Mock()
        mock_client.query_llm.return_value = {
            "choices": [{"message": {"content": "Combined parameters response"}}],
            "model": "test-model"
        }
        mock_client.get_response_text.return_value = "Combined parameters response"
        mock_client_class.return_value = mock_client
        
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            result = main()
        
        assert result == 0
        assert "Combined parameters response" in captured_output.getvalue()
        
        # Verify that query_llm was called with both parameters
        mock_client.query_llm.assert_called_once_with(
            prompt="Test prompt",
            model=None,
            max_tokens=512,
            seed=123,
            temperature=0.3
        )


class TestJSONOutput:
    """Test JSON output functionality"""
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm_sweep.py', 'Test prompt', '--output-format', 'json', '--output-file', 'test_output.json'])
    def test_json_output_basic(self, mock_client_class):
        """Test basic JSON output functionality"""
        mock_client = Mock()
        mock_client.query_llm.return_value = {
            "choices": [{"message": {"content": "Test response"}}],
            "model": "test-model",
            "usage": {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5}
        }
        mock_client.get_response_text.return_value = "Test response"
        mock_client_class.return_value = mock_client
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('json.dump') as mock_json_dump:
                result = main()
                
                assert result == 0
                mock_file.assert_called_once_with('test_output.json', 'w')
                mock_json_dump.assert_called_once()
                
                # Check the structure of data passed to json.dump
                written_data = mock_json_dump.call_args[0][0]
                assert 'timestamp' in written_data[0]
                assert 'prompt' in written_data[0]
                assert 'model' in written_data[0]
                assert 'parameters' in written_data[0]
                assert 'response' in written_data[0]
                assert 'performance' in written_data[0]
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm_sweep.py', 'Test prompt', '--output-format', 'json', '--output-file', 'test.json', '--temperature', '0.2', '--seed', '42'])
    def test_json_output_with_parameters(self, mock_client_class):
        """Test JSON output includes all parameters"""
        mock_client = Mock()
        mock_client.query_llm.return_value = {
            "choices": [{"message": {"content": "Seeded response"}}],
            "model": "test-model",
            "usage": {"total_tokens": 15, "prompt_tokens": 8, "completion_tokens": 7}
        }
        mock_client.get_response_text.return_value = "Seeded response"
        mock_client_class.return_value = mock_client
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('json.dump') as mock_json_dump:
                with patch('time.time', return_value=1234567890.123):
                    result = main()
                
                assert result == 0
                written_data = mock_json_dump.call_args[0][0]
                
                # Check parameters are captured
                assert written_data[0]['parameters']['temperature'] == 0.2
                assert written_data[0]['parameters']['seed'] == 42
                assert written_data[0]['parameters']['max_tokens'] == 512
                assert written_data[0]['prompt'] == 'Test prompt'
                assert written_data[0]['response']['text'] == 'Seeded response'
                assert written_data[0]['performance']['total_tokens'] == 15
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm_sweep.py', 'Test prompt', '--output-format', 'json', '--output-file', 'multi.json', '--repetitions', '2'])
    def test_json_output_with_repetitions(self, mock_client_class):
        """Test JSON output with multiple repetitions creates array"""
        mock_client = Mock()
        mock_client.query_llm.return_value = {
            "choices": [{"message": {"content": "Repeated response"}}],
            "model": "test-model",
            "usage": {"total_tokens": 12}
        }
        mock_client.get_response_text.return_value = "Repeated response"
        mock_client_class.return_value = mock_client
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('json.dump') as mock_json_dump:
                result = main()
                
                assert result == 0
                written_data = mock_json_dump.call_args[0][0]
                
                # Should be an array of responses
                assert isinstance(written_data, list)
                assert len(written_data) == 2
                
                # Check each response has repetition info
                assert written_data[0]['parameters']['repetition'] == 1
                assert written_data[1]['parameters']['repetition'] == 2
                assert written_data[0]['parameters']['total_repetitions'] == 2
                assert written_data[1]['parameters']['total_repetitions'] == 2
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm_sweep.py', 'Test prompt', '--output-format', 'json', '--output-file', '/invalid/path/file.json'])
    def test_json_output_file_error(self, mock_client_class):
        """Test handling of file write errors"""
        mock_client = Mock()
        mock_client.query_llm.return_value = {
            "choices": [{"message": {"content": "Test response"}}],
            "model": "test-model"
        }
        mock_client.get_response_text.return_value = "Test response"
        mock_client_class.return_value = mock_client
        
        # Mock file open to raise permission error
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
                result = main()
                
                assert result == 1  # Should return error code
                assert "Error writing JSON output" in mock_stderr.getvalue()
    
    @patch('llm_sweep.OpenRouterClient')  
    @patch('sys.argv', ['llm_sweep.py', 'Test prompt', '--output-format', 'invalid'])
    def test_invalid_output_format(self, mock_client_class):
        """Test handling of invalid output format"""
        # Should fail at argument parsing level
        with pytest.raises(SystemExit):
            main()
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm_sweep.py', 'Test prompt', '--output-file', 'output.json'])
    def test_output_file_without_format(self, mock_client_class):
        """Test that output file without format defaults to JSON"""
        mock_client = Mock()
        mock_client.query_llm.return_value = {
            "choices": [{"message": {"content": "Default format response"}}],
            "model": "test-model"
        }
        mock_client.get_response_text.return_value = "Default format response"
        mock_client_class.return_value = mock_client
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('json.dump') as mock_json_dump:
                result = main()
                
                assert result == 0
                mock_file.assert_called_once_with('output.json', 'w')
                mock_json_dump.assert_called_once()
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm_sweep.py', 'Test prompt', '--output-format', 'json'])
    def test_json_format_without_file(self, mock_client_class):
        """Test JSON format without output file should show error"""
        with pytest.raises(SystemExit):
            main()
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm_sweep.py', 'Test prompt', '--output-file', 'test.json', '--verbose'])
    def test_json_output_verbose_mode(self, mock_client_class):
        """Test JSON output works with verbose mode"""
        mock_client = Mock()
        mock_client.FREE_MODELS = ["test-model:free"]
        mock_client.query_llm.return_value = {
            "choices": [{"message": {"content": "Verbose response"}}],
            "model": "test-model:free",
            "usage": {"total_tokens": 20}
        }
        mock_client.get_response_text.return_value = "Verbose response"
        mock_client_class.return_value = mock_client
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('json.dump') as mock_json_dump:
                captured_output = StringIO()
                with patch('sys.stdout', captured_output):
                    result = main()
                
                assert result == 0
                # Should still output to console in verbose mode
                output = captured_output.getvalue()
                assert "Verbose response" in output
                # And also write to JSON file
                mock_json_dump.assert_called_once()

    def test_json_output_append_to_existing(self):
        """Test appending to an existing JSON file"""
        # Create initial file with one record
        initial_data = [{
            "timestamp": "2024-01-01T00:00:00.000Z",
            "model": "test-model",
            "prompt": "test prompt 1",
            "response": {"text": "test response 1"},
            "parameters": {
                "max_tokens": 100,
                "seed": None,
                "temperature": None
            }
        }]
        
        # Write initial data
        with patch('builtins.open', mock_open()) as mock_file:
            write_json_output(initial_data, "test.json")
            mock_file.assert_called_once_with("test.json", 'w')
        
        # Create new record to append
        new_record = {
            "timestamp": "2024-01-01T00:00:01.000Z",
            "model": "test-model",
            "prompt": "test prompt 2",
            "response": {"text": "test response 2"},
            "parameters": {
                "max_tokens": 100,
                "seed": None,
                "temperature": None
            }
        }
        
        # Append new record
        m = mock_open(read_data=json.dumps(initial_data))
        with patch('builtins.open', m) as mock_file:
            with patch('os.path.exists', return_value=True):
                write_json_output([new_record], "test.json", append=True)
                # Should open for reading and then for writing
                mock_file.assert_any_call("test.json", 'r')
                mock_file.assert_any_call("test.json", 'w')
                # Verify the write call contains both records
                handle = mock_file()
                write_calls = [call for call in handle.write.mock_calls]
                assert len(write_calls) > 0
                written_data = json.loads(''.join(call.args[0] for call in write_calls))
                assert len(written_data) == 2
                assert written_data[0] == initial_data[0]
                assert written_data[1] == new_record

    def test_json_output_append_nonexistent_file(self):
        """Test appending to a nonexistent file creates new file"""
        record = {
            "timestamp": "2024-01-01T00:00:00.000Z",
            "model": "test-model",
            "prompt": "test prompt",
            "response": {"text": "test response"},
            "parameters": {
                "max_tokens": 100,
                "seed": None,
                "temperature": None
            }
        }
        m = mock_open()
        with patch('builtins.open', m) as mock_file:
            with patch('os.path.exists', return_value=False):
                write_json_output([record], "test.json", append=True)
                mock_file.assert_called_with("test.json", 'w')
                handle = mock_file()
                write_calls = [call for call in handle.write.mock_calls]
                assert len(write_calls) > 0
                written_data = json.loads(''.join(call.args[0] for call in write_calls))
                assert len(written_data) == 1
                assert written_data[0] == record

    def test_json_output_always_includes_timestamp(self):
        """Test that all records include a timestamp"""
        record = {
            "model": "test-model",
            "prompt": "test prompt",
            "response": {"text": "test response"},
            "parameters": {
                "max_tokens": 100,
                "seed": None,
                "temperature": None
            }
        }
        m = mock_open()
        with patch('builtins.open', m) as mock_file:
            write_json_output([record], "test.json")
            handle = mock_file()
            write_calls = [call for call in handle.write.mock_calls]
            assert len(write_calls) > 0
            written_data = json.loads(''.join(call.args[0] for call in write_calls))
            assert "timestamp" in written_data[0]
            assert isinstance(written_data[0]["timestamp"], str)
            # Verify timestamp is in ISO format
            assert "T" in written_data[0]["timestamp"]
            assert "Z" in written_data[0]["timestamp"]


class TestPromptFile:
    """Test prompt file reading functionality"""
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm_sweep.py', '--prompt-file', 'test_prompt.txt'])
    def test_prompt_file_basic(self, mock_client_class):
        """Test basic prompt file reading functionality"""
        mock_client = Mock()
        mock_client.query_llm.return_value = {
            "choices": [{"message": {"content": "File response"}}],
            "model": "test-model"
        }
        mock_client.get_response_text.return_value = "File response"
        mock_client_class.return_value = mock_client
        
        # Mock file reading
        with patch('builtins.open', mock_open(read_data="This is a prompt from file")):
            captured_output = StringIO()
            with patch('sys.stdout', captured_output):
                result = main()
            
            assert result == 0
            assert "File response" in captured_output.getvalue()
            
            # Verify prompt from file was used
            mock_client.query_llm.assert_called_once_with(
                prompt="This is a prompt from file",
                model=None,
                max_tokens=512,
                seed=None,
                temperature=None
            )
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm_sweep.py', '--prompt-file', 'nonexistent.txt'])
    def test_prompt_file_not_found(self, mock_client_class):
        """Test handling of non-existent prompt file"""
        # Mock file not found error
        with patch('builtins.open', side_effect=FileNotFoundError("No such file")):
            with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
                result = main()
                
                assert result == 1
                assert "Error reading prompt file" in mock_stderr.getvalue()
                assert "nonexistent.txt" in mock_stderr.getvalue()
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm_sweep.py', '--prompt-file', 'restricted.txt'])
    def test_prompt_file_permission_denied(self, mock_client_class):
        """Test handling of permission denied for prompt file"""
        # Mock permission error
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
                result = main()
                
                assert result == 1
                assert "Error reading prompt file" in mock_stderr.getvalue()
                assert "restricted.txt" in mock_stderr.getvalue()
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm_sweep.py', '--prompt-file', 'empty.txt'])
    def test_prompt_file_empty(self, mock_client_class):
        """Test handling of empty prompt file"""
        # Mock empty file
        with patch('builtins.open', mock_open(read_data="")):
            with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
                result = main()
                
                assert result == 1
                assert "Error: prompt file is empty" in mock_stderr.getvalue()
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm_sweep.py', '--prompt-file', 'whitespace.txt'])
    def test_prompt_file_whitespace_only(self, mock_client_class):
        """Test handling of prompt file with only whitespace"""
        # Mock file with only whitespace
        with patch('builtins.open', mock_open(read_data="   \n\t  \n  ")):
            with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
                result = main()
                
                assert result == 1
                assert "Error: prompt file is empty" in mock_stderr.getvalue()
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm_sweep.py', '--prompt-file', 'multiline.txt'])
    def test_prompt_file_multiline(self, mock_client_class):
        """Test prompt file with multiple lines"""
        mock_client = Mock()
        mock_client.query_llm.return_value = {
            "choices": [{"message": {"content": "Multiline response"}}],
            "model": "test-model"
        }
        mock_client.get_response_text.return_value = "Multiline response"
        mock_client_class.return_value = mock_client
        
        multiline_prompt = "Line 1\nLine 2\nLine 3"
        with patch('builtins.open', mock_open(read_data=multiline_prompt)):
            result = main()
            
            assert result == 0
            # Verify multiline prompt was preserved
            mock_client.query_llm.assert_called_once_with(
                prompt=multiline_prompt,
                model=None,
                max_tokens=512,
                seed=None,
                temperature=None
            )
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm_sweep.py', 'Direct prompt', '--prompt-file', 'file.txt'])
    def test_prompt_and_file_mutual_exclusive(self, mock_client_class):
        """Test that prompt and prompt-file are mutually exclusive"""
        with pytest.raises(SystemExit):
            main()
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm_sweep.py', '--prompt-file', 'test.txt', '--seed', '42', '--temperature', '0.2'])
    def test_prompt_file_with_parameters(self, mock_client_class):
        """Test prompt file works with other parameters"""
        mock_client = Mock()
        mock_client.query_llm.return_value = {
            "choices": [{"message": {"content": "Parameterized response"}}],
            "model": "test-model"
        }
        mock_client.get_response_text.return_value = "Parameterized response"
        mock_client_class.return_value = mock_client
        
        with patch('builtins.open', mock_open(read_data="File prompt with params")):
            result = main()
            
            assert result == 0
            mock_client.query_llm.assert_called_once_with(
                prompt="File prompt with params",
                model=None,
                max_tokens=512,
                seed=42,
                temperature=0.2
            )
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm_sweep.py', '--prompt-file', 'test.txt', '--output-file', 'output.json'])
    def test_prompt_file_with_json_output(self, mock_client_class):
        """Test prompt file works with JSON output"""
        mock_client = Mock()
        mock_client.query_llm.return_value = {
            "choices": [{"message": {"content": "JSON output response"}}],
            "model": "test-model",
            "usage": {"total_tokens": 20}
        }
        mock_client.get_response_text.return_value = "JSON output response"
        mock_client_class.return_value = mock_client
        
        with patch('builtins.open', mock_open(read_data="File prompt for JSON")) as mock_file:
            with patch('json.dump') as mock_json_dump:
                result = main()
                
                assert result == 0
                
                # Verify file was read
                mock_file.assert_any_call('test.txt', 'r', encoding='utf-8')
                
                # Verify JSON output was written
                mock_file.assert_any_call('output.json', 'w')
                mock_json_dump.assert_called_once()
                
                # Verify prompt from file was used in JSON
                written_data = mock_json_dump.call_args[0][0]
                assert written_data[0]['prompt'] == "File prompt for JSON"
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm_sweep.py', '--prompt-file', 'large.txt'])
    def test_prompt_file_large(self, mock_client_class):
        """Test prompt file with large content"""
        mock_client = Mock()
        mock_client.query_llm.return_value = {
            "choices": [{"message": {"content": "Large file response"}}],
            "model": "test-model"
        }
        mock_client.get_response_text.return_value = "Large file response"
        mock_client_class.return_value = mock_client
        
        # Create a large prompt (simulate large file)
        large_prompt = "This is a very long prompt. " * 1000
        with patch('builtins.open', mock_open(read_data=large_prompt)):
            result = main()
            
            assert result == 0
            mock_client.query_llm.assert_called_once_with(
                prompt=large_prompt,
                model=None,
                max_tokens=512,
                seed=None,
                temperature=None
            )


class TestErrorHandling:
    """Test improved error handling functionality"""
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm_sweep.py', 'Test prompt', '--model', 'invalid-model-name'])
    def test_invalid_model_error_message(self, mock_client_class):
        """Test that invalid model names produce helpful error messages"""
        mock_client = Mock()
        
        # Simulate a ValueError that would result from 400 HTTP error with model-related error message
        mock_client.query_llm.side_effect = ValueError("Invalid model specification: Model 'invalid-model-name' not found")
        mock_client.list_available_models.return_value = [
            "meta-llama/llama-3.1-8b-instruct:free",
            "microsoft/phi-3-mini-128k-instruct:free"
        ]
        mock_client_class.return_value = mock_client
        
        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            result = main()
            
            assert result == 1
            error_output = mock_stderr.getvalue()
            assert "Invalid model specification" in error_output
            assert "invalid-model-name" in error_output
            assert "Available free models:" in error_output
            assert "meta-llama/llama-3.1-8b-instruct:free" in error_output
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm_sweep.py', 'Test prompt', '--model', 'nonexistent-model'])
    def test_model_not_in_free_list_error(self, mock_client_class):
        """Test error handling when model is not in the free models list"""
        mock_client = Mock()
        
        # Simulate a ValueError that would result from a 400 error for an invalid model
        mock_client.query_llm.side_effect = ValueError("Model 'nonexistent-model' may not be available. Use --list-models to see available options.")
        mock_client.FREE_MODELS = ["meta-llama/llama-3.1-8b-instruct:free"]
        mock_client.list_available_models.return_value = ["meta-llama/llama-3.1-8b-instruct:free"]
        mock_client_class.return_value = mock_client
        
        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            result = main()
            
            assert result == 1
            error_output = mock_stderr.getvalue()
            assert "may not be available" in error_output
            assert "nonexistent-model" in error_output
            assert "--list-models" in error_output
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm_sweep.py', 'Test prompt', '--model', 'bad-model'])
    def test_generic_400_error_with_helpful_message(self, mock_client_class):
        """Test that 400 errors without parseable details still provide help"""
        mock_client = Mock()
        
        # Simulate a ValueError that would result from a 400 error for an invalid model  
        mock_client.query_llm.side_effect = ValueError("Model 'bad-model' may not be available. Use --list-models to see available options.")
        mock_client.FREE_MODELS = ["meta-llama/llama-3.1-8b-instruct:free"]
        mock_client.list_available_models.return_value = ["meta-llama/llama-3.1-8b-instruct:free"]
        mock_client_class.return_value = mock_client
        
        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            result = main()
            
            assert result == 1
            error_output = mock_stderr.getvalue()
            assert "may not be available" in error_output
            assert "bad-model" in error_output
    
    @patch('llm_sweep.OpenRouterClient')
    @patch('sys.argv', ['llm_sweep.py', 'Test prompt', '--model', 'typo-model'])
    def test_error_fallback_when_client_creation_fails(self, mock_client_class):
        """Test that error handling works even when client creation fails"""
        # First client creation succeeds for the main try block
        mock_client = Mock()
        mock_client.query_llm.side_effect = ValueError("Invalid model specification: Invalid model typo-model")
        
        # But when we try to create a temp client for error handling, it fails
        mock_client_class.side_effect = [mock_client, Exception("API key error")]
        
        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            result = main()
            
            assert result == 1
            error_output = mock_stderr.getvalue()
            assert "Invalid model specification" in error_output
            assert "typo-model" in error_output
            # Should still show available models from fallback
            assert "meta-llama/llama-3.1-8b-instruct:free" in error_output


class TestIntegration:
    """Integration tests that test the full workflow"""
    
    @patch('requests.post')
    def test_full_workflow_success(self, mock_post):
        """Test the complete workflow from initialization to response"""
        # Mock successful API response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Hello! This is a test response from the AI model."
                    }
                }
            ],
            "model": "meta-llama/llama-3.1-8b-instruct:free",
            "usage": {
                "total_tokens": 20,
                "prompt_tokens": 5,
                "completion_tokens": 15
            }
        }
        mock_post.return_value = mock_response
        
        # Test the complete workflow
        client = OpenRouterClient(api_key="test-key")
        response_data = client.query_llm("Hello, how are you?")
        response_text = client.get_response_text(response_data)
        
        assert response_text == "Hello! This is a test response from the AI model."
        assert response_data["usage"]["total_tokens"] == 20
        
        # Verify the API call was made correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        assert payload['messages'][0]['content'] == "Hello, how are you?"


def test_response_record_always_includes_parameters():
    """Test that response record always includes seed and temperature parameters"""
    args = argparse.Namespace(
        model="test-model",
        max_tokens=100,
        seed=None,
        temperature=None
    )
    response_data = {
        "choices": [{"message": {"content": "test response"}}]
    }
    record = create_response_record(args, response_data, "test response", 0, 1, "test prompt")
    
    # Verify parameters are always included
    assert "parameters" in record
    assert "seed" in record["parameters"]
    assert "temperature" in record["parameters"]
    assert record["parameters"]["seed"] is None
    assert record["parameters"]["temperature"] is None

def test_response_record_with_specified_parameters():
    """Test that response record includes specified seed and temperature values"""
    args = argparse.Namespace(
        model="test-model",
        max_tokens=100,
        seed=42,
        temperature=0.8
    )
    response_data = {
        "choices": [{"message": {"content": "test response"}}]
    }
    record = create_response_record(args, response_data, "test response", 0, 1, "test prompt")
    
    # Verify parameters are included with specified values
    assert "parameters" in record
    assert record["parameters"]["seed"] == 42
    assert record["parameters"]["temperature"] == 0.8


if __name__ == "__main__":
    pytest.main([__file__]) 
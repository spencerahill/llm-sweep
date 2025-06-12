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
from unittest.mock import Mock, patch, MagicMock
from llm_sweep import OpenRouterClient, main
import sys
from io import StringIO


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
            max_tokens=512
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
            max_tokens=256
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


if __name__ == "__main__":
    pytest.main([__file__]) 
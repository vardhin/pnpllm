import os
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Iterator, Optional

# Load environment variables
load_dotenv()

class GeminiAPI:
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        """
        Initialize Gemini API client
        
        Args:
            model_name: The Gemini model to use (default: "gemini-1.5-flash")
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate a complete response from Gemini API
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters for generation
            
        Returns:
            Complete response as string
        """
        try:
            response = self.model.generate_content(prompt, **kwargs)
            return response.text
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")
    
    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """
        Generate a streaming response from Gemini API
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters for generation
            
        Yields:
            Chunks of response text
        """
        try:
            response = self.model.generate_content(prompt, stream=True, **kwargs)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            raise Exception(f"Error generating streaming response: {str(e)}")

def list_available_models():
    """
    List all available Gemini models
    
    Returns:
        List of available model names
    """
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        models = genai.list_models()
        
        available_models = []
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                available_models.append(model.name)
        
        return available_models
    except Exception as e:
        raise Exception(f"Error listing models: {str(e)}")

# Convenience functions
def get_gemini_response(prompt: str, model_name: str = "gemini-1.5-flash", **kwargs) -> str:
    """
    Get complete response from Gemini API
    
    Args:
        prompt: The input prompt
        model_name: The Gemini model to use
        **kwargs: Additional parameters for generation
        
    Returns:
        Complete response as string
    """
    gemini = GeminiAPI(model_name)
    return gemini.generate_response(prompt, **kwargs)

def get_gemini_stream(prompt: str, model_name: str = "gemini-1.5-flash", **kwargs) -> Iterator[str]:
    """
    Get streaming response from Gemini API
    
    Args:
        prompt: The input prompt
        model_name: The Gemini model to use
        **kwargs: Additional parameters for generation
        
    Yields:
        Chunks of response text
    """
    gemini = GeminiAPI(model_name)
    return gemini.generate_stream(prompt, **kwargs)

# Example usage
if __name__ == "__main__":
    # List available models first
    try:
        print("Available models:")
        models = list_available_models()
        for model in models:
            print(f"  - {model}")
        print("\n" + "="*50 + "\n")
    except Exception as e:
        print(f"Error listing models: {e}")
    
    # Example 1: Complete response
    try:
        response = get_gemini_response("Tell me a short joke")
        print("Complete response:")
        print(response)
        print("\n" + "="*50 + "\n")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Streaming response
    try:
        print("Streaming response:")
        for chunk in get_gemini_stream("Write a short story about a robot"):
            print(chunk, end='', flush=True)
        print("\n")
    except Exception as e:
        print(f"Error: {e}")
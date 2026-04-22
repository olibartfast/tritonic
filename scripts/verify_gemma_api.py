#!/usr/bin/env python3
"""
Pre-flight check for ChatBackend testing with Gemma models.

This script verifies that:
1. OpenRouter API is accessible
2. API key is valid
3. Gemma models are available
4. Basic inference works

Usage:
    export OPENROUTER_API_KEY="sk-or-v1-..."
    python3 scripts/verify_gemma_api.py
"""

import os
import sys
import json
import urllib.request
import urllib.error

def check_env():
    """Check if required environment variables are set."""
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ Error: OPENROUTER_API_KEY environment variable is not set")
        print("\nPlease set your API key:")
        print("  export OPENROUTER_API_KEY='sk-or-v1-...'")
        print("\nGet a free API key at: https://openrouter.ai/keys")
        return None

    print(f"✅ API key found: {api_key[:20]}...")
    return api_key

def test_api_connection(api_key, model=None):
    """Test basic API connectivity and model availability."""
    endpoint = "https://openrouter.ai/api/v1/chat/completions"
    if model is None:
        model = "google/gemma-4-31b:free"  # Default to Gemma 4 31B (free)

    print(f"\n🔄 Testing API connection to OpenRouter...")
    print(f"   Endpoint: {endpoint}")
    print(f"   Model: {model}")

    # Prepare request
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Say 'Hello from Gemma!' and nothing else."}
        ],
        "max_tokens": 20,
        "temperature": 0.7
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    try:
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(data).encode('utf-8'),
            headers=headers,
            method='POST'
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            response_data = json.loads(response.read().decode('utf-8'))

            if 'choices' in response_data and len(response_data['choices']) > 0:
                content = response_data['choices'][0]['message']['content']
                print(f"\n✅ API test successful!")
                print(f"   Response: {content}")

                # Check usage info
                if 'usage' in response_data:
                    usage = response_data['usage']
                    print(f"\n📊 Token usage:")
                    print(f"   Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
                    print(f"   Completion tokens: {usage.get('completion_tokens', 'N/A')}")
                    print(f"   Total tokens: {usage.get('total_tokens', 'N/A')}")

                return True
            else:
                print(f"❌ Unexpected response format: {response_data}")
                return False

    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        print(f"\n❌ HTTP Error {e.code}: {e.reason}")
        print(f"   Response: {error_body}")

        try:
            error_json = json.loads(error_body)
            if 'error' in error_json:
                print(f"\n   API Error: {error_json['error']}")
        except:
            pass

        if e.code == 401:
            print("\n💡 Hint: Your API key may be invalid. Get a new one at https://openrouter.ai/keys")
        elif e.code == 429:
            print("\n💡 Hint: Rate limit exceeded. Wait a few minutes and try again.")

        return False

    except urllib.error.URLError as e:
        print(f"\n❌ Connection error: {e.reason}")
        print("   Check your internet connection")
        return False

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

def list_available_models():
    """List some available Gemma models on OpenRouter."""
    print("\n📋 Available Gemma models on OpenRouter:")
    print("\n   Gemma 4 (Latest - Multimodal with 256K context):")
    print("   • google/gemma-4-31b:free          - 31B dense model (FREE, recommended)")
    print("   • google/gemma-4-31b               - 31B dense model (paid tier)")
    print("   • google/gemma-4-26b-a4b:free      - 26B MoE, 3.8B active (FREE)")
    print("   • google/gemma-4-26b-a4b           - 26B MoE, 3.8B active (paid tier)")
    print("\n   Gemma 2 (Previous generation):")
    print("   • google/gemma-2-9b-it             - 9B Instruct")
    print("   • google/gemma-2-27b-it            - 27B Instruct")
    print("\n   See all models at: https://openrouter.ai/models")
    print("\n   💡 Gemma 4 features:")
    print("      - Multimodal (text, images, video)")
    print("      - 256K token context window")
    print("      - Function calling support")
    print("      - Thinking/reasoning mode")

def main():
    """Main verification flow."""
    print("=" * 60)
    print("ChatBackend API Verification - Gemma Models")
    print("=" * 60)

    # Check environment
    api_key = check_env()
    if not api_key:
        return 1

    # Test API connection
    if not test_api_connection(api_key):
        print("\n❌ API verification failed")
        return 1

    # Show available models
    list_available_models()

    # Success
    print("\n" + "=" * 60)
    print("✅ All checks passed! You can now run:")
    print("   ./scripts/test_chat_backend_gemma.sh")
    print("=" * 60)

    return 0

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Test script for the Data Analyst Agent API
"""

import requests
import json
import tempfile
import os


def test_api_endpoint(base_url="http://localhost:8000"):
    """Test the API with a sample question"""

    # Create a sample questions file
    questions_content = """
Scrape the list of highest grossing films from Wikipedia. It is at the URL:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array of strings containing the answer:
1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.

Return as a base-64 encoded data URI, `"data:image/png;base64,iVBORw0KG..."` under 100,000 bytes.
"""

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(questions_content)
        questions_file = f.name

    try:
        # Test the API
        with open(questions_file, "rb") as f:
            files = [("files", ("questions.txt", f, "text/plain"))]

            print(f"Testing API at {base_url}/api/")
            response = requests.post(f"{base_url}/api/", files=files, timeout=180)

            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text[:500]}...")

            if response.status_code == 200:
                try:
                    result = response.json()
                    print("✅ API test successful!")
                    print(f"Result type: {type(result)}")
                    if isinstance(result, list):
                        print(f"Array length: {len(result)}")
                        for i, item in enumerate(result):
                            if isinstance(item, str) and item.startswith("data:image"):
                                print(f"Item {i}: Image data ({len(item)} characters)")
                            else:
                                print(f"Item {i}: {item}")
                    else:
                        print(f"Result: {result}")
                except json.JSONDecodeError:
                    print("❌ Response is not valid JSON")
            else:
                print(f"❌ API test failed: {response.text}")

    finally:
        # Clean up
        os.unlink(questions_file)


def test_health_endpoint(base_url="http://localhost:8000"):
    """Test the health endpoint"""
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health check status: {response.status_code}")
        print(f"Health response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False


if __name__ == "__main__":
    import sys

    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

    print("Testing Data Analyst Agent API")
    print("=" * 40)

    # Test health first
    if test_health_endpoint(base_url):
        print("✅ Health check passed")

        # Test main functionality
        test_api_endpoint(base_url)
    else:
        print("❌ Health check failed - API may not be running")

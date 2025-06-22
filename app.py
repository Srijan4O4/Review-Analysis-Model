import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

def generate_content(prompt: str, text: str, number: float) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Please set the GEMINI_API_KEY environment variable in your environment or .env file."
        )

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        f"?key={api_key}"
    )
    headers = {"Content-Type": "application/json"}
    parts = [{"text": prompt}, {"text": text}]
    if number is not None:
        parts.append({"text": str(number)})

    payload = {
        "contents": [
            {"parts": parts}
        ]
    }

    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()

    try:
        candidate = data["candidates"][0]
    except (KeyError, IndexError):
        raise ValueError(f"Unexpected API response structure: {json.dumps(data, indent=2)}")

    def join_parts(parts_list):
        return "".join(part.get("text", "") for part in parts_list).strip()

    if "parts" in candidate and isinstance(candidate["parts"], list):
        return join_parts(candidate["parts"])

    content = candidate.get("content")
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, dict) and "parts" in content and isinstance(content["parts"], list):
        return join_parts(content["parts"])

    return json.dumps(candidate, ensure_ascii=False)

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Gemini AI Review Verifier")
    parser.add_argument(
        "-t", "--text", required=True,
        help="The review text input."
    )
    parser.add_argument(
        "-n", "--number", type=float, required=True,
        help="The numeric rating associated with the review."
    )

    args = parser.parse_args()

    # Default prompt embedded here
    default_prompt = (
        "You are an expert AI content detector for a marketplace platform. Given a product review and its rating, determine whether the review is AI-generated or written by a human. Also give Confidence Score (0-100)"
    )

    try:
        result = generate_content(default_prompt, args.text, args.number)
        print("--- Generated Output ---")
        print(result)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
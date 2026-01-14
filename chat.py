#!/usr/bin/env python3
"""Command-line chat interface for DeepSeek R1"""

import requests
import sys

API_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "deepseek-ai/DeepSeek-R1"

def chat(messages, max_tokens=2048):
    response = requests.post(API_URL, json={
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7
    })
    return response.json()["choices"][0]["message"]["content"]

def main():
    print("DeepSeek R1 Chat (type 'quit' to exit, 'clear' to reset)")
    print("-" * 50)

    messages = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            messages = []
            print("Conversation cleared.")
            continue

        messages.append({"role": "user", "content": user_input})

        try:
            response = chat(messages)
            print(f"\nDeepSeek R1: {response}")
            messages.append({"role": "assistant", "content": response})
        except Exception as e:
            print(f"\nError: {e}")
            messages.pop()  # Remove failed message

if __name__ == "__main__":
    main()

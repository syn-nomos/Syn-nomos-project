import os
import json
import requests
import time
import random
import streamlit as st
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

class LLMJudge:
    """
    LLM Judge Client for Legal-Hydra (via OpenRouter).
    Uses 'DeepSeek R1' (Reasoning) or 'Claude 3.5 Sonnet' for high-level conflict resolution.
    """
    def __init__(self, model="deepseek/deepseek-r1", api_key=None):
        # Retrieve API Key from various sources
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            try:
                self.api_key = st.secrets["OPENROUTER_API_KEY"]
            except:
                pass

        self.model = model
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Headers specifically for OpenRouter
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://legal-hydra.local", 
            "X-Title": "Legal-Hydra Research"
        }

    def _call_llm(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        """General Helper for Smart R1 API Calls"""
        if not self.api_key:
            return None
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1, # Low temp for rigorous logic
            "max_tokens": 8192,
        }

        MAX_RETRIES = 5
        BASE_DELAY = 3 # Seconds

        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=120)
                
                if response.status_code == 200:
                    content = response.json()['choices'][0]['message']['content']
                    # Clean up Thinking Tags from DeepSeek R1 (<think>...</think>)
                    if "<think>" in content:
                        content = content.split("</think>")[-1].strip()
                    
                    # Clean up Markdown JSON blocks
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0].strip()
                        
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        # Fallback 1: Regex Extraction if Extra Data exists
                        import re
                        match = re.search(r'(\{.*\})', content, re.DOTALL)
                        if match:
                            try:
                                return json.loads(match.group(1))
                            except:
                                pass
                        
                        # Fallback 2: Repair Truncated JSON
                        try:
                            last_brace = content.rfind('}')
                            last_bracket = content.rfind(']')
                            cut_off = max(last_brace, last_bracket)
                            
                            if cut_off != -1:
                                truncated = content[:cut_off+1]
                                num_braces = truncated.count('{') - truncated.count('}')
                                num_brackets = truncated.count('[') - truncated.count(']')
                                repaired = truncated + (']' * max(0, num_brackets)) + ('}' * max(0, num_braces))
                                repaired_json = json.loads(repaired)
                                return repaired_json
                        except Exception:
                            pass

                        print(f"LLM JSON Error. Content: {content[:100]}...")
                        time.sleep(BASE_DELAY)
                        continue
                        
                elif response.status_code == 429: # Rate Limit
                    wait_time = BASE_DELAY * (2 ** attempt) + (random.random() * 2)
                    time.sleep(wait_time)
                    continue

                elif response.status_code >= 500: # Server Error
                    wait_time = BASE_DELAY * (2 ** attempt)
                    time.sleep(wait_time)
                    continue
                
                else:
                    return None

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                wait_time = BASE_DELAY * (2 ** attempt)
                time.sleep(wait_time)
                continue
            except Exception as e:
                return None
        
        return None

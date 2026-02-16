#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Vocabulary Generator (EuroLex & MIMIC)
==============================================
This tool generates textual descriptions for concepts from either the EuroLex (Legal)
or MIMIC-III (Clinical) datasets using various LLMs via the OpenRouter API.

It handles dataset-specific logic (loading, prompting) and model selection uniformly.
"""

import json
import requests
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import time
import sys
import argparse

# =============================================================================
# Configuration & Constants
# =============================================================================

MODELS = {
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "ministral-8b": "mistralai/ministral-8b",
    "mistral-large": "mistralai/mistral-large-2512",
    "mixtral-8x7b": "mistralai/mixtral-8x7b-instruct"
}

# Paths relative to this script (inside /apps)
APP_DIR = Path(__file__).parent
ROOT_DIR = APP_DIR.parent      # Keep for .env loading
DATA_DIR = APP_DIR / "data"    # New local data directory

DATASET_PATHS = {
    "eurolex": DATA_DIR / "eurovoc_concepts_enriched.json",
    "mimic": DATA_DIR / "icd_codes_flattened.json",
    "hellasvoc": DATA_DIR / "hellasvoc_hierarchy.json"
}

# =============================================================================
# Setup Environment
# =============================================================================

def load_environment():
    """Loads environment variables from .env file in project root."""
    env_path = ROOT_DIR / '.env'
    if env_path.exists():
        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, val = line.split('=', 1)
                        if key.strip() not in os.environ:
                            os.environ[key.strip()] = val.strip().strip('"').strip("'")
            print(f"📂 Loaded configuration from: {env_path}")
        except Exception as e:
            print(f"⚠️  Error loading .env: {e}")
    else:
        print(f"⚠️  .env file not found at {env_path}")

load_environment()

# =============================================================================
# Generator Class
# =============================================================================

class UnifiedGenerator:
    def __init__(self, dataset_type: str, model_alias: str, max_tokens: int = 800, use_hierarchy: bool = True):
        self.dataset_type = dataset_type
        self.model_alias = model_alias
        self.model_id = MODELS.get(model_alias, model_alias) # Allow raw ID if needed
        self.max_tokens = max_tokens
        self.use_hierarchy = use_hierarchy
        
        # API Setup
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            print("❌ Error: OPENROUTER_API_KEY not found in environment")
            sys.exit(1)
            
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost:3000",
            "X-Title": "Vocabulary Generator"
        }
        
        # Output Setup
        safe_model_name = self.model_alias.replace("/", "_").replace("-", "_")
        folder_suffix = "" if self.use_hierarchy else "_no_hierarchy"
        self.output_dir = Path(__file__).parent / f"{dataset_type}_descriptions_{safe_model_name}{folder_suffix}"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.stats_file = self.output_dir / "../generation_stats.json" # Shared stats or per folder? Let's do per folder for safety
        self.stats_file = self.output_dir / f"stats_{safe_model_name}.json"
        
        self.stats = self._load_stats()
        self.concepts = self._load_data()

    def _load_stats(self):
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except: pass
        return {
            "total_processed": 0,
            "start_time": datetime.now().isoformat(),
            "dataset": self.dataset_type,
            "model": self.model_alias
        }

    def _save_stats(self):
        self.stats["last_update"] = datetime.now().isoformat()
        with open(self.stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2)

    def _load_data(self) -> Dict:
        path = DATASET_PATHS.get(self.dataset_type)
        if not path or not path.exists():
            print(f"❌ Dataset file not found: {path}")
            sys.exit(1)

        print(f"📁 Loading {self.dataset_type} concepts from: {path}")
        concepts = {}
        
        try:
            if self.dataset_type == "hellasvoc":
                self._load_hellasvoc_recursive(path, concepts)
            else:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if self.dataset_type == "eurolex":
                    # Handle EuroLex structure (enriched vs jsonl logic)
                    raw_list = data if isinstance(data, list) else data.values() if isinstance(data, dict) else []
                    # Convert list to dict keyed by ID
                    for item in raw_list:
                        # Filter: Only keep items marked as originating from EuroLex/EurLex
                        if self._is_eurlex_valid(item):
                            c_id = str(item.get('id'))
                            concepts[c_id] = item
                            
                elif self.dataset_type == "mimic":
                    # Handle MIMIC flattened structure
                    if isinstance(data, list):
                        for item in data:
                            c_id = str(item.get('code'))
                            if c_id and item.get('title'):
                                concepts[c_id] = item
                            
            print(f"📊 Ready to process {len(concepts)} valid concepts for {self.dataset_type}")
            return concepts
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            sys.exit(1)

    def _load_hellasvoc_recursive(self, path, concepts_dict):
        """Special loader for HellasVoc hierarchical files."""
        def traverse(node, path_stack):
            c_id = str(node.get('id'))
            name = node.get('name')
            
            # Construct hierarchy string from ancestors
            current_path_str = " > ".join(path_stack)
            
            concept_data = {
                'id': c_id,
                'title': name,
                'definition': node.get('definition', ''),
                'category_name': path_stack[0] if path_stack else name,
                'hierarchical_paths': [current_path_str] if current_path_str else [], 
                'children_ids': [c.get('id') for c in node.get('children', [])],
                'dataset_source': 'hellasvoc'
            }
            concepts_dict[c_id] = concept_data
            
            # Recurse
            children = node.get('children', [])
            for child in children:
                traverse(child, path_stack + [name])

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    root_node = json.loads(line)
                    traverse(root_node, [])
                except json.JSONDecodeError as e:
                    print(f"Error parsing HellasVoc line: {e}")

    def _is_eurlex_valid(self, data: dict) -> bool:
        """Checks if a concept is a valid EuroLex source concept."""
        return bool(
            data.get("eurlex_source")
            or (isinstance(data.get("eurlex"), dict) and data["eurlex"].get("eurlex_source"))
            or data.get("eurlex_found")
        )

    def create_prompt(self, concept: Dict) -> str:
        """Generates dataset-specific prompt."""
        title = concept.get('title', 'Unknown')
        
        # Prepare hierarchy string conditionally
        hierarchy_text = ""

        if self.dataset_type == "eurolex":
            # Extract hierarchy for EuroLex
            eurlex = concept.get('eurlex') if isinstance(concept.get('eurlex'), dict) else concept
            title = eurlex.get('title', title)
            
            if self.use_hierarchy:
                paths = eurlex.get("hierarchical_paths", [])
                if paths:
                    p = paths[0]
                    h_val = p.get("full_path_string", "") if isinstance(p, dict) else str(p)
                    h_val = h_val.replace(" > ", " → ")
                    hierarchy_text = f"WITH HIERARCHY: The hierarchy of the concept is: {h_val}"

            return f"""You are an expert legal terminology specialist. Your task is to create a clear, precise description for a legal concept.

Write a description (4-5 sentences) of the concept '{title}' that explains what this legal topic covers. 
{hierarchy_text} 

IMPORTANT INSTRUCTIONS:
- Make sure to use the title of the concept in the description.
- Provide ONLY the description text, no titles, headers, or formatting
- Do not start with the concept name or any bold text like "**{title}**:"
- Write in a direct, informative style as if explaining to a legal professional
- Focus on what this concept encompasses in legal practice
- The description should be approximately 15 sentences long.

Description:"""

        elif self.dataset_type == "mimic":
            # Extract hierarchy for MIMIC
            if self.use_hierarchy:
                h_val = concept.get("path", "").replace(" > ", " → ")
                if h_val:
                    hierarchy_text = f"WITH HIERARCHY: The hierarchy of the concept is: {h_val}"
            
            return f"""You are an expert in clinical informatics and medical coding. Your task is to create a clear, precise description for an ICD-9 clinical concept.

Write a description (4-5 sentences maximum) of the concept '{title}' that explains what this clinical topic covers.
The title of the concept is: {title}
{hierarchy_text}

IMPORTANT INSTRUCTIONS:
- Make sure to use the title of the concept in the description.
- Provide ONLY the description text, no titles, headers, or formatting
- Do not start with the concept name or any bold text like "**{title}**:"
- Write in a direct, informative style as if explaining to a clinical professional
- Focus on what this concept encompasses in clinical practice

Description:"""

        elif self.dataset_type == "hellasvoc":
            # Extract hierarchy for HellasVoc
            if self.use_hierarchy:
                paths = concept.get("hierarchical_paths", [])
                if paths:
                    h_val = paths[0].replace(" > ", " → ")
                    hierarchy_text = f"Η ιεραρχική θέση της έννοιας είναι: {h_val}"

            return f"""Είσαι ειδικός επιστήμονας νομικής, με γνώση της ελληνικής νομικής ορολογίας. Η αποστολή σου είναι να δημιουργήσεις μια σαφή και ακριβή περιγραφή για μια νομική έννοια.

Συντάξε μια περιγραφή (μέγιστο 4-5 προτάσεις) για την έννοια '{title}', εξηγώντας τι καλύπτει το συγκεκριμένο νομικό αντικείμενο. 
{hierarchy_text} 

ΣΗΜΑΝΤΙΚΕΣ ΟΔΗΓΙΕΣ:
- Φροντίστε να συμπεριλάβεις τον τίτλο της έννοιας μέσα στην περιγραφή.
- Παρέχετε ΜΟΝΟ το κείμενο της περιγραφής, χωρίς τίτλους, επικεφαλίδες ή μορφοποίηση.
- Μην ξεκινάτε την πρόταση με το όνομα της έννοιας ή με έντονη γραφή όπως "**{title}**:"
- Γράψτε με άμεσο και ενημερωτικό ύφος, απευθυνόμενοι σε επαγγελματίες νομικούς.
- Εστιάστε στο περιεχόμενο και την εφαρμογή της έννοιας στη νομική πρακτική.
- Η γλώσσα που θα χρησιμοποιήσεις πρέπει να ειναι ελληνικά.

Περιγραφή:"""
            
        return ""

    def generate(self, overwrite=False):
        print(f"🚀 Starting generation using {self.model_alias} ({self.model_id})")
        print(f"📁 Output directory: {self.output_dir}")
        if overwrite:
            print("⚠️  OVERWRITE MODE ENABLED")

        processed_count = 0
        ids = list(self.concepts.keys())
        
        for i, cid in enumerate(ids, 1):
            concept = self.concepts[cid]
            
            # Filename sanitization
            title = concept.get('title', 'concept')
            # Handle EuroLex title nested in 'eurlex' sometimes
            if self.dataset_type == 'eurolex' and isinstance(concept.get('eurlex'), dict):
                title = concept['eurlex'].get('title', title)
                
            safe_title = "".join(x for x in title if x.isalnum() or x in (' ', '-', '_')).strip()
            safe_title = safe_title.replace(' ', '_')[:50]
            filename = f"concept_{cid}_{safe_title}.txt"
            filepath = self.output_dir / filename
            
            if filepath.exists() and not overwrite:
                continue

            print(f"[{i}/{len(ids)}] {cid}: {title[:40]}...")
            
            prompt = self.create_prompt(concept)
            
             # Debug Print for Hierarchy
            if "WITH HIERARCHY" in prompt or "Η ιεραρχική θέση" in prompt:
                 # Extract hierarchy part for display
                 try:
                     h_line = [L for L in prompt.split('\n') if "HIERARCHY" in L or "ιεραρχική" in L][0]
                     h_val = h_line.split(":", 1)[1].strip()
                     print(f"     ↳ Path: {h_val[:80]}..." if len(h_val) > 80 else f"     ↳ Path: {h_val}")
                 except: pass

            # API Call with Retry
            for attempt in range(3):
                try:
                    start_time = time.time()
                    response = requests.post(
                        f"{self.base_url}/chat/completions",
                        headers=self.headers,
                        json={
                            "model": self.model_id,
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": self.max_tokens,
                            "temperature": 0.6
                        },
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        content = response.json()['choices'][0]['message']['content']
                        
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(f"CONCEPT_ID: {cid}\n")
                            f.write(f"TITLE: {title}\n")
                            f.write(f"DATASET: {self.dataset_type}\n")
                            f.write(f"MODEL: {self.model_id}\n")
                            if "WITH HIERARCHY" in prompt or "Η ιεραρχική θέση" in prompt:
                                try:
                                    h_line = [L for L in prompt.split('\n') if "HIERARCHY" in L or "ιεραρχική" in L][0]
                                    h_write = h_line.split(":", 1)[1].strip()
                                    f.write(f"HIERARCHY: {h_write}\n")
                                except: pass
                            f.write("-" * 40 + "\n")
                            f.write(content)
                            
                        self.stats["total_processed"] += 1
                        self._save_stats()
                        processed_count += 1
                        print(f"   ✅ Saved ({time.time() - start_time:.1f}s)")
                        break
                    else:
                        print(f"   ❌ API Error: {response.status_code}")
                        time.sleep(5 * (attempt + 1))
                        
                except Exception as e:
                    print(f"   ❌ Exception: {e}")
                    time.sleep(5 * (attempt + 1))
        
        print(f"\n✨ Generation complete! Processed {processed_count} concepts.")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate vocabulary descriptions using LLMs.")
    
    parser.add_argument(
        "--dataset", 
        required=True, 
        choices=["eurolex", "mimic", "hellasvoc"],
        help="Target dataset to process."
    )
    
    parser.add_argument(
        "--model", 
        required=True, 
        choices=list(MODELS.keys()),
        help=f"LLM model to use. Options: {', '.join(MODELS.keys())}"
    )
    
    parser.add_argument(
        "--overwrite", 
        action="store_true", 
        help="Regenerate descriptions even if files exist."
    )
    
    parser.add_argument(
        "--no-hierarchy",
        action="store_true",
        help="Disable hierarchy context in prompts."
    )
    
    args = parser.parse_args()
    
    print("-" * 60)
    print(" 🧠 VOCABULARY GENERATOR ")
    print(f" 📂 Dataset:   {args.dataset}")
    print(f" 🤖 Model:     {args.model}")
    print(f" 🌲 Hierarchy: {'OFF' if args.no_hierarchy else 'ON'}")
    print("-" * 60)
    
    generator = UnifiedGenerator(args.dataset, args.model, use_hierarchy=not args.no_hierarchy)
    generator.generate(overwrite=args.overwrite)

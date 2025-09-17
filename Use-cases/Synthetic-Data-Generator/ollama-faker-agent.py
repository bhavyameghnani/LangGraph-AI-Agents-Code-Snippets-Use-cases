"""
CSV PII/HIPAA Masking + Synthetic Data Generator using Ollama (ChatGoogleGenerativeAI)

Multi-Agent Design 
1) Discovery Agent
   - Scans all CSVs in INPUT_DIR
   - Reads samples from every file and builds a combined view
   - Calls Ollama once (or per distinct schema group) to classify columns into PII/PHI/NONE and suggest subtypes
   - Detects linked columns across files (same column name or high value-overlap). Creates linking groups so the same original value maps to the same synthetic value across files.
   - Produces a canonical col_info + link_groups that guide anonymization.

2) Parallel Anonymizer Agents
   - Using the global plan, anonymize CSVs in parallel (ThreadPoolExecutor) ensuring group-level mapping is shared
   - Mapping is deterministic and stored to MAPPING_FILE for re-use and audit

Key Features:
- Link detection across columns to preserve referential integrity (account IDs, client IDs, emails)
- Single classification step so Ollama "thinks" about the whole dataset first
- Parallelized anonymization for speed

Usage
- Set INPUT_DIR and OUTPUT_DIR at top of file
- Ensure GOOGLE_API_KEY/Ollama access (via .env) and required packages installed
"""

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Dict, Any, List, Tuple
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from faker import Faker
import pandas as pd
import os
import json
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_ollama import ChatOllama      




# ----------------------
# Config - set path here
# ----------------------
INPUT_DIR = "data/input"        # folder containing original CSVs
OUTPUT_DIR = "data/output"      # folder where anonymized CSVs will be placed
MAPPING_FILE = "data/mapping.json"  # mapping file (stores original -> synthetic)
SAMPLE_ROWS = 20                 # how many sample rows to send to Ollama for classification
SEED = 42                        # deterministic seed for Faker + mapping
LINK_OVERLAP_THRESHOLD = 0.2     # fraction of overlap to consider columns linked
MAX_WORKERS = 4                  # parallel workers for anonymization

# ----------------------
# Setup
# ----------------------
load_dotenv()
llm = ChatOllama(model="llama3:8b") 

faker = Faker()
faker.seed_instance(SEED)

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------
# Utility functions
# ----------------------

def deterministic_hash(val: str) -> str:
    """Create a short deterministic hash of the string."""
    if val is None:
        return ""
    h = hashlib.sha256(val.encode("utf-8", errors="ignore")).hexdigest()
    return h[:16]


def load_mapping(path: str) -> Dict[str, Dict[str, str]]:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_mapping(path: str, mapping: Dict[str, Dict[str, str]]):
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2)

# ----------------------
# Ollama helpers
# ----------------------

def build_combined_prompt(all_samples: Dict[str, pd.DataFrame]) -> str:
    """Build a single prompt for Ollama that contains samples from all files and asks for classification per column name.
    The model will be asked to consider that columns with the same name across files should be treated consistently.
    """
    parts = []
    for path, df in all_samples.items():
        snippet = df.to_csv(index=False, lineterminator="\n")
        parts.append(f"---- FILE: {os.path.basename(path)} ----\n{snippet}")
    combined = "\n\n".join(parts)
    prompt = (
        "You are a data classification assistant.\n\n"
        "Given CSV snippets from multiple files, identify for each distinct column name whether it contains PII, PHI (HIPAA), or NONE. "
        "If the same column name appears in multiple files, treat it consistently. Also suggest a subtype (one of: name, email, phone, ssn, national_id, address, date_of_birth, medical_condition, medical_record_number, account_number, generic_id, other).\n\n"
        "Return ONLY valid JSON: a list of objects with fields: column, label (PII|PHI|NONE), subtype, confidence (0-1), examples (up to 5 example values).\n\n"
        f"Data snippets:\n{combined}\n\nAnswer now."
    )
    return prompt


def call_Ollama_combined(all_samples: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    prompt = build_combined_prompt(all_samples)
    messages = [SystemMessage(content="You are a helpful classifier."), HumanMessage(content=prompt)]
    resp = llm.invoke(messages).content
    try:
        start = min([i for i in (resp.find('['), resp.find('{')) if i != -1])
        end = max(resp.rfind(']'), resp.rfind('}'))
        json_text = resp[start:end+1]
        parsed = json.loads(json_text)
        return parsed
    except Exception:
        # fallback: return nothing classified
        all_cols = set()
        for df in all_samples.values():
            all_cols.update(df.columns.tolist())
        return [{"column": c, "label": "NONE", "subtype": "other", "confidence": 0.0, "examples": []} for c in sorted(all_cols)]

# ----------------------
# Linking detection
# ----------------------

def find_linked_columns(dfs: Dict[str, pd.DataFrame]) -> List[List[Tuple[str, str]]]:
    """Detect columns that refer to the same identifier across files.
    Returns groups as lists of (file_path, column_name).
    Heuristics:
    - Same column name -> linked
    - OR substantial overlap of unique values between two columns -> linked
    """
    # map (path, col) -> set(values)
    col_values = {}
    for path, df in dfs.items():
        for col in df.columns:
            vals = set(df[col].dropna().astype(str).unique().tolist())
            col_values[(path, col)] = vals

    # start with groups by column name
    name_groups = {}
    for (path, col) in col_values.keys():
        name_groups.setdefault(col, []).append((path, col))

    groups = [g[:] for g in name_groups.values() if len(g) > 1]

    # compare cross-file different-named columns for overlap
    items = list(col_values.items())
    n = len(items)
    for i in range(n):
        (p1c, vals1) = items[i]
        for j in range(i+1, n):
            (p2c, vals2) = items[j]
            if p1c[0] == p2c[0]:
                continue  # same file, skip
            if p1c[1] == p2c[1]:
                continue  # already grouped by name
            if not vals1 or not vals2:
                continue
            inter = len(vals1 & vals2)
            denom = min(len(vals1), len(vals2))
            if denom == 0:
                continue
            overlap = inter / denom
            if overlap >= LINK_OVERLAP_THRESHOLD:
                # find existing group and merge or create
                merged = False
                for g in groups:
                    if p1c in g or p2c in g:
                        if p1c not in g:
                            g.append(p1c)
                        if p2c not in g:
                            g.append(p2c)
                        merged = True
                        break
                if not merged:
                    groups.append([p1c, p2c])

    # normalize groups (unique entries)
    norm = []
    seen = set()
    for g in groups:
        uniq = []
        for item in g:
            if item not in seen:
                uniq.append(item)
                seen.add(item)
        if uniq:
            norm.append(uniq)
    return norm

# ----------------------
# Synthetic generators mapping
# ----------------------

def synth_value_for_subtype(subtype: str, original_val: str) -> str:
    """Generate a synthetic value for a given subtype deterministically.
    Uses deterministic_hash + Faker to keep mappings stable across runs.
    """
    base = deterministic_hash(original_val or "")
    seed = int(base[:8], 16) % (2**32)
    faker.seed_instance(seed)
    if subtype == "name":
        return faker.name()
    if subtype == "email":
        return faker.safe_email()
    if subtype in ("phone", "phone_number"):
        return faker.phone_number()
    if subtype in ("ssn", "national_id"):
        return faker.bothify(text="###-##-####")
    if subtype in ("address",):
        return faker.address().replace('\n', ', ')
    if subtype in ("date_of_birth", "date"):
        return faker.date_of_birth().isoformat()
    if subtype in ("medical_record_number", "medical_condition"):
        return f"MRN-{faker.bothify(text='????-#####') }"
    if subtype in ("generic_id", "id", "account_number"):
        return faker.bothify(text="ACC-########")
    # fallback
    return faker.word() + "_anon"

# ----------------------
# Main anonymization logic (two-stage)
# ----------------------

def build_global_plan(csv_paths: List[str]) -> Tuple[Dict[str, Any], List[List[Tuple[str, str]]]]:
    """Stage 1 (Discovery): read samples, call Ollama, detect links, return col_info and link_groups."""
    samples = {}
    full_dfs = {}
    for p in csv_paths:
        df = pd.read_csv(p, low_memory=False)
        full_dfs[p] = df
        samples[p] = df.head(SAMPLE_ROWS)

    print("Calling Ollama to classify columns across all files...")
    classifications = call_Ollama_combined(samples)
    # normalize into dict col -> info
    col_info = {c['column']: c for c in classifications}

    print("Detecting linked columns (same IDs across files)...")
    link_groups = find_linked_columns(full_dfs)

    # Build a canonical group mapping: group_id -> list of (path,col)
    return col_info, link_groups


def anonymize_with_plan(csv_paths: List[str], col_info: Dict[str, Any], link_groups: List[List[Tuple[str, str]]]):
    """Stage 2: parallel anonymization using the global plan."""
    mapping = load_mapping(MAPPING_FILE)

    # Build group key -> canonical mapping dict name
    group_key_for = {}
    for gidx, group in enumerate(link_groups):
        key = f"group_{gidx}"
        for (path, col) in group:
            group_key_for[(path, col)] = key

    # Also map single columns by column name when not in groups
    # We'll use column name as fallback key (so same column name across files uses same mapping)

    def get_key(path, col):
        return group_key_for.get((path, col), f"col::{col}")

    # Ensure mapping structure exists for keys
    for p in csv_paths:
        df = pd.read_csv(p, low_memory=False)
    
    # Worker to anonymize a single file
    def anonymize_file(path: str):
        print("Anonymizing file:", path)
        df = pd.read_csv(path, low_memory=False)
        df_out = df.copy()
        for col in df.columns:
            info = col_info.get(col, {"label": "NONE", "subtype": "other"})
            label = info.get('label', 'NONE').upper()
            subtype = info.get('subtype', 'other')
            if label in ('PII', 'PHI'):
                key = get_key(path, col)
                if key not in mapping:
                    mapping[key] = {}
                unique_vals = pd.Index(df_out[col].fillna("__NULL__")).unique().tolist()
                for val in unique_vals:
                    val_key = "" if pd.isna(val) else str(val)
                    if val_key not in mapping[key]:
                        mapping[key][val_key] = synth_value_for_subtype(subtype, val_key)
                # apply mapping
                df_out[col] = df_out[col].fillna("__NULL__").astype(str).map(lambda x: mapping[key].get(x, x))
        out_path = os.path.join(OUTPUT_DIR, os.path.basename(path))
        df_out.to_csv(out_path, index=False)
        print("Wrote anonymized:", out_path)
        return out_path

    # Parallel run
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(anonymize_file, p): p for p in csv_paths}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print("Error anonymizing", futures[fut], e)

    # Save mapping
    save_mapping(MAPPING_FILE, mapping)
    print("Saved mapping to", MAPPING_FILE)


def anonymize_csv_files():
    csv_paths = sorted([str(p) for p in Path(INPUT_DIR).glob("*.csv")])
    if not csv_paths:
        print("No CSV files found in INPUT_DIR. Put CSVs into:", INPUT_DIR)
        return

    col_info, link_groups = build_global_plan(csv_paths)
    anonymize_with_plan(csv_paths, col_info, link_groups)

# ----------------------
# Optional: StateGraph agent wrapper (creative agent)
# ----------------------

class MaskState(TypedDict):
    input_dir: str
    output_dir: str
    mapping_file: str
    status: str


def prepare(state: MaskState):
    # trivial initializer
    return {"status": "prepared"}


def run_mask(state: MaskState):
    anonymize_csv_files()
    return {"status": "done"}

# Build a tiny graph if desired
mask_graph = StateGraph(MaskState)
mask_graph.add_node("prepare", prepare)
mask_graph.add_node("run", run_mask)
mask_graph.add_edge(START, "prepare")
mask_graph.add_edge("prepare", "run")
mask_workflow = mask_graph.compile()

# ----------------------
# Usage
# ----------------------
if __name__ == '__main__':
    initial = {"input_dir": INPUT_DIR, "output_dir": OUTPUT_DIR, "mapping_file": MAPPING_FILE}
    print("Starting anonymization agent...\nInput:", INPUT_DIR, "\nOutput:", OUTPUT_DIR)
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    result = mask_workflow.invoke(initial)
    print("Agent finished with status:", result.get("status", "unknown"))

# End of file
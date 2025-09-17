"""
CSV PII/HIPAA Masking + Synthetic Data Generator using Gemini (ChatGoogleGenerativeAI)

How it works
- Scans INPUT_DIR for CSV files.
- Loads a sample of rows from each CSV, asks Gemini to classify which columns contain PII / PHI / sensitive info.
- Creates deterministic synthetic replacements (using Faker) for values in sensitive columns.
  - Uses a stored mapping (JSON) so repeats and linking columns across files get the same anonymized values.
- Writes anonymized CSV files to OUTPUT_DIR and writes a mapping JSON file (MAPPING_FILE).

Requirements
- Python 3.9+
- pip install pandas faker python-dotenv langchain-google-genai
- Set environment variable GOOGLE_API_KEY for Gemini access.

Paths are defined in the code (INPUT_DIR, OUTPUT_DIR, MAPPING_FILE). Modify them in the file if needed.

Note: Gemini is used only to *detect and classify* sensitive columns and to help design replacement strategies. The actual replacement is deterministic and performed locally so results are reproducible and can be inspected.

"""

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from faker import Faker
import pandas as pd
import os
import json
import hashlib
from pathlib import Path

# ----------------------
# Config - set path here
# ----------------------
INPUT_DIR = "data/input"        # folder containing original CSVs
OUTPUT_DIR = "data/output"      # folder where anonymized CSVs will be placed
MAPPING_FILE = "data/mapping.json"  # mapping file (stores original -> synthetic)
SAMPLE_ROWS = 10                 # how many sample rows to send to Gemini for classification
SEED = 42                        # deterministic seed for Faker + mapping
MODEL_NAME = "gemini-1.5-flash"  # change if necessary

# ----------------------
# Setup
# ----------------------
load_dotenv()
llm = ChatGoogleGenerativeAI(model=MODEL_NAME)
Faker.seed(SEED)     # Seed the global Faker generator
faker = Faker()

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
# Gemini helpers
# ----------------------

def build_classification_prompt(sample_table: pd.DataFrame) -> str:
    """Construct a prompt to ask Gemini to classify columns as PII/PHI/None and suggest the type.
    The response should be a JSON array of objects: [{"column":"colname","label":"PII|PHI|NONE","type":"email|name|ssn|phone|date|id|medical|other","reason":"..."}, ...]
    """
    # Show column names and SAMPLE_ROWS rows as CSV-ish snippet
    snippet = sample_table.to_csv(index=False, lineterminator="\n")
    prompt = f"You are a data classification assistant.\n\nGiven the following CSV snippet (columns + up to {SAMPLE_ROWS} rows), identify for each column whether it contains PII, PHI (HIPAA), or NONE. Also suggest a likely subtype (one of: name, email, phone, ssn, national_id, address, date_of_birth, medical_condition, medical_record_number, account_number, generic_id, other).\n\nReturn ONLY valid JSON: a list of objects with fields: column, label (PII|PHI|NONE), subtype, confidence (0-1).\n\nCSV snippet:\n{snippet}\n\nAnswer now."
    return prompt


def call_gemini_classify(sample_df: pd.DataFrame) -> list:
    prompt = build_classification_prompt(sample_df)
    messages = [SystemMessage(content="You are a helpful classifier."), HumanMessage(content=prompt)]
    resp = llm.invoke(messages).content
    # Try to parse JSON from response; be resilient if model adds commentary
    try:
        # find first '{' or '[' and last '}' or ']' to extract JSON
        start = min([i for i in (resp.find('['), resp.find('{')) if i != -1])
        end = max(resp.rfind(']'), resp.rfind('}'))
        json_text = resp[start:end+1]
        parsed = json.loads(json_text)
        return parsed
    except Exception:
        # fallback: try to parse line-by-line (best-effort)
        # We'll default all to NONE if it fails
        cols = sample_df.columns.tolist()
        return [{"column": c, "label": "NONE", "subtype": "other", "confidence": 0.0} for c in cols]

# ----------------------
# Synthetic generators mapping
# ----------------------

def synth_value_for_subtype(subtype: str, original_val: str) -> str:
    """Generate a synthetic value for a given subtype deterministically.
    Uses deterministic_hash + Faker to keep mappings stable across runs.
    """
    base = deterministic_hash(original_val or "")
    # Use subtype-based generation
    if subtype == "name":
        # Use hashed seed to generate a name
        faker.seed_instance(int(base[:8], 16) % (2**32))
        return faker.name()
    if subtype == "email":
        faker.seed_instance(int(base[:8], 16) % (2**32))
        return faker.safe_email()
    if subtype in ("phone", "phone_number"):
        faker.seed_instance(int(base[:8], 16) % (2**32))
        return faker.phone_number()
    if subtype in ("ssn", "national_id"):
        # generate pseudo-ssn like string
        faker.seed_instance(int(base[:8], 16) % (2**32))
        return faker.bothify(text="###-##-####")
    if subtype in ("address",):
        faker.seed_instance(int(base[:8], 16) % (2**32))
        return faker.address().replace('\n', ', ')
    if subtype in ("date_of_birth", "date"):
        faker.seed_instance(int(base[:8], 16) % (2**32))
        return faker.date_of_birth().isoformat()
    if subtype in ("medical_record_number", "medical_condition"):
        faker.seed_instance(int(base[:8], 16) % (2**32))
        return f"MRN-{faker.bothify(text='????-#####') }"
    if subtype in ("generic_id", "id", "account_number"):
        # keep format but randomize digits
        faker.seed_instance(int(base[:8], 16) % (2**32))
        return faker.bothify(text="ACC-########")
    # fallback
    faker.seed_instance(int(base[:8], 16) % (2**32))
    return faker.word() + "_anon"

# ----------------------
# Main anonymization logic
# ----------------------

def anonymize_csv_files():
    mapping = load_mapping(MAPPING_FILE)

    # find CSVs
    csv_paths = sorted([str(p) for p in Path(INPUT_DIR).glob("*.csv")])
    if not csv_paths:
        print("No CSV files found in INPUT_DIR. Put CSVs into:", INPUT_DIR)
        return

    # Read first SAMPLE_ROWS from each to create classification prompt
    samples = {}
    for p in csv_paths:
        df = pd.read_csv(p, low_memory=False)
        samples[p] = df.head(SAMPLE_ROWS)

    # Decide whether all CSVs share same columns
    all_columns = [tuple(df.columns.tolist()) for df in (pd.read_csv(p, nrows=0) for p in csv_paths)]
    same_schema = all(c == all_columns[0] for c in all_columns)
    print(f"Detected {len(csv_paths)} CSV(s). Same schema across files: {same_schema}")

    # If same schema -> classify once using the first file's samples
    if same_schema:
        sample_df = samples[csv_paths[0]]
        classification = call_gemini_classify(sample_df)
        # normalize classification into dict col->info
        col_info = {c['column']: c for c in classification}

        # For each file, apply anonymization with same mapping
        for p in csv_paths:
            df = pd.read_csv(p, low_memory=False)
            anon_df, mapping = anonymize_dataframe(df, col_info, mapping)
            out_path = os.path.join(OUTPUT_DIR, os.path.basename(p))
            anon_df.to_csv(out_path, index=False)
            print("Wrote anonymized:", out_path)
    else:
        # Different schemas: classify each file separately but reuse mapping across columns with same name
        for p in csv_paths:
            print("Classifying file:", p)
            sample_df = samples[p]
            classification = call_gemini_classify(sample_df)
            col_info = {c['column']: c for c in classification}
            df = pd.read_csv(p, low_memory=False)
            anon_df, mapping = anonymize_dataframe(df, col_info, mapping)
            out_path = os.path.join(OUTPUT_DIR, os.path.basename(p))
            anon_df.to_csv(out_path, index=False)
            print("Wrote anonymized:", out_path)

    # Save final mapping
    save_mapping(MAPPING_FILE, mapping)
    print("Saved mapping to", MAPPING_FILE)


def anonymize_dataframe(df: pd.DataFrame, col_info: Dict[str, Any], mapping: Dict[str, Dict[str, str]]):
    """Takes a dataframe and column classification info, returns anonymized df and updated mapping."""
    df_out = df.copy()
    for col, info in col_info.items():
        label = info.get('label', 'NONE').upper()
        subtype = info.get('subtype', 'other')
        if label in ('PII', 'PHI'):
            print(f"Anonymizing column: {col} -> {label} ({subtype})")
            if col not in mapping:
                mapping[col] = {}
            # Build mapping for unique values
            unique_vals = pd.Index(df_out[col].fillna("__NULL__")).unique().tolist()
            for val in unique_vals:
                val_key = "" if pd.isna(val) else str(val)
                if val_key not in mapping[col]:
                    synth = synth_value_for_subtype(subtype, val_key)
                    mapping[col][val_key] = synth
            # Apply mapping
            df_out[col] = df_out[col].fillna("__NULL__").astype(str).map(lambda x: mapping[col].get(x, x))
    return df_out, mapping

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
    # You can change paths here if you want (but paths are already set at top of file)
    initial = {"input_dir": INPUT_DIR, "output_dir": OUTPUT_DIR, "mapping_file": MAPPING_FILE}
    print("Starting anonymization agent...\nInput:", INPUT_DIR, "\nOutput:", OUTPUT_DIR)
    # Ensure directories exist
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run workflow (calls anonymize_csv_files internally)
    result = mask_workflow.invoke(initial)
    print("Agent finished with status:", result.get("status", "unknown"))

# End of file

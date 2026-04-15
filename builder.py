import json
import requests
import time
import os
from neo4j import GraphDatabase
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gpt-oss:120b-cloud"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "neo4jdesktop2"  

INPUT_FILE = "trials.json"
REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 1.5

OLLAMA_URL = os.getenv("OLLAMA_URL", OLLAMA_URL)
MODEL_NAME = os.getenv("MODEL_NAME", MODEL_NAME)
NEO4J_URI = os.getenv("NEO4J_URI", NEO4J_URI)
NEO4J_USER = os.getenv("NEO4J_USER", NEO4J_USER)
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", NEO4J_PASSWORD)

# ==============================
# ONTOLOGY
# ==============================

ALLOWED_ENTITY_TYPES = {
    "Trial",
    "Drug",
    "Condition",
    "Biomarker",
    "Outcome",
    "Phase",
    "Sponsor"
}

ALLOWED_RELATIONS = {
    "tests",
    "targets",
    "measures",
    "has_phase",
    "sponsored_by",
    "affects"
}

# ==============================
# OLLAMA CALL
# ==============================

def extract_with_llm(text):
    prompt = f"""
You are an information extraction system.

Extract entities and relationships from the text.

Use ONLY these entity types:
Trial, Drug, Condition, Biomarker, Outcome, Phase, Sponsor

Use ONLY these relationships:
tests, targets, measures, has_phase, sponsored_by, affects

Return output strictly in JSON format:

{{
  "entities": [
    {{"name": "...", "type": "..."}}
  ],
  "relations": [
    {{"source": "...", "relation": "...", "target": "..."}}
  ]
}}

Text:
\"\"\"
{text}
\"\"\"
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as e:
        print("❌ LLM request failed:", e)
        return None
    except ValueError as e:
        print("❌ Invalid JSON response from LLM endpoint:", e)
        return None

    output = payload.get("response")
    if not isinstance(output, str):
        print("❌ Missing or invalid `response` field in LLM payload.")
        return None

    # Try parsing model output JSON safely
    try:
        json_start = output.find("{")
        if json_start == -1:
            raise ValueError("No JSON object found in model output.")
        json_data = json.loads(output[json_start:])
        return json_data
    except Exception as e:
        print("❌ JSON parsing failed:", e)
        return None

# ==============================
# VALIDATION
# ==============================

def validate_extraction(data):
    if not isinstance(data, dict):
        return None

    valid_entities = []
    valid_relations = []
    known_entity_names = set()

    for ent in data.get("entities", []):
        if not isinstance(ent, dict):
            continue
        name = ent.get("name")
        ent_type = ent.get("type")
        if (
            isinstance(name, str)
            and name.strip()
            and ent_type in ALLOWED_ENTITY_TYPES
        ):
            cleaned = {"name": name.strip(), "type": ent_type}
            valid_entities.append(cleaned)
            known_entity_names.add(cleaned["name"])

    for rel in data.get("relations", []):
        if not isinstance(rel, dict):
            continue
        source = rel.get("source")
        relation = rel.get("relation")
        target = rel.get("target")
        if (
            isinstance(source, str)
            and source.strip()
            and isinstance(target, str)
            and target.strip()
            and relation in ALLOWED_RELATIONS
            and source.strip() in known_entity_names
            and target.strip() in known_entity_names
        ):
            valid_relations.append(
                {
                    "source": source.strip(),
                    "relation": relation,
                    "target": target.strip(),
                }
            )

    return {
        "entities": valid_entities,
        "relations": valid_relations
    }

# ==============================
# NEO4J FUNCTIONS
# ==============================

class KnowledgeGraph:

    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        # Fail fast if the DB is unreachable.
        self.driver.verify_connectivity()

    def close(self):
        self.driver.close()

    def insert_data(self, trial_id, data):
        with self.driver.session() as session:
            session.execute_write(self._insert_tx, trial_id, data)

    @staticmethod
    def _insert_tx(tx, trial_id, data):

        # Create trial node
        tx.run(
            "MERGE (t:Trial {name: $name})",
            name=trial_id
        )

        # Insert entities
        for ent in data["entities"]:
            tx.run(
                f"MERGE (e:{ent['type']} {{name: $name}})",
                name=ent["name"]
            )

        # Insert relationships
        for rel in data["relations"]:
            tx.run(
                f"""
                MATCH (a {{name: $source}})
                MATCH (b {{name: $target}})
                MERGE (a)-[:{rel['relation'].upper()}]->(b)
                """,
                source=rel["source"],
                target=rel["target"]
            )

# ==============================
# MAIN PIPELINE
# ==============================

def main():
    input_file = os.getenv("INPUT_FILE", INPUT_FILE)

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            trials = json.load(f)
    except FileNotFoundError:
        print(f"❌ Input file not found: {input_file}")
        return
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON from {input_file}: {e}")
        return

    if not isinstance(trials, list):
        print("❌ Expected input JSON to be a list of trial objects.")
        return

    try:
        kg = KnowledgeGraph()
    except Exception as e:
        print(f"❌ Failed to initialize Neo4j connection: {e}")
        return

    inserted_count = 0
    skipped_count = 0

    try:
        for trial in tqdm(trials):
            if not isinstance(trial, dict):
                skipped_count += 1
                continue

            trial_id = trial.get("nct_id")
            title = trial.get("title", "")
            description = trial.get("description", "")

            if not trial_id or (not title and not description):
                skipped_count += 1
                continue

            combined_text = f"{title}\n{description}".strip()
            validated = None

            for attempt in range(MAX_RETRIES):
                extracted = extract_with_llm(combined_text)
                validated = validate_extraction(extracted)

                if validated and validated["entities"]:
                    try:
                        kg.insert_data(trial_id, validated)
                        inserted_count += 1
                        break
                    except Exception as e:
                        print(f"❌ Neo4j insert failed for trial {trial_id}: {e}")
                else:
                    print(f"Retrying trial {trial_id}")

                # Backoff between retries.
                sleep_seconds = RETRY_BACKOFF_SECONDS * (attempt + 1)
                time.sleep(sleep_seconds)

            if not validated:
                skipped_count += 1
    finally:
        kg.close()

    print(
        f"✅ Knowledge graph construction complete. "
        f"Inserted: {inserted_count}, Skipped: {skipped_count}"
    )


if __name__ == "__main__":
    main()

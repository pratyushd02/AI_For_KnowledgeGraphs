import json
import requests
import time
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

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )

    output = response.json()["response"]

    # Try parsing JSON safely
    try:
        json_start = output.find("{")
        json_data = json.loads(output[json_start:])
        return json_data
    except Exception as e:
        print("❌ JSON parsing failed:", e)
        return None

# ==============================
# VALIDATION
# ==============================

def validate_extraction(data):
    if not data:
        return None

    valid_entities = []
    valid_relations = []

    for ent in data.get("entities", []):
        if ent["type"] in ALLOWED_ENTITY_TYPES:
            valid_entities.append(ent)

    for rel in data.get("relations", []):
        if rel["relation"] in ALLOWED_RELATIONS:
            valid_relations.append(rel)

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

    with open(INPUT_FILE, "r") as f:
        trials = json.load(f)

    kg = KnowledgeGraph()

    for trial in tqdm(trials):

        trial_id = trial["nct_id"]
        combined_text = trial["title"] + "\n" + trial["description"]

        for attempt in range(3):
            extracted = extract_with_llm(combined_text)
            validated = validate_extraction(extracted)

            if validated:
                kg.insert_data(trial_id, validated)
                break
            else:
                print(f"Retrying trial {trial_id}")
                time.sleep(1)

    kg.close()
    print("✅ Knowledge graph construction complete.")


if __name__ == "__main__":
    main()

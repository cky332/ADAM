"""Synthetic agent memories with *known* latent topic distributions.

A literal reproduction would populate memory from MIMIC-III (EHRAgent),
HotpotQA (ReAct) and WebShop (RAP). MIMIC-III is credentialed patient data, and
reproducing a privacy attack on *real* patient records would itself be a privacy
problem, so we instead synthesise records that mimic the *structure* the attack
exploits:

  * each record has a latent topic z_i drawn from a non-uniform (Zipfian)
    ground-truth distribution P(D) -- this is what the oracle test (Fig. 4)
    compares the estimator against;
  * query text is topic-conditioned and contains the topic word plus a little
    topic-specific vocabulary, so same-topic records cluster in embedding space
    exactly as real domain queries do.

The memory's topic set is a *superset* of the attacker's seed topics, so the
attacker must *discover* new topics through anchor extraction -- which is where
ADAM's distribution estimation pays off.
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

from adam.embeddings import get_encoder
from adam.memory import Memory, Record

# topic -> (relative weight, [templates]).  {pid}/{date}/{ent} are entity slots.
EHRAGENT_TOPICS: Dict[str, Tuple[float, List[str]]] = {
    "diagnosis":    (5, ["What diagnosis did patient {pid} receive on {date}?",
                          "List the IDs of patients with a {ent} diagnosis since {date}."]),
    "medication":   (4, ["What was the name of the medication patient {pid} received on {date}?",
                          "Which medication dose did patient {pid} get for {ent}?"]),
    "output":       (4, ["What was the name of the output that patient {pid} first had on {date}?",
                          "What was the total volume of output patient {pid} had on the {ent} icu visit?"]),
    "lab":          (3, ["What was the lab test result of patient {pid} on {date}?",
                          "What is the latest {ent} lab value measured for patient {pid}?"]),
    "microbiology": (2, ["Has patient {pid} had any {ent} microbiology test result since {date}?",
                          "What microbiology specimen was collected from patient {pid} on {date}?"]),
    "vital":        (3, ["What was the vital sign reading of patient {pid} on {date}?",
                          "What is the change in vital {ent} of patient {pid} since {date}?"]),
    "patient":      (3, ["What is the date of birth of patient {pid}?",
                          "What is the {ent} of patient {pid} on admission?"]),
    "prescription": (2, ["What prescription was issued to patient {pid} on {date}?",
                          "List prescriptions for patient {pid} related to {ent}."]),
    "icu":          (2, ["When was patient {pid} admitted to the icu for {ent}?",
                          "What happened on patient {pid}'s first icu visit on {date}?"]),
    "allergy":      (1, ["Does patient {pid} have any recorded allergy to {ent}?",
                          "What allergy was documented for patient {pid} on {date}?"]),
}

REACT_TOPICS: Dict[str, Tuple[float, List[str]]] = {
    "history":    (5, ["What history event in {ent} connects the two referenced figures?",
                        "Which history war involving {ent} preceded the treaty mentioned?"]),
    "geography":  (4, ["What geography region contains both the {ent} river and mountain?",
                        "Which geography continent borders the {ent} ocean described above?"]),
    "science":    (4, ["What science discipline studies the {ent} particle mentioned earlier?",
                        "Which science theory explains the {ent} phenomenon referenced?"]),
    "biography":  (3, ["Whose biography mentions winning the {ent} prize and a Nobel medal?",
                        "What biography links the {ent} inventor to the laureate above?"]),
    "politics":   (3, ["What politics office did the {ent} party leader hold?",
                        "Which politics election involved the {ent} candidate referenced?"]),
    "literature": (3, ["Which literature novel by the {ent} author won the referenced award?",
                        "What literature genre is the {ent} poet best known for?"]),
    "sports":     (2, ["What sports record did the {ent} athlete set at the championship?",
                        "Which sports team drafted the {ent} player mentioned above?"]),
    "culture":    (2, ["What culture festival celebrates the {ent} tradition referenced?",
                        "Which culture custom is linked to the {ent} ceremony above?"]),
    "discovery":  (2, ["What discovery about {ent} is the referenced scientist credited with?",
                        "Which discovery of the {ent} element preceded the prize mentioned?"]),
    "invention":  (2, ["What invention of the {ent} machine is linked to the referenced engineer?",
                        "Which invention used the {ent} mechanism described in the article?"]),
}

RAP_TOPICS: Dict[str, Tuple[float, List[str]]] = {
    "hair":        (5, ["Find me sulfate free paraben free hair masks for {ent} hair."]),
    "skin":        (4, ["Find me cruelty free skin care with {ent} for sensitive skin."]),
    "grocery":     (4, ["Find me low sugar grocery cookies with {ent} flavor."]),
    "snack":       (3, ["Find me high protein snack crackers that are {ent}."]),
    "supplement":  (3, ["Recommend a {ent} supplement for daily use."]),
    "beauty":      (3, ["Find me a {ent} beauty product with good reviews."]),
    "cleaning":    (2, ["Find me a {ent} cleaning product that is eco friendly."]),
    "electronics": (2, ["Find me a {ent} electronics accessory under budget."]),
    "clothing":    (2, ["Find me {ent} clothing in travel size."]),
    "pet":         (1, ["Find me {ent} pet care items with natural ingredients."]),
}

_ENTITIES = {
    "ehragent": ["sepsis", "anemia", "diabetes", "hemoglobin", "urine", "sputum",
                 "cannabis abuse", "anxiety", "first", "second", "creatinine",
                 "pneumonia", "stroke", "insulin", "potassium", "ventilator"],
    "react": ["alpha", "beta", "gamma", "delta", "omega", "northern", "southern",
              "eastern", "western", "ancient", "modern", "imperial", "atomic",
              "solar", "lunar", "golden", "silver", "crimson", "azure", "ironclad"],
    "rap": ["argan oil", "tea tree", "vanilla", "gluten free", "vitamin c",
            "lavender", "organic", "wireless", "cotton", "salmon", "almond",
            "charcoal", "collagen", "bamboo", "ceramic", "stainless", "merino"],
}

_AGENTS = {"ehragent": EHRAGENT_TOPICS, "react": REACT_TOPICS, "rap": RAP_TOPICS}

# The attacker's initial domain seeds. To make retrieval meaningful in the
# synthetic setting, these overlap the memory vocabulary -- but only as a
# *subset* of the agent's true topics, so the remaining topics must be
# discovered through anchor extraction (which is where ADAM's distribution
# estimation pays off). The paper's documented seed lists are in
# adam.config.SEED_TOPICS.
SIM_SEEDS = {
    "ehragent": ["diagnosis", "medication", "patient", "lab", "vital", "icu"],
    "react": ["history", "geography", "science", "biography", "politics", "sports"],
    "rap": ["hair", "skin", "grocery", "snack", "beauty", "electronics"],
}


def sim_seeds(domain: str) -> List[str]:
    return list(SIM_SEEDS[domain])


def _fill(template: str, rng: random.Random, domain: str) -> str:
    return template.format(
        pid=rng.randint(1000, 99999),
        date=f"{rng.randint(1,12):02d}/{rng.randint(1,28):02d}/2{rng.randint(100,115)}",
        ent=rng.choice(_ENTITIES[domain]),
    )


def build_memory(domain: str, size: int = 300, encoder=None, seed: int = 0) -> Memory:
    """Create a victim memory of ``size`` records for ``domain``."""
    assert domain in _AGENTS, f"unknown domain '{domain}'"
    rng = random.Random(seed)
    encoder = encoder or get_encoder("hashing")
    topics = _AGENTS[domain]
    names = list(topics.keys())
    weights = [topics[t][0] for t in names]

    records: List[Record] = []
    for i in range(size):
        topic = rng.choices(names, weights=weights, k=1)[0]
        template = rng.choice(topics[topic][1])
        query = _fill(template, rng, domain)
        solution = f"[solution for record {i} about {topic}]"
        records.append(Record(qid=i, query=query, solution=solution, topic=topic))
    return Memory(records, encoder)


def ground_truth_topics(domain: str) -> Dict[str, float]:
    """Normalised P(D) over latent topics (the oracle distribution)."""
    topics = _AGENTS[domain]
    total = float(sum(w for w, _ in topics.values()))
    return {t: w / total for t, (w, _) in topics.items()}

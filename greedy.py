import random
import json
import re
import torch
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

# =========================
# Phase 0: GPU + Model Load
# =========================

bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )

assert torch.cuda.is_available(), "CUDA not available."

device = torch.device("cuda")
#MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    #dtype=torch.float16,
    device_map="auto"
)

model.eval()

def generate(prompt, max_tokens=128):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=0.0
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def extract_json_object(text):
    decoder = json.JSONDecoder()
    text = text.strip()

    # Find first JSON object properly
    obj, end = decoder.raw_decode(text[text.find("{"):])
    return obj


# =========================
# Phase 1: Semantic Source
# =========================

def semantic_source():
    return {
        "objects": [
            "3-story residential building",
            "fire on 3rd floor east window",
            "2 trapped civilians",
            "fire truck",
            "dense smoke plume"
        ],
        "actions": [
            "burning intensely",
            "civilians waving for help",
            "firefighters deploying ladder"
        ],
        "relations": [
            "fire located on east facade",
            "civilians inside third floor",
            "ladder positioned below window"
        ],
        "context": [
            "urban disaster response",
            "evening time",
            "low visibility due to smoke"
        ]
    }

# =========================
# Phase 2: Importance Scoring (LLM Ranking)
# =========================

def score_importance_batch(S):
    flat = []
    for cat in S:
        for elem in S[cat]:
            flat.append((cat, elem))

    numbered = "\n".join(
        [f"{i}. [{cat}] {elem}" for i, (cat, elem) in enumerate(flat)]
    )

    prompt = f"""
Rank the following items from MOST important to LEAST important
for emergency rescue decision making.

Return ONLY a comma-separated list of indices.
Example:
3,1,0,2

Items:
{numbered}
"""

    response = generate(prompt, max_tokens=100)

    print("\n=== RAW IMPORTANCE RESPONSE ===")
    print(response)

    # Extract numeric indices
    indices = re.findall(r'\d+', response)

    valid = []
    seen = set()

    for i in indices:
        idx = int(i)
        if 0 <= idx < len(flat) and idx not in seen:
            valid.append(idx)
            seen.add(idx)

    # Append any missing indices at end (lowest priority)
    for i in range(len(flat)):
        if i not in seen:
            valid.append(i)

    scored = []
    total = len(flat)

    for rank, idx in enumerate(valid):
        cat, elem = flat[idx]
        score = max(0.0, 1.0 - (rank / total))
        scored.append((cat, elem, score))

    print("\n=== PROCESSED IMPORTANCE SCORES ===")
    for s in scored:
        print(s)

    return scored


# =========================
# Phase 3: Greedy Pruning
# =========================

def prune_semantics(S, scored, max_tokens):
    S_new = deepcopy(S)

    total = sum(len(v) for v in S.values())
    if total <= max_tokens:
        return S_new

    scored.sort(key=lambda x: x[2])
    to_remove = total - max_tokens

    for i in range(to_remove):
        cat, elem, _ = scored[i]
        if elem in S_new[cat]:
            S_new[cat].remove(elem)

    return S_new

# =========================
# Phase 4: Channel
# =========================

def corrupt(S, p):
    return {k: [x for x in S[k] if random.random() > p] for k in S}

# =========================
# Phase 5: SER
# =========================

def set_ser(original, received):
    if len(original) == 0:
        return 0.0
    diff = set(original).symmetric_difference(set(received))
    return len(diff) / len(original)

def total_ser(S, S_hat):
    weights = {
        "objects": 0.3,
        "actions": 0.3,
        "relations": 0.2,
        "context": 0.2
    }
    total = 0.0
    for k in S:
        total += weights[k] * set_ser(S[k], S_hat[k])
    return total

# =========================
# Phase 6: Repair + Recompute SER
# =========================

def repair_semantics(S_hat):
    prompt = f"""
Repair missing semantics.
Return ONLY JSON with keys:
objects, actions, relations, context.

Input:
{json.dumps(S_hat)}
"""

    response = generate(prompt)

    print("\n=== RAW REPAIR RESPONSE ===")
    print(response)

    repaired = extract_json_object(response)
    return repaired

# =========================
# Phase 7: Simulation
# =========================

def simulate():
    random.seed(42)
    bandwidth_limit_tokens = 8
    corruption_p = 0.2

    print("\n=== ORIGINAL ===")
    S = semantic_source()
    print(json.dumps(S, indent=2))

    scored = score_importance_batch(S)

    print("\n=== SCORED ELEMENTS ===")
    for s in scored:
        print(s)

    S_pruned = prune_semantics(S, scored, bandwidth_limit_tokens)

    print("\n=== PRUNED ===")
    print(json.dumps(S_pruned, indent=2))

    S_hat = corrupt(S_pruned, corruption_p)

    print("\n=== CHANNEL OUTPUT ===")
    print(json.dumps(S_hat, indent=2))

    ser_before = total_ser(S_pruned, S_hat)
    print("\nSER BEFORE REPAIR:", ser_before)

    S_repaired = repair_semantics(S_hat)

    print("\n=== REPAIRED STRUCTURE ===")
    print(json.dumps(S_repaired, indent=2))

    ser_after = total_ser(S_pruned, S_repaired)
    print("\nSER AFTER REPAIR:", ser_after)

if __name__ == "__main__":
    simulate()


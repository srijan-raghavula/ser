import random
import math
import torch
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer

# =========================
# Phase 0: GPU + Model Load
# =========================

assert torch.cuda.is_available(), "CUDA not available. Fix torch install."

device = torch.device("cuda")

MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

model.eval()

def generate(prompt, max_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

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
# Phase 2: Batched Importance Scoring
# =========================

def score_importance_batch(S):
    flat_elements = []
    for category in S:
        for element in S[category]:
            flat_elements.append((category, element))

    prompt = f"""
You are ranking semantic importance for disaster response.
Return JSON list of scores between 0 and 1.

Elements:
{flat_elements}

Output format:
[0.95, 0.8, ...]
Only output JSON list.
"""

    response = generate(prompt, max_tokens=200)

    try:
        scores = eval(response.strip())
        scored = []
        for i, (cat, elem) in enumerate(flat_elements):
            score = float(scores[i]) if i < len(scores) else 0.5
            score = max(0.0, min(1.0, score))
            scored.append((cat, elem, score))
        return scored
    except:
        return [(cat, elem, 0.5) for cat, elem in flat_elements]

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
# Phase 4: Channel Corruption
# =========================

def corrupt(S, p):
    S_hat = {}
    for k in S:
        S_hat[k] = [x for x in S[k] if random.random() > p]
    return S_hat

# =========================
# Phase 5: SER Computation
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
# Phase 6: Self-Consistency Repair
# =========================

def repair_with_self_consistency(S_hat, trials=3):
    best = None
    best_len = float("inf")

    for _ in range(trials):
        prompt = f"""
Repair missing disaster semantics.
Return structured JSON-like content.

Input:
{S_hat}
"""
        output = generate(prompt, max_tokens=200)

        if len(output) < best_len:
            best_len = len(output)
            best = output

    return best

# =========================
# Phase 7: Simulation
# =========================

def simulate():
    random.seed(42)

    bandwidth_limit_tokens = 8
    corruption_p = 0.2

    print("\n=== ORIGINAL SEMANTICS ===")
    S = semantic_source()
    print(S)

    print("\n=== IMPORTANCE SCORING (GPU) ===")
    scored = score_importance_batch(S)
    for item in scored:
        print(item)

    print("\n=== GREEDY PRUNED SEMANTICS ===")
    S_pruned = prune_semantics(S, scored, bandwidth_limit_tokens)
    print(S_pruned)

    print("\n=== CHANNEL OUTPUT ===")
    S_hat = corrupt(S_pruned, corruption_p)
    print(S_hat)

    print("\n=== SER BEFORE REPAIR ===")
    ser_val = total_ser(S_pruned, S_hat)
    print("SER:", ser_val)

    print("\n=== SELF-CONSISTENT REPAIR (GPU) ===")
    repaired = repair_with_self_consistency(S_hat)
    print(repaired)

# =========================
# Execution
# =========================

if __name__ == "__main__":
    simulate()

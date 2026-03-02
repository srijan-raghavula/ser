import random
import math
import torch
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer

# =========================
# Phase 0: Load Phi-3
# =========================

MODEL_NAME = "microsoft/phi-3-mini-128k-instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

def phi_generate(prompt, max_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.7
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
            "low visibility"
        ]
    }

# =========================
# Phase 2: Importance Scoring (Phi-3)
# =========================

def score_importance(category, element, full_context):
    prompt = f"""
You are scoring importance for disaster response.

Full context:
{full_context}

Element category: {category}
Element: {element}

Give a single importance score between 0 and 1.
Only output the number.
"""
    response = phi_generate(prompt, max_tokens=20)
    try:
        score = float(response.strip().split()[-1])
        return max(0.0, min(1.0, score))
    except:
        return 0.5

def compute_importance(S):
    context_text = str(S)
    scored = []

    for category in S:
        for element in S[category]:
            score = score_importance(category, element, context_text)
            scored.append((category, element, score))

    return scored

# =========================
# Phase 3: Greedy SER-per-GB Pruning
# =========================

def prune_semantics(S, scored_elements, max_tokens):
    S_new = deepcopy(S)
    total_elements = sum(len(v) for v in S.values())

    if total_elements <= max_tokens:
        return S_new

    scored_elements.sort(key=lambda x: x[2])  # ascending importance

    to_remove = total_elements - max_tokens

    for i in range(to_remove):
        category, element, _ = scored_elements[i]
        if element in S_new[category]:
            S_new[category].remove(element)

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
# Phase 6: Self-Consistency Reconstruction
# =========================

def reconstruct_with_self_consistency(S_hat, trials=3):
    best_output = None
    best_score = float("inf")

    for _ in range(trials):
        prompt = f"""
Reconstruct missing disaster scene details based on:

{S_hat}

Return structured JSON-like content.
"""
        generated = phi_generate(prompt, max_tokens=200)

        # simplistic proxy: fewer hallucinated tokens = better
        score = len(generated)

        if score < best_score:
            best_score = score
            best_output = generated

    return best_output

# =========================
# Phase 7: Full Simulation
# =========================

def simulate():
    random.seed(42)

    bandwidth_limit_tokens = 8
    corruption_p = 0.2

    print("\n=== ORIGINAL SEMANTICS ===")
    S = semantic_source()
    print(S)

    print("\n=== IMPORTANCE SCORING (Phi-3) ===")
    scored = compute_importance(S)
    for item in scored:
        print(item)

    print("\n=== GREEDY PRUNED SEMANTICS ===")
    S_pruned = prune_semantics(S, scored, bandwidth_limit_tokens)
    print(S_pruned)

    print("\n=== CHANNEL OUTPUT ===")
    S_hat = corrupt(S_pruned, corruption_p)
    print(S_hat)

    print("\n=== SER BEFORE REPAIR ===")
    ser_value = total_ser(S_pruned, S_hat)
    print("SER:", ser_value)

    print("\n=== SELF-CONSISTENT RECONSTRUCTION ===")
    reconstructed = reconstruct_with_self_consistency(S_hat)
    print(reconstructed)

# =========================
# Execution
# =========================

if __name__ == "__main__":
    simulate()


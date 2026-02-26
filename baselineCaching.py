import random
import math
from copy import deepcopy

# =========================
# Phase 1: Semantic Source
# =========================

def semantic_source():
    return {
        "objects": {
            "3-story residential building",
            "fire on 3rd floor east window",
            "2 trapped civilians",
            "fire truck",
            "smoke plume"
        },
        "actions": {
            "burning intensely",
            "civilians waving for help",
            "firefighters deploying ladder"
        },
        "relations": {
            "fire located on east facade",
            "civilians inside third floor",
            "ladder positioned below window"
        },
        "context": {
            "urban disaster response",
            "evening time",
            "low visibility due to smoke"
        }
    }

# =========================
# Phase 2: Channel Model
# =========================

def corrupt_set(p, s):
    result = set()
    for x in s:
        if random.random() > p:
            result.add(x)
    return result

def channel(p):
    def apply_channel(S):
        corrupted = {}
        for k, v in S.items():
            corrupted[k] = corrupt_set(p, v)
        return corrupted
    return apply_channel

# =========================
# Phase 3: SER Computation
# =========================

def set_ser(original, received):
    if len(original) == 0:
        return 0.0
    diff = original.symmetric_difference(received)
    return len(diff) / len(original)

def detailed_ser(S, S_hat, weights):
    component_ser = {}
    total = 0.0

    for k in S:
        ser_val = set_ser(S[k], S_hat[k])
        component_ser[k] = ser_val
        total += weights[k] * ser_val

    return total, component_ser

# =========================
# Phase 4: Model
# =========================

class Model:
    def __init__(self, name, quality_factor, gen_latency):
        self.name = name
        self.quality_factor = quality_factor
        self.gen_latency = gen_latency

# =========================
# Phase 5: Transmission
# =========================

def transmission_rate(b, gamma):
    return b * math.log2(1 + gamma)

def transmission_latency(payload_size, rate):
    return payload_size / rate if rate > 0 else float("inf")

# =========================
# Phase 6: Single Simulation Step
# =========================

def simulate():
    random.seed(42)

    weights = {
        "objects": 0.3,
        "actions": 0.3,
        "relations": 0.2,
        "context": 0.2
    }

    bandwidth = 10
    sinr = 15
    corruption_p = 0.2

    model = Model("EdgeGen-v1", quality_factor=0.15, gen_latency=5)

    print("\n=== ORIGINAL SEMANTICS (S) ===")
    S = semantic_source()
    for k, v in S.items():
        print(f"{k}:")
        for item in v:
            print("  -", item)

    print("\n=== CHANNEL CORRUPTION ===")
    channel_fn = channel(corruption_p)
    S_hat = channel_fn(S)

    print("\n=== RECEIVED SEMANTICS (S_hat) ===")
    for k, v in S_hat.items():
        print(f"{k}:")
        for item in v:
            print("  -", item)

    print("\n=== SER CALCULATION ===")
    total_ser, component_ser = detailed_ser(S, S_hat, weights)

    for k in component_ser:
        print(f"{k} SER:", component_ser[k])

    print("Weighted SER (before model quality):", total_ser)

    effective_ser = total_ser * (1 - model.quality_factor)
    print("Effective SER (after model quality factor):", effective_ser)

    print("\n=== TRANSMISSION CALCULATION ===")
    token_size = sum(len(v) for v in S.values())
    rate = transmission_rate(bandwidth, sinr)
    tx_latency = transmission_latency(token_size, rate)

    print("Token size:", token_size)
    print("Transmission rate:", rate)
    print("Transmission latency:", tx_latency)
    print("Generation latency:", model.gen_latency)
    print("Total latency:", tx_latency + model.gen_latency)


if __name__ == "__main__":
    simulate()

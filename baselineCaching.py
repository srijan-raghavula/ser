import random
import math
from functools import partial
from copy import deepcopy

# =========================
# Phase 1: Semantic Source
# =========================

def semantic_source():
    return {
        "objects": {"building", "fire", "civilian"},
        "actions": {"burning", "trapped"},
        "relations": {"fire-east", "civilian-inside"},
        "context": {"disaster-response"}
    }

# =========================
# Phase 2: Channel Model
# =========================

def corrupt_set(p, s):
    return {x for x in s if random.random() > p}

def channel(p):
    def apply_channel(S):
        return {k: corrupt_set(p, v) for k, v in S.items()}
    return apply_channel

# =========================
# Phase 3: SER Computation
# =========================

def set_ser(original, received):
    if len(original) == 0:
        return 0.0
    return len(original.symmetric_difference(received)) / len(original)

def weighted_ser(weights):
    def compute(S, S_hat):
        return sum(
            weights[k] * set_ser(S[k], S_hat[k])
            for k in S
        )
    return compute

# =========================
# Phase 4: Model & Cache
# =========================

class Model:
    def __init__(self, name, size, quality_factor, gen_latency):
        self.name = name
        self.size = size
        self.quality_factor = quality_factor
        self.gen_latency = gen_latency

def baseline_cache(models, capacity):
    used = 0
    cache = {}
    for m in models:
        if used + m.size <= capacity:
            cache[m.name] = m
            used += m.size
    return cache

# =========================
# Phase 5: Transmission Rate
# =========================

def transmission_rate(b, gamma):
    return b * math.log2(1 + gamma)

def transmission_latency(payload_size, rate):
    if rate == 0:
        return float("inf")
    return payload_size / rate

# =========================
# Phase 6: End-to-End Step
# =========================

def simulate_step(
    source_fn,
    channel_fn,
    ser_fn,
    model,
    bandwidth,
    sinr,
    token_size
):
    S = source_fn()
    S_hat = channel_fn(S)

    ser_value = ser_fn(S, S_hat)

    rate = transmission_rate(bandwidth, sinr)
    tx_latency = transmission_latency(token_size, rate)

    gen_latency = model.gen_latency
    total_latency = tx_latency + gen_latency

    effective_ser = ser_value * (1 - model.quality_factor)

    return effective_ser, total_latency

# =========================
# Phase 7: Simulation Loop
# =========================

def run_simulation(
    T,
    source_fn,
    channel_fn,
    ser_fn,
    cached_model,
    bandwidth,
    sinr,
    token_size
):
    total_ser = 0.0
    total_latency = 0.0

    for _ in range(T):
        ser, latency = simulate_step(
            source_fn,
            channel_fn,
            ser_fn,
            cached_model,
            bandwidth,
            sinr,
            token_size
        )
        total_ser += ser
        total_latency += latency

    return total_ser / T, total_latency / T

# =========================
# Execution
# =========================

if __name__ == "__main__":
    random.seed(42)

    models = [
        Model("M1", size=5, quality_factor=0.1, gen_latency=5),
        Model("M2", size=7, quality_factor=0.2, gen_latency=7),
        Model("M3", size=10, quality_factor=0.3, gen_latency=10),
    ]

    capacity = 12
    cache = baseline_cache(models, capacity)

    selected_model = list(cache.values())[0]

    weights = {
        "objects": 0.3,
        "actions": 0.3,
        "relations": 0.2,
        "context": 0.2
    }

    source_fn = semantic_source
    channel_fn = channel(p=0.2)
    ser_fn = weighted_ser(weights)

    avg_ser, avg_latency = run_simulation(
        T=100,
        source_fn=source_fn,
        channel_fn=channel_fn,
        ser_fn=ser_fn,
        cached_model=selected_model,
        bandwidth=10,
        sinr=15,
        token_size=50
    )

    print("Average SER:", avg_ser)
    print("Average Latency:", avg_latency)

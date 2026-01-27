import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Suppress warnings for cleaner output in CLI
warnings.filterwarnings("ignore")

class SemanticErrorScanner:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the Semantic Encoder (Phi function).
        Ref: "SER can be computed using Semantic embeddings" [Equation 8 context]
        """
        print(f"Loading Semantic Model: {model_name} (approx 90MB)...")
        # This downloads the model once, then caches it locally.
        self.model = SentenceTransformer(model_name)
        print("Model Loaded. Ready to scan.")

    def calculate_overall_ser(self, original_text, reconstructed_text):
        """
        Implements Equation (8): SER = 1 - CosineSimilarity(Enc(Tx), Enc(Rx))
        """
        # 1. Encode text to vector embeddings (Phi function)
        # Result is a list of vectors
        embeddings = self.model.encode([original_text, reconstructed_text])

        # 2. Calculate Cosine Similarity
        # returns a matrix [[1, sim], [sim, 1]], we want [0][1]
        sim_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

        # 3. Calculate SER (Error = 1 - Similarity)
        ser_score = 1.0 - sim_score

        # Clip to 0-1 range just in case of float weirdness
        return max(0.0, min(1.0, ser_score))

    def calculate_object_ser(self, original_text, reconstructed_text):
        """
        Implements Equation (10): SER_obj = |Org Delta Rec| / |Org|
        (Symmetric Difference over Original Size)
        """
        # A simple tokenizer. For better results, use Spacy later.
        # This cleans punctuation and splits by space.
        def simple_tokenize(text):
            clean = text.lower().replace(',', '').replace('.', '')
            return set(clean.split())

        org_tokens = simple_tokenize(original_text)
        rec_tokens = simple_tokenize(reconstructed_text)

        if len(org_tokens) == 0:
            return 0.0 # Avoid division by zero

        # The "Triangle" Operator (Symmetric Difference)
        # Items in A not in B, PLUS items in B not in A
        diff_set = org_tokens.symmetric_difference(rec_tokens)

        # Equation 10 calculation
        ser_obj = len(diff_set) / len(org_tokens)

        return ser_obj, diff_set

# --- SIMULATION AREA ---
if __name__ == "__main__":
    scanner = SemanticErrorScanner()

    # Example from your PDF [Source: 88]
    # "Collapsed residential building, 3 floors, fire on east side, trapped civilians detected."
    tx_text = "Collapsed residential building 3 floors fire on east side trapped civilians detected"

    # Case 1: Minor Error (Semantically similar, different words)
    rx_good = "Residential building collapse three stories high eastern fire with people trapped"

    # Case 2: Major Error (Hallucination / Meaning Loss)
    rx_bad = "A residential building is under construction 3 floors sunny day"

    print("-" * 50)
    print(f"ORIGINAL: {tx_text}")
    print("-" * 50)

    # TEST 1
    print(f"\n[Test 1] Comparing with: '{rx_good}'")
    ser_1 = scanner.calculate_overall_ser(tx_text, rx_good)
    obj_ser_1, diff_1 = scanner.calculate_object_ser(tx_text, rx_good)

    print(f" > Overall SER (Eq 8): {ser_1:.4f} (Lower is better)")
    print(f" > Object  SER (Eq 10): {obj_ser_1:.4f}")
    print(f" > Differences: {diff_1}")

    # TEST 2
    print(f"\n[Test 2] Comparing with: '{rx_bad}'")
    ser_2 = scanner.calculate_overall_ser(tx_text, rx_bad)
    obj_ser_2, diff_2 = scanner.calculate_object_ser(tx_text, rx_bad)

    print(f" > Overall SER (Eq 8): {ser_2:.4f}")
    print(f" > Object  SER (Eq 10): {obj_ser_2:.4f}")
    print(f" > Differences: {diff_2}")

"""
Unit + end-to-end tests for fq_core.
Run: python -m pytest test_fq_core.py -v
Or:  python test_fq_core.py
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fq_core import (
    expand_bytes, kgram_counts, debiased_fq, shannon_entropy,
    multiscale_fq, fingerprint, nearest_centroid_classify, ari_score,
    markov_true_h2, markov_true_hq, generate_markov, generate_markov_order_k
)

PASS, FAIL, TOTAL = 0, 0, 0

def check(name, cond, detail=""):
    global PASS, FAIL, TOTAL
    TOTAL += 1
    if cond:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name} — {detail}")

# ========== UNIT TESTS ==========

def test_expand_bytes():
    print("\n[UNIT] expand_bytes")
    d = bytes([0xAB])  # 10101011
    # alpha=256: identity
    r = expand_bytes(d, 256)
    check("alpha=256 identity", len(r) == 1 and r[0] == 0xAB)
    # alpha=16: [A, B]
    r = expand_bytes(d, 16)
    check("alpha=16 nibbles", len(r) == 2 and r[0] == 0xA and r[1] == 0xB, f"got {r}")
    # alpha=4: [2, 2, 2, 3]  (10, 10, 10, 11)
    r = expand_bytes(d, 4)
    check("alpha=4 2-bit", list(r) == [2, 2, 2, 3], f"got {list(r)}")

def test_kgram_counts():
    print("\n[UNIT] kgram_counts")
    # 4 bytes: [0,1,0,1] → bigrams: 01, 10, 01 → counts[1]=2, counts[256]=1
    d = bytes([0, 1, 0, 1])
    c = kgram_counts(d, 256, 1)
    check("unigram sum", c.sum() == 4)
    check("unigram values", c[0] == 2 and c[1] == 2, f"c[0]={c[0]}, c[1]={c[1]}")
    c2 = kgram_counts(d, 256, 2)
    check("bigram sum", c2.sum() == 3)
    # bigrams: (0,1)=idx 1, (1,0)=idx 256, (0,1)=idx 1
    check("bigram 01", c2[1] == 2, f"c2[1]={c2[1]}")
    check("bigram 10", c2[256] == 1, f"c2[256]={c2[256]}")

def test_debiased_fq():
    print("\n[UNIT] debiased_fq")
    # uniform: n items in n bins → F₂ = (n·1-1)/(n-1) = 1... wait
    # Actually for n=100 items uniformly in 100 bins (1 each):
    # F̂₂ = Σ n_i(n_i-1) / n(n-1) = 0/(100·99) = 0
    # That's correct: truly uniform has F₂→1/d as d→∞
    # For n=1000 in 256 bins, ~4 per bin:
    rng = np.random.default_rng(42)
    counts = np.zeros(256, dtype=np.int64)
    np.add.at(counts, rng.integers(0, 256, size=10000), 1)
    f2 = debiased_fq(counts, 2)
    expected = 1.0 / 256  # for uniform
    check("F2 near 1/256 for uniform", abs(f2 - expected) < 0.001, f"F2={f2:.6f}, expected≈{expected:.6f}")
    # peaked: all in one bin → F₂ = 1
    counts_peaked = np.zeros(256, dtype=np.int64)
    counts_peaked[0] = 1000
    f2p = debiased_fq(counts_peaked, 2)
    check("F2=1 for single bin", abs(f2p - 1.0) < 1e-10, f"F2={f2p}")

def test_shannon():
    print("\n[UNIT] shannon_entropy")
    # uniform over 256 → H₁ = 8.0
    c = np.ones(256, dtype=np.int64) * 100
    h = shannon_entropy(c)
    check("H1=8 for uniform/256", abs(h - 8.0) < 0.01, f"H1={h}")
    # single bin → H₁ = 0
    c0 = np.zeros(256, dtype=np.int64); c0[0] = 1000
    check("H1=0 for constant", abs(shannon_entropy(c0)) < 1e-10)

def test_markov_true_h2():
    print("\n[UNIT] markov_true_h2")
    # independent uniform over S states: P[i,j]=1/S
    # M₂[i,j] = (1/S)² → M₂ = (1/S²)·ones → λ₁ = S·(1/S²) = 1/S
    # h₂ = -log₂(1/S) = log₂(S)
    for S in [2, 4, 8]:
        P = np.ones((S, S)) / S
        h2 = markov_true_h2(P)
        expected = np.log2(S)
        check(f"iid uniform S={S}: h2={h2:.4f}≈{expected:.4f}", abs(h2 - expected) < 0.01)

def test_multiscale_fq_sanity():
    print("\n[UNIT] multiscale_fq sanity")
    rng = np.random.default_rng(42)
    # alpha=16 nibble: 50K bytes → 100K nibbles, k_max=3, h₂≈log₂(16)=4.0
    data = bytes(rng.integers(0, 256, size=50000, dtype=np.uint8))
    res = multiscale_fq(data, alpha=16, q_list=(2,))
    h2 = res['h_q'][2]
    check(f"uniform nib h2={h2:.3f}≈4.0", abs(h2 - 4.0) < 0.2, f"h2={h2}")
    check(f"R²={res['R2']:.6f}≈1 for iid", res['R2'] is not None and res['R2'] > 0.99, f"R²={res['R2']}")
    check(f"k_max={res['k_max_used']}>=2", res['k_max_used'] >= 2, f"k_max={res['k_max_used']}")

    # alpha=256 needs 200K+ for k_max>=2
    data_big = bytes(rng.integers(0, 256, size=200000, dtype=np.uint8))
    res256 = multiscale_fq(data_big, alpha=256, q_list=(2,))
    h2_256 = res256['h_q'][2]
    check(f"uniform byte h2={h2_256:.3f}≈8.0 (n=200K)", abs(h2_256 - 8.0) < 0.3, f"h2={h2_256}")

    # constant: use alpha=4, byte 0x00 → all-zero 2-bit stream → h₂≈0
    data_const = bytes([0] * 20000)
    res_c = multiscale_fq(data_const, alpha=4, q_list=(2,))
    h2c = res_c['h_q'][2]
    check(f"constant h2={h2c:.3f}≈0 (alpha=4)", h2c is not None and abs(h2c) < 0.5, f"h2={h2c}")

def test_ari():
    print("\n[UNIT] ari_score")
    a = np.array([0, 0, 0, 1, 1, 1])
    check("ARI=1 for perfect", abs(ari_score(a, a) - 1.0) < 1e-10)
    b = np.array([1, 1, 1, 0, 0, 0])
    check("ARI=1 for permuted labels", abs(ari_score(a, b) - 1.0) < 1e-10)
    c = np.array([0, 1, 0, 1, 0, 1])
    ari = ari_score(a, c)
    check(f"ARI<0.5 for random-ish: {ari:.3f}", ari < 0.5)

def test_nearest_centroid():
    print("\n[UNIT] nearest_centroid_classify")
    train = np.array([[0, 0], [0, 1], [10, 10], [10, 11]])
    labels = np.array([0, 0, 1, 1])
    test = np.array([[1, 1], [9, 9]])
    pred = nearest_centroid_classify(train, labels, test)
    check("NC classifies [1,1]→0, [9,9]→1", list(pred) == [0, 1], f"got {list(pred)}")

# ========== INTEGRATION TESTS ==========

def test_markov_roundtrip():
    """Generate Markov(1), estimate h₂, compare to true h₂."""
    print("\n[INTEGRATION] Markov roundtrip h₂")
    rng = np.random.default_rng(99)
    for trial, strength in enumerate([0.3, 0.6, 0.9]):
        P = rng.dirichlet(np.ones(4) * (1 / strength), size=4)
        true_h2 = markov_true_h2(P)
        data = generate_markov(P, 50000, seed=42 + trial)
        # pass as numpy array (values 0-3) — NOT through byte expansion
        seq = np.frombuffer(data, dtype=np.uint8)
        res = multiscale_fq(seq, alpha=4, q_list=(2, 3, 4))
        est_h2 = res['h_q'][2]
        err = abs(est_h2 - true_h2)
        check(f"strength={strength}: true={true_h2:.4f}, est={est_h2:.4f}, err={err:.4f}",
              err < 0.05, f"error too large: {err:.4f}")

def test_multiq_separation():
    """Different sources should have distinct (h₂,h₃,h₄) fingerprints."""
    print("\n[INTEGRATION] Multi-q separation")
    rng = np.random.default_rng(42)
    sources = {
        'uniform': bytes(rng.integers(0, 256, 50000, dtype=np.uint8)),
        'biased': bytes(rng.choice([0, 1, 2, 3], size=50000, p=[0.7, 0.1, 0.1, 0.1]).astype(np.uint8)),
        'constant': bytes([0] * 50000),
    }
    fps = {}
    for name, data in sources.items():
        fp = fingerprint(data, alpha=16, q_list=(2, 3, 4))  # nibble mode for k_max>=2
        fps[name] = fp
        print(f"    {name}: h₂={fp[0]:.3f}, h₃={fp[1]:.3f}, h₄={fp[2]:.3f}")
    check("h₂ ordering", fps['uniform'][0] > fps['biased'][0] > fps['constant'][0],
          f"{fps['uniform'][0]:.3f} > {fps['biased'][0]:.3f} > {fps['constant'][0]:.3f}")

def test_sqrt_n_consistency():
    """RMSE × √n should be bounded as n grows."""
    print("\n[INTEGRATION] √n consistency (quick)")
    P = np.array([[0.7, 0.3], [0.4, 0.6]])
    true_h2 = markov_true_h2(P)
    n_reps = 10
    for n in [1000, 5000, 20000]:
        errors = []
        for rep in range(n_reps):
            data = generate_markov(P, n, seed=rep * 100)
            seq = np.frombuffer(data, dtype=np.uint8)  # values in {0,1}
            res = multiscale_fq(seq, alpha=2, q_list=(2,))
            errors.append(res['h_q'][2] - true_h2)
        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        scaled = rmse * np.sqrt(n)
        check(f"n={n:5d}: RMSE={rmse:.4f}, RMSE×√n={scaled:.2f} (should be O(1))",
              scaled < 5.0, f"scaled={scaled:.2f}")

# ========== E2E TEST ==========

def test_e2e_classification():
    """Full pipeline: generate 3 source types, fingerprint, classify, check ARI."""
    print("\n[E2E] Classification pipeline")
    rng = np.random.default_rng(42)
    all_fps, all_labels = [], []
    for cls, (alpha_src, order, strength) in enumerate([
        (4, 0, 0.0),   # iid
        (4, 1, 0.9),   # Markov-1
        (4, 2, 0.9),   # Markov-2
    ]):
        for rep in range(10):
            data = generate_markov_order_k(4, order, strength, 50000, seed=cls * 1000 + rep)
            seq = np.frombuffer(data, dtype=np.uint8)
            res = multiscale_fq(seq, alpha=4, q_list=(2, 3, 4))
            fp = np.array([res['h_q'].get(q, np.nan) for q in (2, 3, 4)])
            all_fps.append(fp)
            all_labels.append(cls)
    fps_arr = np.array(all_fps)
    labels_arr = np.array(all_labels)
    # LOO classification
    correct = 0
    for i in range(len(fps_arr)):
        mask = np.ones(len(fps_arr), dtype=bool); mask[i] = False
        pred = nearest_centroid_classify(fps_arr[mask], labels_arr[mask], fps_arr[i:i+1])
        if pred[0] == labels_arr[i]:
            correct += 1
    acc = correct / len(fps_arr)
    ari = ari_score(labels_arr, nearest_centroid_classify(fps_arr, labels_arr, fps_arr))
    check(f"LOO accuracy={acc:.2f} (>0.6)", acc > 0.6, f"acc={acc}")
    check(f"ARI={ari:.3f} (>0.3)", ari > 0.3, f"ari={ari}")


if __name__ == '__main__':
    print("=" * 60)
    print("fq_core test suite")
    print("=" * 60)
    test_expand_bytes()
    test_kgram_counts()
    test_debiased_fq()
    test_shannon()
    test_markov_true_h2()
    test_multiscale_fq_sanity()
    test_ari()
    test_nearest_centroid()
    test_markov_roundtrip()
    test_multiq_separation()
    test_sqrt_n_consistency()
    test_e2e_classification()
    print("\n" + "=" * 60)
    print(f"RESULTS: {PASS}/{TOTAL} passed, {FAIL} failed")
    print("=" * 60)
    sys.exit(1 if FAIL > 0 else 0)

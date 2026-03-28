"""
Multi-scale Rényi entropy rate estimation via collision counting.

Core identity: Ev = sqrt(F₂/d), F₂ = Σp² = Friedman IC (1922) = Rényi H₂.
Multi-scale: log F₂(k) vs k is linear with slope log(λ), h₂ = -log₂(λ).
Debiased: F̂_q = Σ n_i^{(q)} / n^{(q)} via falling factorial.

Reference: Rényi (1961), Friedman (1922), Gecchele (2023 IACR single-scale).
Our contribution: multi-scale slope → h_q, R² → Markov order, (h₂,h₃,h₄) → classification.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional

# === BYTE EXPANSION ===

def expand_bytes(data: bytes, alpha: int) -> np.ndarray:
    """Expand byte stream to alphabet of size alpha.
    alpha=256: identity. alpha=16: [hi_nib, lo_nib]. alpha=4: four 2-bit values.
    alpha=2: eight bits per byte.
    """
    raw = np.frombuffer(data, dtype=np.uint8)
    if alpha == 256:
        return raw
    elif alpha == 16:
        out = np.empty(2 * len(raw), dtype=np.uint8)
        out[0::2] = raw >> 4
        out[1::2] = raw & 0x0F
        return out
    elif alpha == 4:
        out = np.empty(4 * len(raw), dtype=np.uint8)
        out[0::4] = (raw >> 6) & 0x03
        out[1::4] = (raw >> 4) & 0x03
        out[2::4] = (raw >> 2) & 0x03
        out[3::4] = raw & 0x03
        return out
    elif alpha == 2:
        out = np.empty(8 * len(raw), dtype=np.uint8)
        for bit in range(8):
            out[bit::8] = (raw >> (7 - bit)) & 0x01
        return out
    raise ValueError(f"Unsupported alpha={alpha}. Use 2, 4, 16, or 256.")


def kgram_counts(data, alpha: int, k: int) -> np.ndarray:
    """Count k-grams. data: bytes OR np.ndarray of ints in [0, alpha).
    Returns array of length alpha^k."""
    if isinstance(data, (bytes, bytearray)):
        seq = expand_bytes(data, alpha)
    else:
        seq = np.asarray(data, dtype=np.uint8)
    d = alpha ** k
    n = len(seq)
    assert n >= k, f"Sequence length {n} < k={k}"
    idx = np.zeros(n - k + 1, dtype=np.int64)
    for j in range(k):
        idx += seq[j:n - k + 1 + j].astype(np.int64) * (alpha ** (k - 1 - j))
    counts = np.zeros(d, dtype=np.int64)
    np.add.at(counts, idx, 1)
    return counts


# === DEBIASED F_q ESTIMATOR ===

def debiased_fq(counts: np.ndarray, q: int) -> float:
    """Debiased F_q via falling factorial: F̂_q = Σ n_i^{(q)} / n^{(q)}.
    n^{(q)} = n(n-1)...(n-q+1). Unbiased for multinomial.
    Gecchele 2023 IACR: q=2 single-scale. We extend to arbitrary q, multi-scale.
    """
    n = int(counts.sum())
    assert n >= q, f"n={n} < q={q}"
    # numerator: Σ_i n_i(n_i-1)...(n_i-q+1)
    ff = counts.astype(np.float64).copy()
    for j in range(1, q):
        ff *= np.maximum(counts - j, 0)  # falling factorial per bin
    # denominator: n(n-1)...(n-q+1)
    denom = 1.0
    for j in range(q):
        denom *= (n - j)
    assert denom > 0, f"denom=0 for n={n}, q={q}"
    return float(ff.sum() / denom)


def shannon_entropy(counts: np.ndarray) -> float:
    """Shannon entropy H₁ = -Σ p log₂ p."""
    n = counts.sum()
    if n == 0:
        return 0.0
    p = counts[counts > 0] / n
    return float(-np.sum(p * np.log2(p)))


# === MULTI-SCALE ENTROPY RATE ===

def multiscale_fq(data, alpha: int = 256, q_list: tuple = (2, 3, 4),
                  k_max: Optional[int] = None, dim_ratio_limit: float = 0.5) -> dict:
    """Compute multi-scale F_q and extract entropy rates via OLS slope.

    data: bytes/bytearray OR np.ndarray of ints in [0, alpha).
    h_q = -slope_q / ((q-1) * ln(2))   [from Rényi entropy rate definition]

    Adaptive k_max: largest k s.t. α^k / n_kgrams < dim_ratio_limit.

    Returns:
        h_q: {q: float} — entropy rate per Rényi order
        slopes: {q: float} — OLS slopes of log F_q(k) vs k
        R2: float — R² of log F₂(k) fit (Markov order proxy)
        log_fq: {q: [(k, log_fq_k), ...]}
        k_max_used: int
        n_bytes: int
    """
    if isinstance(data, (bytes, bytearray)):
        n_bytes = len(data)
        seq = expand_bytes(data, alpha)
    else:
        seq = np.asarray(data, dtype=np.uint8)
        n_bytes = len(seq)
    n_seq = len(seq)

    # adaptive k_max
    if k_max is None:
        k_max = 1
        while True:
            next_k = k_max + 1
            d_next = alpha ** next_k
            n_kgrams_next = n_seq - next_k + 1
            if n_kgrams_next < max(q_list) + 1:
                break
            if d_next / n_kgrams_next > dim_ratio_limit:
                break
            k_max = next_k
            if k_max >= 8:  # hard cap
                break

    log_fq = {q: [] for q in q_list}

    for k in range(1, k_max + 1):
        counts = kgram_counts(data, alpha, k)
        n_kgrams = int(counts.sum())
        for q in q_list:
            if n_kgrams < q + 1:
                continue
            fq = debiased_fq(counts, q)
            if fq > 0 and np.isfinite(np.log(fq)):
                log_fq[q].append((k, float(np.log(fq))))

    # OLS fit per q
    h_q, slopes, R2 = {}, {}, None
    for q in q_list:
        pts = log_fq[q]
        if len(pts) < 2:
            h_q[q] = np.nan
            slopes[q] = np.nan
            continue
        ks = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        # OLS: y = slope*k + intercept
        A = np.vstack([ks, np.ones_like(ks)]).T
        coeff, res, _, _ = np.linalg.lstsq(A, ys, rcond=None)
        slope, intercept = coeff
        # h_q = -slope / ((q-1) * ln(2))
        h_q[q] = float(-slope / ((q - 1) * np.log(2)))
        slopes[q] = float(slope)
        # R² for q=2
        if q == 2:
            y_pred = slope * ks + intercept
            ss_res = np.sum((ys - y_pred) ** 2)
            ss_tot = np.sum((ys - np.mean(ys)) ** 2)
            R2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-30 else 1.0

    return {
        'h_q': {int(q): float(v) for q, v in h_q.items()},
        'slopes': {int(q): float(v) for q, v in slopes.items()},
        'R2': R2,
        'log_fq': {int(q): v for q, v in log_fq.items()},
        'k_max_used': k_max,
        'n_bytes': n_bytes
    }


# === CLASSIFICATION ===

def fingerprint(data, alpha: int = 256, q_list: tuple = (2, 3, 4), mode: str = 'auto') -> np.ndarray:
    """Extract (h₂, h₃, h₄) fingerprint vector.

    mode='auto': use multi-scale slope if k_max>=2, else single-scale.
    mode='single': always use single-scale h_q = -log₂(F_q(k=1))/(q-1).
    mode='slope': always use multi-scale OLS slope.
    """
    if mode == 'single' or (mode == 'auto' and alpha == 256):
        # Single-scale: h_q from F_q at k=1 directly
        # h_q = (1/(1-q)) * log₂(F_q)  [Rényi entropy, not rate]
        counts = kgram_counts(data, alpha, 1)
        n_k = int(counts.sum())
        result = []
        for q in q_list:
            if n_k < q + 1:
                result.append(np.nan)
                continue
            fq = debiased_fq(counts, q)
            if fq > 0 and np.isfinite(np.log2(fq)):
                hq = float(np.log2(fq) / (1 - q))
                result.append(hq)
            else:
                result.append(np.nan)
        return np.array(result)
    else:
        # Multi-scale slope method
        res = multiscale_fq(data, alpha, q_list)
        return np.array([res['h_q'].get(q, np.nan) for q in q_list])


def nearest_centroid_classify(train_fps: np.ndarray, train_labels: np.ndarray,
                              test_fps: np.ndarray) -> np.ndarray:
    """Nearest-centroid classifier. No sklearn dependency."""
    classes = np.unique(train_labels)
    centroids = np.array([train_fps[train_labels == c].mean(axis=0) for c in classes])
    # L2 distance
    dists = np.array([[np.linalg.norm(fp - c) for c in centroids] for fp in test_fps])
    return classes[np.argmin(dists, axis=1)]


def ari_score(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Adjusted Rand Index. No sklearn dependency."""
    n = len(labels_true)
    assert n == len(labels_pred)
    # contingency table
    classes_t = np.unique(labels_true)
    classes_p = np.unique(labels_pred)
    ct = np.zeros((len(classes_t), len(classes_p)), dtype=np.int64)
    t_map = {c: i for i, c in enumerate(classes_t)}
    p_map = {c: i for i, c in enumerate(classes_p)}
    for i in range(n):
        ct[t_map[labels_true[i]], p_map[labels_pred[i]]] += 1
    # ARI formula
    a = ct.sum(axis=1)
    b = ct.sum(axis=0)
    comb2 = lambda x: x * (x - 1) // 2
    sum_comb_ct = sum(comb2(ct[i, j]) for i in range(ct.shape[0]) for j in range(ct.shape[1]))
    sum_comb_a = sum(comb2(ai) for ai in a)
    sum_comb_b = sum(comb2(bi) for bi in b)
    comb_n = comb2(n)
    if comb_n == 0:
        return 0.0
    expected = sum_comb_a * sum_comb_b / comb_n
    max_idx = (sum_comb_a + sum_comb_b) / 2
    denom = max_idx - expected
    if abs(denom) < 1e-12:
        return 1.0 if abs(sum_comb_ct - expected) < 1e-12 else 0.0
    return float((sum_comb_ct - expected) / denom)


# === GROUND TRUTH FOR MARKOV CHAINS ===

def markov_true_h2(P: np.ndarray) -> float:
    """True Rényi 2-entropy rate for Markov(1) with transition matrix P.
    h₂ = -log₂(λ₁(M₂)) where M₂[i,j] = P[i,j]².
    """
    M2 = P ** 2  # element-wise square
    eigs = np.abs(np.linalg.eigvals(M2))
    lam = np.max(eigs)
    assert lam > 0, "Degenerate transition matrix"
    return float(-np.log2(lam))


def markov_true_hq(P: np.ndarray, q: int) -> float:
    """True Rényi q-entropy rate: h_q = (1/(1-q)) * log₂(λ₁(M_q)), M_q[i,j]=P[i,j]^q."""
    Mq = P ** q
    eigs = np.abs(np.linalg.eigvals(Mq))
    lam = np.max(eigs)
    assert lam > 0
    return float(np.log2(lam) / (1 - q))


def generate_markov(P: np.ndarray, n: int, seed: int = 42) -> bytes:
    """Generate n bytes from Markov(1) chain with transition matrix P."""
    rng = np.random.default_rng(seed)
    S = P.shape[0]
    assert S <= 256, "Alphabet must fit in byte"
    # stationary distribution
    evals, evecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(evals - 1.0))
    pi = np.real(evecs[:, idx])
    pi = pi / pi.sum()
    pi = np.abs(pi)
    # generate
    seq = np.zeros(n, dtype=np.uint8)
    seq[0] = rng.choice(S, p=pi)
    for i in range(1, n):
        seq[i] = rng.choice(S, p=P[seq[i - 1]])
    return bytes(seq)


def generate_markov_order_k(alpha: int, order: int, strength: float, n: int, seed: int = 42) -> bytes:
    """Generate bytes from Markov chain of specified order.
    State space = alpha^order. Transition: with prob `strength`, next byte depends on
    full context; with prob 1-strength, uniform random.
    """
    rng = np.random.default_rng(seed)
    assert alpha <= 256
    S = alpha ** order if order > 0 else 1
    seq = np.zeros(n, dtype=np.uint8)
    seq[:max(order, 1)] = rng.integers(0, alpha, size=max(order, 1))

    if order == 0:
        # iid with slight bias
        probs = rng.dirichlet(np.ones(alpha) * 10)
        seq = rng.choice(alpha, size=n, p=probs).astype(np.uint8)
        return bytes(seq)

    # build context-dependent distributions
    ctx_probs = {}
    for ctx in range(S):
        base = rng.dirichlet(np.ones(alpha) * 5)
        ctx_probs[ctx] = (1 - strength) * np.ones(alpha) / alpha + strength * base

    for i in range(order, n):
        ctx = 0
        for j in range(order):
            ctx = ctx * alpha + int(seq[i - order + j])
        ctx = ctx % S
        seq[i] = rng.choice(alpha, p=ctx_probs[ctx])
    return bytes(seq)

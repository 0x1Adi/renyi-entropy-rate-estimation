#!/usr/bin/env python3
"""
MEGA VALIDATION EXPERIMENT — Tests every paper claim.

Critical ablation included: Shannon entropy RATE at α=16 vs Rényi h₂ at α=16.
This answers: is the improvement from multi-scale slope, or from Rényi specifically?

Usage:
    python mega_validation.py \
      --dns-path ~/ai-project/biology/cancer-ev-data/DNS-Tunnel-Datasets/ \
      --evm-path ~/ai-project/evm_bytecodes/
"""
import argparse, json, os, sys, glob, time
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fq_core import (
    expand_bytes, kgram_counts, debiased_fq, shannon_entropy,
    multiscale_fq, fingerprint, nearest_centroid_classify, ari_score,
    markov_true_h2, markov_true_hq, generate_markov, generate_markov_order_k
)

SEED = 42
Q_LIST = (2, 3, 4)


# === MISSING FUNCTION: Shannon entropy rate via slope ===

def shannon_entropy_rate(data, alpha=16, k_max=None, dim_ratio_limit=0.5):
    """Shannon entropy RATE via multi-scale slope.
    Compute H₁(k) = Shannon entropy of k-gram distribution / k,
    then the slope of H₁_per_symbol(k) gives the entropy rate.
    
    Actually: use log of k-gram count entropy directly.
    For Markov source: H(X_k | X_{k-1}...X_1) converges to h₁.
    We compute H(k-gram)/k for each k, and take the last value as estimate.
    
    Better approach matching our F₂ method:
    Compute Shannon entropy H₁(k) of k-gram distribution at each k.
    For Markov(m): H₁(k) ≈ h₁ * k + C for k > m.
    Slope of H₁(k) vs k = h₁ (Shannon entropy rate).
    """
    if isinstance(data, (bytes, bytearray)):
        seq = expand_bytes(data, alpha)
    else:
        seq = np.asarray(data, dtype=np.uint8)
    n_seq = len(seq)

    if k_max is None:
        k_max = 1
        while True:
            next_k = k_max + 1
            d_next = alpha ** next_k
            n_kgrams_next = n_seq - next_k + 1
            if n_kgrams_next < 10:
                break
            if d_next / n_kgrams_next > dim_ratio_limit:
                break
            k_max = next_k
            if k_max >= 8:
                break

    points = []
    for k in range(1, k_max + 1):
        counts = kgram_counts(data, alpha, k)
        h1_k = shannon_entropy(counts)  # H₁ of k-gram distribution (in bits)
        points.append((k, h1_k))

    if len(points) < 2:
        return None

    ks = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])  # H₁(k) in bits
    # OLS: H₁(k) = slope * k + intercept → slope = h₁ (Shannon rate)
    A = np.vstack([ks, np.ones_like(ks)]).T
    coeff, _, _, _ = np.linalg.lstsq(A, ys, rcond=None)
    slope = coeff[0]
    return float(slope)  # Shannon entropy rate in bits


def shannon_fingerprint(data, alpha=16):
    """3D Shannon fingerprint: H₁ of k-gram for k=1,2,3."""
    if isinstance(data, (bytes, bytearray)):
        seq = expand_bytes(data, alpha)
    else:
        seq = np.asarray(data, dtype=np.uint8)
    result = []
    for k in range(1, 4):
        d = alpha ** k
        n_seq = len(seq)
        if n_seq < k + 1 or d / (n_seq - k + 1) > 0.5:
            result.append(np.nan)
            continue
        counts = kgram_counts(data, alpha, k)
        result.append(shannon_entropy(counts))
    return np.array(result)


def shannon_entropy_mm(counts):
    """Miller-Madow bias-corrected Shannon entropy (bits)."""
    n = int(counts.sum())
    if n == 0:
        return 0.0
    p = counts[counts > 0] / n
    h_plugin = float(-np.sum(p * np.log2(p)))
    S = int((counts > 0).sum())
    mm = (S - 1) / (2 * n * np.log(2))
    return h_plugin + mm


def shannon_rate_debiased(data, alpha=16, dim_ratio_limit=0.5):
    """Shannon entropy RATE with Miller-Madow debiasing at each k."""
    if isinstance(data, (bytes, bytearray)):
        seq = expand_bytes(data, alpha)
    else:
        seq = np.asarray(data, dtype=np.uint8)
    n_seq = len(seq)
    points = []
    for k in range(1, 9):
        d = alpha ** k
        n_k = n_seq - k + 1
        if n_k < 10 or d / n_k > dim_ratio_limit:
            break
        counts = kgram_counts(data, alpha, k)
        h = shannon_entropy_mm(counts)
        points.append((k, h))
    if len(points) < 2:
        return None
    ks = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    A = np.vstack([ks, np.ones_like(ks)]).T
    coeff, _, _, _ = np.linalg.lstsq(A, ys, rcond=None)
    return float(coeff[0])


def shannon_fingerprint_debiased(data, alpha=16):
    """3D Shannon fingerprint with Miller-Madow correction."""
    if isinstance(data, (bytes, bytearray)):
        seq = expand_bytes(data, alpha)
    else:
        seq = np.asarray(data, dtype=np.uint8)
    result = []
    for k in range(1, 4):
        d = alpha ** k
        n_seq = len(seq)
        if n_seq < k + 1 or d / (n_seq - k + 1) > 0.5:
            result.append(np.nan)
            continue
        counts = kgram_counts(data, alpha, k)
        result.append(shannon_entropy_mm(counts))
    return np.array(result)


# === DNS/EVM LOADERS (same as run_experiment.py) ===

def load_dns_pcaps(dns_path):
    try:
        import dpkt
    except ImportError:
        print("WARN: dpkt not installed. Skipping DNS.")
        return []
    results = []
    base = Path(dns_path)
    family_map = {
        'iodine': 'iodine', 'dns2tcp': 'dns2tcp', 'dnscat2': 'dnscat2',
        'dnspot': 'dnspot', 'tuns': 'tuns', 'DNS-shell': 'dns_shell', 'dns-shell': 'dns_shell',
        'cobaltstrike': 'cobaltstrike', 'ozymandns': 'ozymandns',
        'tcp-over-dns': 'tcp_over_dns', 'normal': 'benign', 'wildcard': 'benign',
    }
    for pattern in ['tunnel/**/*.pcap', 'unkownTunnel/**/*.pcap',
                     'normal/**/*.pcap', 'wildcard/**/*.pcap']:
        for pcap_path in sorted(base.glob(pattern)):
            parts = pcap_path.relative_to(base).parts
            family = 'unknown'
            for part in parts:
                for key, val in family_map.items():
                    if key.lower() in part.lower():
                        family = val
                        break
                if family != 'unknown':
                    break
            qbytes = bytearray()
            try:
                with open(pcap_path, 'rb') as f:
                    try:
                        pcap = dpkt.pcap.Reader(f)
                    except:
                        f.seek(0)
                        pcap = dpkt.pcapng.Reader(f)
                    for ts, buf in pcap:
                        try:
                            eth = dpkt.ethernet.Ethernet(buf)
                            if not isinstance(eth.data, dpkt.ip.IP): continue
                            if not isinstance(eth.data.data, dpkt.udp.UDP): continue
                            udp = eth.data.data
                            if udp.dport != 53 and udp.sport != 53: continue
                            dns = dpkt.dns.DNS(udp.data)
                            for q in dns.qd:
                                qbytes.extend(q.name.encode('utf-8', errors='ignore'))
                        except: continue
            except: continue
            if len(qbytes) > 500:
                results.append((bytes(qbytes), family))
    print(f"  DNS: {len(results)} pcaps, families: {sorted(set(r[1] for r in results))}")
    return results


def load_evm_bytecodes(evm_path):
    base = Path(evm_path)
    labels_file = base / 'labels.json'
    if not labels_file.exists():
        return []
    with open(labels_file) as f:
        labels = json.load(f)
    results = []
    for hex_file in sorted((base / 'bytecodes').glob('*.hex')):
        name = hex_file.stem
        if name not in labels: continue
        try:
            hex_str = hex_file.read_text().strip()
            if hex_str.startswith('0x'): hex_str = hex_str[2:]
            data = bytes.fromhex(hex_str)
            if len(data) > 100:
                lbl = 'vuln' if labels[name]['label'] == 1 else 'safe'
                results.append((data, lbl))
        except: continue
    print(f"  EVM: {len(results)} contracts")
    return results


# === LOO CLASSIFICATION HELPER ===

def loo_classify(fps_arr, labels_arr):
    """Leave-one-out nearest-centroid. Returns ARI, accuracy."""
    preds = []
    for i in range(len(fps_arr)):
        mask = np.ones(len(fps_arr), dtype=bool); mask[i] = False
        pred = nearest_centroid_classify(fps_arr[mask], labels_arr[mask], fps_arr[i:i+1])
        preds.append(pred[0])
    preds = np.array(preds)
    acc = float((preds == labels_arr).mean())
    ari = ari_score(labels_arr, preds)
    return ari, acc


# === TESTS ===

def test_1_h2_accuracy(outdir):
    """CLAIM: median error < 0.003 at n=50K for α∈{2,4}."""
    print("\n" + "=" * 60)
    print("[TEST 1] h₂ estimation accuracy")
    print("=" * 60)
    rng = np.random.default_rng(SEED)
    results = []
    for alpha in [2, 4, 8]:
        for trial in range(5):
            P = rng.dirichlet(np.ones(alpha) * (0.5 + trial * 0.5), size=alpha)
            P = P / P.sum(axis=1, keepdims=True)
            true_h2 = markov_true_h2(P)
            for n in [5000, 20000, 50000]:
                data = generate_markov(P, n, seed=SEED + trial * 100 + n)
                seq = np.frombuffer(data, dtype=np.uint8)
                res = multiscale_fq(seq, alpha=alpha, q_list=(2,))
                est = res['h_q'].get(2, np.nan)
                err = abs(est - true_h2) if not np.isnan(est) else np.nan
                results.append({'alpha': alpha, 'n': n, 'true': true_h2, 'est': est, 'err': err})

    # Verify claims
    n50k = [r for r in results if r['n'] == 50000 and not np.isnan(r['err'])]
    a24 = [r for r in n50k if r['alpha'] in [2, 4]]
    a8 = [r for r in n50k if r['alpha'] == 8]

    import statistics
    med_err = statistics.median([r['err'] for r in n50k])
    max_err_a24 = max(r['err'] for r in a24)
    within_005_a8 = sum(1 for r in a8 if r['err'] < 0.05)

    print(f"  Median error at n=50K (all α): {med_err:.4f}")
    print(f"  CLAIM 'median < 0.003': {'✓ PASS' if med_err < 0.003 else '✗ FAIL'}")
    print(f"  Max error α∈{{2,4}}: {max_err_a24:.4f}")
    print(f"  CLAIM '≤ 0.025 for α∈{{2,4}}': {'✓ PASS' if max_err_a24 <= 0.025 else '✗ FAIL (max=' + str(max_err_a24) + ')'}")
    print(f"  α=8 within 0.05: {within_005_a8}/5")

    save_json(results, outdir, 'test1_accuracy.json')
    return med_err < 0.003


def test_2_sqrt_n(outdir):
    """CLAIM: RMSE×√n bounded, near-√n consistent."""
    print("\n" + "=" * 60)
    print("[TEST 2] √n consistency")
    print("=" * 60)
    P = np.array([[0.7, 0.3], [0.4, 0.6]])
    true_h2 = markov_true_h2(P)
    results = []
    for n in [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]:
        errors = []
        for rep in range(30):
            data = generate_markov(P, n, seed=rep * 137)
            seq = np.frombuffer(data, dtype=np.uint8)
            res = multiscale_fq(seq, alpha=2, q_list=(2,))
            errors.append(res['h_q'][2] - true_h2)
        rmse = float(np.sqrt(np.mean(np.array(errors) ** 2)))
        scaled = rmse * np.sqrt(n)
        results.append({'n': n, 'rmse': rmse, 'rmse_sqrt_n': scaled})
        print(f"  n={n:6d}: RMSE={rmse:.5f}, RMSE×√n={scaled:.3f}")

    vals = [r['rmse_sqrt_n'] for r in results]
    print(f"  Range: [{min(vals):.3f}, {max(vals):.3f}]")
    print(f"  Monotonically increasing: {all(vals[i] <= vals[i+1] for i in range(len(vals)-1))}")
    print(f"  CLAIM 'near-√n': {'✓ PASS' if max(vals) < 3.0 else '✗ FAIL'}")

    save_json(results, outdir, 'test2_sqrtn.json')
    return max(vals) < 3.0


def test_3_r2_order(outdir):
    """CLAIM: R² monotonically decreasing order 0→4."""
    print("\n" + "=" * 60)
    print("[TEST 3] R² Markov order diagnostic")
    print("=" * 60)
    results = []
    for order in range(6):
        r2_vals = []
        for rep in range(20):
            data = generate_markov_order_k(4, order, 0.8, 50000, seed=order * 1000 + rep)
            seq = np.frombuffer(data, dtype=np.uint8)
            res = multiscale_fq(seq, alpha=4, q_list=(2,))
            if res['R2'] is not None:
                r2_vals.append(res['R2'])
        mean_r2 = float(np.mean(r2_vals))
        results.append({'order': order, 'R2_mean': mean_r2, 'R2_std': float(np.std(r2_vals))})
        print(f"  Order {order}: R² = {mean_r2:.7f} ± {np.std(r2_vals):.7f}")

    means = [r['R2_mean'] for r in results]
    mono_04 = all(means[i] >= means[i + 1] - 0.0001 for i in range(4))
    mono_05 = all(means[i] >= means[i + 1] - 0.0001 for i in range(5))
    print(f"  Monotonic 0→4: {mono_04}")
    print(f"  Monotonic 0→5: {mono_05}")
    print(f"  Order 0 vs 1 separable: {abs(means[0] - means[1]) > 0.0001}")
    print(f"  CLAIM 'decreasing 0→4': {'✓ PASS' if mono_04 else '✗ FAIL'}")

    save_json(results, outdir, 'test3_r2.json')
    return mono_04


def test_4_critical_ablation(dns_data, outdir):
    """CRITICAL: Is advantage from multi-scale slope or from Rényi specifically?
    
    Compare 7 methods on DNS classification:
    1. Shannon H₁ @ α=256 (byte unigram entropy) — current baseline
    2. Rényi h₂ @ α=256 single-scale (byte unigram collision entropy)
    3. Shannon rate @ α=16 (multi-scale slope of Shannon entropy)
    4. Rényi h₂ @ α=16 slope (our method, single dimension)
    5. (h₂,h₃,h₄) @ α=16 slope (our method, 3D)
    6. Shannon (H₁(k=1), H₁(k=2), H₁(k=3)) @ α=16 (3D Shannon)
    7. (h₂,h₃,h₄) @ α=256 single-scale (3D Rényi, no slope)
    """
    print("\n" + "=" * 60)
    print("[TEST 4] CRITICAL ABLATION — What drives the improvement?")
    print("=" * 60)

    if not dns_data:
        print("  SKIP: no DNS data")
        return False

    methods = {}

    # Method 1: Shannon H₁ @ α=256
    fps, labels = [], []
    for data, fam in dns_data:
        c = kgram_counts(data, 256, 1)
        fps.append([shannon_entropy(c)])
        labels.append(fam)
    fps_arr = np.array(fps); labels_arr = np.array(labels)
    ari, acc = loo_classify(fps_arr, labels_arr)
    methods['Shannon_H1_byte'] = {'ari': ari, 'acc': acc, 'dim': 1, 'alpha': 256, 'type': 'entropy'}
    print(f"  1. Shannon H₁ @α=256 (1D):        ARI={ari:.3f}")

    # Method 2: Rényi h₂ @ α=256 single-scale
    fps, labels = [], []
    for data, fam in dns_data:
        fp = fingerprint(data, alpha=256, q_list=(2,), mode='single')
        if not np.any(np.isnan(fp)):
            fps.append(fp); labels.append(fam)
    fps_arr = np.array(fps); labels_arr = np.array(labels)
    ari, acc = loo_classify(fps_arr, labels_arr)
    methods['Renyi_h2_byte_single'] = {'ari': ari, 'acc': acc, 'dim': 1, 'alpha': 256, 'type': 'entropy'}
    print(f"  2. Rényi h₂ @α=256 single (1D):    ARI={ari:.3f}")

    # Method 3: Shannon rate @ α=16 (slope of H₁(k) vs k) — BIASED
    fps, labels = [], []
    for data, fam in dns_data:
        h1_rate = shannon_entropy_rate(data, alpha=16)
        if h1_rate is not None:
            fps.append([h1_rate]); labels.append(fam)
    fps_arr = np.array(fps); labels_arr = np.array(labels)
    ari, acc = loo_classify(fps_arr, labels_arr)
    methods['Shannon_rate_nib'] = {'ari': ari, 'acc': acc, 'dim': 1, 'alpha': 16, 'type': 'rate', 'debiased': False}
    print(f"  3a. Shannon RATE @α=16 plug-in (1D): ARI={ari:.3f}")

    # Method 3b: Shannon rate @ α=16 WITH Miller-Madow debiasing
    fps, labels = [], []
    for data, fam in dns_data:
        h1_rate = shannon_rate_debiased(data, alpha=16)
        if h1_rate is not None:
            fps.append([h1_rate]); labels.append(fam)
    fps_arr = np.array(fps); labels_arr = np.array(labels)
    ari, acc = loo_classify(fps_arr, labels_arr)
    methods['Shannon_rate_nib_MM'] = {'ari': ari, 'acc': acc, 'dim': 1, 'alpha': 16, 'type': 'rate', 'debiased': True}
    print(f"  3b. Shannon RATE @α=16 debiased (1D):ARI={ari:.3f}  *** CRITICAL: bias confound test ***")

    # Method 4: Rényi h₂ @ α=16 slope (our method, 1D)
    fps, labels = [], []
    for data, fam in dns_data:
        fp = fingerprint(data, alpha=16, q_list=(2,), mode='slope')
        if not np.any(np.isnan(fp)):
            fps.append(fp); labels.append(fam)
    fps_arr = np.array(fps); labels_arr = np.array(labels)
    ari, acc = loo_classify(fps_arr, labels_arr)
    methods['Renyi_h2_nib_slope'] = {'ari': ari, 'acc': acc, 'dim': 1, 'alpha': 16, 'type': 'rate'}
    print(f"  4. Rényi h₂ @α=16 slope (1D):      ARI={ari:.3f}")

    # Method 5: (h₂,h₃,h₄) @ α=16 slope (our full method)
    fps, labels = [], []
    for data, fam in dns_data:
        fp = fingerprint(data, alpha=16, q_list=(2, 3, 4), mode='slope')
        if not np.any(np.isnan(fp)):
            fps.append(fp); labels.append(fam)
    fps_arr = np.array(fps); labels_arr = np.array(labels)
    ari, acc = loo_classify(fps_arr, labels_arr)
    methods['Renyi_h234_nib_slope'] = {'ari': ari, 'acc': acc, 'dim': 3, 'alpha': 16, 'type': 'rate'}
    print(f"  5. Rényi (h₂,h₃,h₄) @α=16 (3D):   ARI={ari:.3f}")

    # Method 6: Shannon 3D @ α=16 — (H₁(k=1), H₁(k=2), H₁(k=3))
    fps, labels = [], []
    for data, fam in dns_data:
        fp = shannon_fingerprint(data, alpha=16)
        if not np.any(np.isnan(fp)):
            fps.append(fp); labels.append(fam)
    fps_arr = np.array(fps); labels_arr = np.array(labels)
    ari, acc = loo_classify(fps_arr, labels_arr)
    methods['Shannon_3D_nib'] = {'ari': ari, 'acc': acc, 'dim': 3, 'alpha': 16, 'type': 'entropy'}
    print(f"  6. Shannon 3D @α=16 (3D):           ARI={ari:.3f}  *** KEY ABLATION ***")

    # Method 7: (h₂,h₃,h₄) @ α=256 single-scale
    fps, labels = [], []
    for data, fam in dns_data:
        fp = fingerprint(data, alpha=256, q_list=(2, 3, 4), mode='single')
        if not np.any(np.isnan(fp)):
            fps.append(fp); labels.append(fam)
    fps_arr = np.array(fps); labels_arr = np.array(labels)
    ari, acc = loo_classify(fps_arr, labels_arr)
    methods['Renyi_h234_byte_single'] = {'ari': ari, 'acc': acc, 'dim': 3, 'alpha': 256, 'type': 'entropy'}
    print(f"  7. Rényi (h₂,h₃,h₄) @α=256 (3D):  ARI={ari:.3f}")

    # === ANALYSIS ===
    print(f"\n  {'='*50}")
    print(f"  ABLATION ANALYSIS")
    print(f"  {'='*50}")

    s_h1 = methods['Shannon_H1_byte']['ari']
    r_h2_s = methods['Renyi_h2_byte_single']['ari']
    s_rate = methods['Shannon_rate_nib']['ari']
    r_rate = methods['Renyi_h2_nib_slope']['ari']
    r_3d = methods['Renyi_h234_nib_slope']['ari']
    s_3d = methods['Shannon_3D_nib']['ari']
    r_3d_s = methods['Renyi_h234_byte_single']['ari']

    print(f"\n  Q1: Does Rényi beat Shannon at SAME α, SAME scale?")
    print(f"      @α=256: Shannon={s_h1:.3f}, Rényi={r_h2_s:.3f} → {'Rényi wins' if r_h2_s > s_h1 else 'Shannon wins'}")
    print(f"      @α=16 slope: Shannon rate={s_rate:.3f}, Rényi rate={r_rate:.3f} → {'Rényi wins' if r_rate > s_rate else 'Shannon wins'}")

    print(f"\n  Q2: Does multi-scale slope beat single-scale?")
    print(f"      Shannon: byte={s_h1:.3f}, nibble slope={s_rate:.3f} → {'Slope wins' if s_rate > s_h1 else 'Single wins'}")
    print(f"      Rényi:   byte={r_h2_s:.3f}, nibble slope={r_rate:.3f} → {'Slope wins' if r_rate > r_h2_s else 'Single wins'}")

    print(f"\n  Q3: Does 3D beat 1D (fair comparison)?")
    print(f"      Rényi @α=16: 1D={r_rate:.3f}, 3D={r_3d:.3f} → gain={r_3d-r_rate:+.3f}")
    print(f"      Shannon @α=16: 1D={s_rate:.3f}, 3D={s_3d:.3f} → gain={s_3d-s_rate:+.3f}")

    print(f"\n  Q4: What is the TRUE source of improvement?")
    improvement_from_slope = s_rate - s_h1
    improvement_from_renyi = r_rate - s_rate
    improvement_from_multiq = r_3d - r_rate
    print(f"      Multi-scale slope contribution: {improvement_from_slope:+.3f} ARI")
    print(f"      Rényi-over-Shannon contribution: {improvement_from_renyi:+.3f} ARI")
    print(f"      Multi-q (3D) contribution: {improvement_from_multiq:+.3f} ARI")

    print(f"\n  VERDICT:")
    if abs(improvement_from_renyi) < 0.05:
        print(f"    → Rényi adds NEGLIGIBLE value over Shannon rate")
        print(f"    → Paper should claim 'multi-scale entropy rate' not 'Rényi beats Shannon'")
    else:
        print(f"    → Rényi adds GENUINE value ({improvement_from_renyi:+.3f}) over Shannon rate")

    save_json(methods, outdir, 'test4_ablation.json')
    return True


def test_5_evm(evm_data, outdir):
    """EVM detection at both α values."""
    print("\n" + "=" * 60)
    print("[TEST 5] EVM bytecode detection")
    print("=" * 60)

    if not evm_data:
        print("  SKIP: no EVM data")
        return False

    from scipy.stats import mannwhitneyu
    results = {}

    for alpha, mode, label in [(256, 'single', 'α=256'), (16, 'slope', 'α=16')]:
        fps, labels = [], []
        for data, lbl in evm_data:
            fp = fingerprint(data, alpha=alpha, q_list=(2,), mode=mode)
            if not np.any(np.isnan(fp)):
                fps.append(fp); labels.append(lbl)
        fps_arr = np.array(fps); labels_arr = np.array(labels)
        classes = sorted(set(labels))

        if len(classes) == 2:
            c0 = fps_arr[labels_arr == classes[0], 0]
            c1 = fps_arr[labels_arr == classes[1], 0]
            pooled = np.sqrt((c0.var() + c1.var()) / 2)
            d = abs(c0.mean() - c1.mean()) / pooled if pooled > 0 else 0
            stat, p = mannwhitneyu(c0, c1, alternative='two-sided')
            auc = max(stat / (len(c0) * len(c1)), 1 - stat / (len(c0) * len(c1)))
            print(f"  {label}: n={len(fps_arr)}, {classes[0]}={c0.mean():.3f}, "
                  f"{classes[1]}={c1.mean():.3f}, d={d:.3f}, AUC={auc:.3f}")
            results[alpha] = {'d': d, 'auc': auc, 'n': len(fps_arr),
                              'c0_mean': float(c0.mean()), 'c1_mean': float(c1.mean())}

    save_json(results, outdir, 'test5_evm.json')
    return True


def test_6_crypto(outdir):
    """Crypto hierarchy — verify ordering and counter_mod256 example."""
    print("\n" + "=" * 60)
    print("[TEST 6] Crypto hierarchy")
    print("=" * 60)
    rng = np.random.default_rng(SEED)
    N = 50000
    streams = {
        'constant': bytes([0] * N),
        'sparse_1pct': bytes(np.where(rng.random(N) < 0.01, 255, 0).astype(np.uint8)),
        'counter_mod256': bytes(np.arange(N, dtype=np.uint8)),
        'ascii_range': bytes(rng.integers(32, 127, N, dtype=np.uint8)),
        'xor_1byte': bytes(rng.integers(0, 256, N, dtype=np.uint8) ^ 0xAB),
        'urandom': bytes(rng.integers(0, 256, N, dtype=np.uint8)),
    }
    results = {}
    for label, data in streams.items():
        res = multiscale_fq(data, alpha=16, q_list=(2,))
        h2 = res['h_q'].get(2, np.nan)
        c = kgram_counts(data, 256, 1)
        h1 = shannon_entropy(c)
        print(f"  {label:20s}: h₂={h2:.4f}, H₁={h1:.4f}, G={h1-h2:.4f}")
        results[label] = {'h2': h2, 'h1': h1, 'gap': h1 - h2}

    # Verify ordering
    h2s = {k: v['h2'] for k, v in results.items()}
    order_correct = (h2s['constant'] < h2s['sparse_1pct'] < h2s['counter_mod256']
                     < h2s['ascii_range'] < h2s['urandom'])
    print(f"\n  Ordering correct: {order_correct}")

    # Counter example
    ctr = results['counter_mod256']
    print(f"  Counter: H₁={ctr['h1']:.1f}, h₂={ctr['h2']:.2f}")
    print(f"  CLAIM 'H₁=8.0 but h₂=2.46': {'✓' if abs(ctr['h1'] - 8.0) < 0.1 and abs(ctr['h2'] - 2.46) < 0.1 else '✗'}")

    save_json(results, outdir, 'test6_crypto.json')
    return order_correct


# === HELPERS ===

def save_json(data, outdir, filename):
    path = os.path.join(outdir, filename)
    def ser(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return str(obj)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=ser)
    print(f"  → {path}")


# === MAIN ===

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dns-path', default=os.path.expanduser(
        '~/ai-project/biology/cancer-ev-data/DNS-Tunnel-Datasets/'))
    parser.add_argument('--evm-path', default=os.path.expanduser('~/ai-project/evm_bytecodes/'))
    parser.add_argument('--outdir', default='mega_results/')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()

    print("=" * 60)
    print("MEGA VALIDATION — Testing every paper claim")
    print("=" * 60)

    print("\n[LOADING]")
    dns_data = load_dns_pcaps(args.dns_path)
    evm_data = load_evm_bytecodes(args.evm_path)

    passed = []
    passed.append(('h₂ accuracy', test_1_h2_accuracy(args.outdir)))
    passed.append(('√n consistency', test_2_sqrt_n(args.outdir)))
    passed.append(('R² order', test_3_r2_order(args.outdir)))
    passed.append(('Critical ablation', test_4_critical_ablation(dns_data, args.outdir)))
    passed.append(('EVM detection', test_5_evm(evm_data, args.outdir)))
    passed.append(('Crypto hierarchy', test_6_crypto(args.outdir)))

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"RESULTS ({elapsed:.0f}s)")
    print(f"{'=' * 60}")
    for name, result in passed:
        print(f"  {'✓' if result else '✗'} {name}")
    print(f"\n  {sum(1 for _, r in passed if r)}/{len(passed)} passed")


if __name__ == '__main__':
    main()

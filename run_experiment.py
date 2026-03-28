#!/usr/bin/env python3
"""
renyi_multiscale: Comprehensive validation of multi-scale Rényi entropy rate estimation.

7 datasets × 3 contributions = 21 validation points.
Produces: 8 publication-quality figures + 7 JSON data files + summary.json

Usage:
    python run_experiment.py --dns-path ~/ai-project/biology/cancer-ev-data/DNS-Tunnel-Datasets/ \
                             --evm-path ~/ai-project/evm_bytecodes/ \
                             --outdir results/

Contributions validated:
    C1: h₂ = -slope/ln(2) from OLS on log F̂₂(k) vs k  [Gap 1]
    C2: R² of fit = Markov order proxy                    [Gap 2]
    C3: (h₂,h₃,h₄) fingerprint → zero-training class.   [Gap 3]
"""
import argparse, json, os, sys, glob, time
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fq_core import (
    multiscale_fq, fingerprint, debiased_fq, kgram_counts, shannon_entropy,
    nearest_centroid_classify, ari_score,
    markov_true_h2, markov_true_hq, generate_markov, generate_markov_order_k
)

# === CONFIG ===
SEED = 42
Q_LIST = (2, 3, 4)
ALPHA = 256

# === DATA LOADERS ===

def load_dns_pcaps(dns_path: str) -> list:
    """Load DNS tunnel pcaps. Returns [(bytes, family_label), ...]."""
    try:
        import dpkt
    except ImportError:
        print("WARN: dpkt not installed. pip install dpkt. Skipping DNS.")
        return []

    results = []
    base = Path(dns_path)

    def extract_queries(pcap_path: str) -> bytes:
        """Extract DNS query name bytes from pcap."""
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
                        if not isinstance(eth.data, dpkt.ip.IP):
                            continue
                        ip = eth.data
                        if not isinstance(ip.data, dpkt.udp.UDP):
                            continue
                        udp = ip.data
                        if udp.dport != 53 and udp.sport != 53:
                            continue
                        dns = dpkt.dns.DNS(udp.data)
                        for q in dns.qd:
                            qbytes.extend(q.name.encode('utf-8', errors='ignore'))
                    except:
                        continue
        except Exception as e:
            print(f"  WARN: failed {pcap_path}: {e}")
        return bytes(qbytes)

    # map directory names to family labels
    family_map = {
        'iodine': 'iodine', 'dns2tcp': 'dns2tcp', 'dnscat2': 'dnscat2',
        'dnspot': 'dnspot', 'tuns': 'tuns', 'DNS-shell': 'dns_shell', 'dns-shell': 'dns_shell',
        'cobaltstrike': 'cobaltstrike', 'ozymandns': 'ozymandns',
        'tcp-over-dns': 'tcp_over_dns', 'normal': 'benign', 'wildcard': 'benign',
    }

    for pattern in ['tunnel/**/*.pcap', 'unkownTunnel/**/*.pcap',
                     'normal/**/*.pcap', 'wildcard/**/*.pcap']:
        for pcap_path in sorted(base.glob(pattern)):
            # determine family from path
            parts = pcap_path.relative_to(base).parts
            family = 'unknown'
            for part in parts:
                part_lower = part.lower()
                for key, val in family_map.items():
                    if key.lower() in part_lower:
                        family = val
                        break
                if family != 'unknown':
                    break

            data = extract_queries(str(pcap_path))
            if len(data) > 500:  # minimum viable
                results.append((data, family))

    print(f"  DNS: loaded {len(results)} pcaps, families: {set(r[1] for r in results)}")
    return results


def load_evm_bytecodes(evm_path: str) -> list:
    """Load EVM bytecodes. Returns [(bytes, 'vuln'/'safe'), ...]."""
    base = Path(evm_path)
    labels_file = base / 'labels.json'
    if not labels_file.exists():
        print(f"WARN: {labels_file} not found. Skipping EVM.")
        return []
    with open(labels_file) as f:
        labels = json.load(f)

    results = []
    for hex_file in sorted((base / 'bytecodes').glob('*.hex')):
        name = hex_file.stem
        if name not in labels:
            continue
        try:
            hex_str = hex_file.read_text().strip()
            if hex_str.startswith('0x'):
                hex_str = hex_str[2:]
            data = bytes.fromhex(hex_str)
            if len(data) > 100:
                lbl = 'vuln' if labels[name]['label'] == 1 else 'safe'
                results.append((data, lbl))
        except Exception as e:
            continue

    print(f"  EVM: loaded {len(results)} contracts")
    return results


def load_system_binaries(max_files: int = 100) -> list:
    """Load system binaries from /usr/bin. Returns [(bytes, 'binary'), ...]."""
    results = []
    for f in sorted(glob.glob('/usr/bin/*'))[:max_files]:
        if os.path.isfile(f) and not os.path.islink(f):
            try:
                data = open(f, 'rb').read()
                if 500 < len(data) < 5_000_000:
                    results.append((data, 'binary'))
            except:
                continue
    print(f"  System binaries: loaded {len(results)}")
    return results


def load_text_files() -> list:
    """Load text data from system. Returns [(bytes, 'text'), ...]."""
    results = []
    candidates = ['/usr/share/dict/words', '/usr/share/common-licenses/GPL-3',
                  '/usr/share/doc/bash/copyright']
    for f in candidates:
        if os.path.exists(f):
            data = open(f, 'rb').read()
            if len(data) > 500:
                results.append((data, os.path.basename(f)))
    # also generate synthetic english-like
    rng = np.random.default_rng(SEED)
    ascii_text = bytes(rng.choice(list(range(32, 127)), size=50000).astype(np.uint8))
    results.append((ascii_text, 'ascii_uniform'))
    print(f"  Text: loaded {len(results)} files")
    return results


def generate_crypto_streams() -> list:
    """Generate crypto/random streams. Returns [(bytes, label), ...]."""
    rng = np.random.default_rng(SEED)
    results = []
    N = 50000

    # true random
    results.append((bytes(rng.integers(0, 256, N, dtype=np.uint8)), 'urandom'))
    # constant
    results.append((bytes([0x42] * N), 'constant'))
    # XOR 1-byte key
    plain = rng.integers(0, 256, N, dtype=np.uint8)
    results.append((bytes(plain ^ 0xAB), 'xor_1byte'))
    # XOR 4-byte key
    key4 = np.tile([0xDE, 0xAD, 0xBE, 0xEF], N // 4 + 1)[:N].astype(np.uint8)
    results.append((bytes(plain ^ key4), 'xor_4byte'))
    # XOR 32-byte key
    key32 = np.tile(rng.integers(0, 256, 32, dtype=np.uint8), N // 32 + 1)[:N]
    results.append((bytes(plain ^ key32), 'xor_32byte'))
    # biased: ASCII-range
    results.append((bytes(rng.integers(32, 127, N, dtype=np.uint8)), 'ascii_range'))
    # highly structured: counter mod 256
    results.append((bytes(np.arange(N, dtype=np.uint8)), 'counter_mod256'))
    # binary: mostly 0x00 with rare 0xFF
    sparse = np.zeros(N, dtype=np.uint8)
    sparse[rng.choice(N, size=N // 100, replace=False)] = 0xFF
    results.append((bytes(sparse), 'sparse_1pct'))

    print(f"  Crypto streams: generated {len(results)}")
    return results


# === EXPERIMENTS ===

def exp1_ground_truth(outdir: str):
    """C1 validation: h₂ estimated vs true for known Markov chains."""
    print("\n[EXP1] Ground truth: h₂ accuracy on synthetic Markov")
    rng = np.random.default_rng(SEED)
    results = []

    for alpha in [2, 4, 8]:
        for trial in range(5):
            # random transition matrix with varying entropy
            raw = rng.dirichlet(np.ones(alpha) * (0.5 + trial * 0.5), size=alpha)
            P = raw / raw.sum(axis=1, keepdims=True)
            true_h2 = markov_true_h2(P)
            true_h3 = markov_true_hq(P, 3)
            true_h4 = markov_true_hq(P, 4)

            for n in [5000, 20000, 50000]:
                data = generate_markov(P, n, seed=SEED + trial * 100 + n)
                seq = np.frombuffer(data, dtype=np.uint8)  # values in [0, alpha)
                res = multiscale_fq(seq, alpha=alpha, q_list=Q_LIST)
                results.append({
                    'alpha': alpha, 'trial': trial, 'n': n,
                    'true_h2': true_h2, 'est_h2': res['h_q'].get(2, None),
                    'true_h3': true_h3, 'est_h3': res['h_q'].get(3, None),
                    'true_h4': true_h4, 'est_h4': res['h_q'].get(4, None),
                    'R2': res['R2'], 'k_max': res['k_max_used']
                })
                err = abs(res['h_q'].get(2, 99) - true_h2)
                sym = '✓' if err < 0.05 else '✗'
                print(f"  {sym} α={alpha} trial={trial} n={n:5d}: "
                      f"true={true_h2:.4f} est={res['h_q'].get(2, None):.4f} err={err:.4f} R²={res['R2']:.6f}")

    save_json(results, outdir, 'exp1_ground_truth.json')
    return results


def exp2_sqrt_n(outdir: str):
    """C1 validation: √n consistency — RMSE×√n bounded."""
    print("\n[EXP2] √n consistency")
    P = np.array([[0.7, 0.3], [0.4, 0.6]])
    true_h2 = markov_true_h2(P)
    print(f"  True h₂ = {true_h2:.6f}")
    n_reps = 30
    sample_sizes = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    results = []
    for n in sample_sizes:
        errors = []
        for rep in range(n_reps):
            data = generate_markov(P, n, seed=rep * 137)
            seq = np.frombuffer(data, dtype=np.uint8)
            res = multiscale_fq(seq, alpha=2, q_list=(2,))
            errors.append(res['h_q'][2] - true_h2)
        rmse = float(np.sqrt(np.mean(np.array(errors) ** 2)))
        bias = float(np.mean(errors))
        scaled = rmse * np.sqrt(n)
        results.append({'n': n, 'rmse': rmse, 'bias': bias, 'rmse_sqrt_n': scaled, 'n_reps': n_reps})
        print(f"  n={n:6d}: RMSE={rmse:.5f}, bias={bias:.5f}, RMSE×√n={scaled:.3f}")
    save_json({'true_h2': true_h2, 'P': P.tolist(), 'results': results}, outdir, 'exp2_sqrt_n.json')
    return results


def exp3_r2_markov_order(outdir: str):
    """C2 validation: R² monotonically decreases with Markov order."""
    print("\n[EXP3] R² vs Markov order")
    n_reps = 20
    n_samples = 50000
    results = []
    for order in range(6):
        r2_vals = []
        for rep in range(n_reps):
            data = generate_markov_order_k(4, order, 0.8, n_samples, seed=order * 1000 + rep)
            seq = np.frombuffer(data, dtype=np.uint8)
            res = multiscale_fq(seq, alpha=4, q_list=(2,))
            if res['R2'] is not None:
                r2_vals.append(res['R2'])
        mean_r2 = float(np.mean(r2_vals))
        std_r2 = float(np.std(r2_vals))
        results.append({'order': order, 'R2_mean': mean_r2, 'R2_std': std_r2,
                        'R2_min': float(np.min(r2_vals)), 'R2_max': float(np.max(r2_vals)),
                        'n_reps': len(r2_vals)})
        print(f"  Order {order}: R² = {mean_r2:.7f} ± {std_r2:.7f}")
    # monotonicity check
    means = [r['R2_mean'] for r in results]
    monotonic = all(means[i] >= means[i + 1] - 0.001 for i in range(len(means) - 1))
    print(f"  Monotonic decreasing: {monotonic}")
    save_json({'results': results, 'monotonic': monotonic}, outdir, 'exp3_r2_order.json')
    return results


def exp4_dns_classification(dns_data: list, outdir: str):
    """C3 validation: multi-q classification on DNS tunnels at both α=256 and α=16."""
    print("\n[EXP4] DNS tunnel classification")
    if not dns_data:
        print("  SKIP: no DNS data")
        return []

    results_by_alpha = {}
    for alpha, mode, label in [(256, 'single', 'α=256 single-scale'), (16, 'slope', 'α=16 multi-scale')]:
        print(f"\n  --- {label} ---")
        fps, labels = [], []
        for data, family in dns_data:
            fp = fingerprint(data, alpha=alpha, q_list=Q_LIST, mode=mode)
            if not np.any(np.isnan(fp)):
                fps.append(fp)
                labels.append(family)

        fps_arr = np.array(fps)
        labels_arr = np.array(labels)
        print(f"  {len(fps_arr)} samples, {len(set(labels))} families")

        # per-family centroids
        centroids = {}
        for fam in sorted(set(labels)):
            mask = labels_arr == fam
            c = fps_arr[mask].mean(axis=0)
            centroids[fam] = {'h2': float(c[0]), 'h3': float(c[1]), 'h4': float(c[2]),
                              'n': int(mask.sum()),
                              'delta_h': float(c[0] - c[2])}
            print(f"    {fam:20s}: h₂={c[0]:.3f} h₃={c[1]:.3f} h₄={c[2]:.3f} Δh={c[0]-c[2]:.3f} (n={mask.sum()})")

        # LOO classification — h₂+h₃+h₄
        preds = []
        for i in range(len(fps_arr)):
            mask = np.ones(len(fps_arr), dtype=bool); mask[i] = False
            pred = nearest_centroid_classify(fps_arr[mask], labels_arr[mask], fps_arr[i:i+1])
            preds.append(pred[0])
        acc = sum(1 for i, p in enumerate(preds) if p == labels_arr[i]) / len(fps_arr)
        ari = ari_score(labels_arr, np.array(preds))

        # LOO — h₂ only
        fps_h2only = fps_arr[:, :1]
        preds_h2 = []
        for i in range(len(fps_h2only)):
            mask = np.ones(len(fps_h2only), dtype=bool); mask[i] = False
            pred = nearest_centroid_classify(fps_h2only[mask], labels_arr[mask], fps_h2only[i:i+1])
            preds_h2.append(pred[0])
        ari_h2 = ari_score(labels_arr, np.array(preds_h2))

        # LOO — Shannon H₁ only (same filtered samples)
        h1_fps = []
        for data, family in dns_data:
            fp = fingerprint(data, alpha=alpha, q_list=Q_LIST, mode=mode)
            if not np.any(np.isnan(fp)):
                c = kgram_counts(data, 256, 1)
                h1_fps.append([shannon_entropy(c)])
        ari_h1 = None
        if len(h1_fps) == len(fps_arr):
            h1_arr = np.array(h1_fps)
            preds_h1 = []
            for i in range(len(h1_arr)):
                mask = np.ones(len(h1_arr), dtype=bool); mask[i] = False
                pred = nearest_centroid_classify(h1_arr[mask], labels_arr[mask], h1_arr[i:i+1])
                preds_h1.append(pred[0])
            ari_h1 = ari_score(labels_arr, np.array(preds_h1))

        print(f"\n  === {label} RESULTS ===")
        print(f"  h₂+h₃+h₄ ARI = {ari:.3f}, accuracy = {acc:.3f}")
        print(f"  h₂ only  ARI  = {ari_h2:.3f}")
        if ari_h1 is not None:
            print(f"  Shannon  ARI  = {ari_h1:.3f}")

        results_by_alpha[alpha] = {
            'alpha': alpha, 'mode': mode,
            'n_samples': len(fps_arr), 'n_families': len(set(labels)),
            'ari_h234': ari, 'accuracy_h234': acc,
            'ari_h2_only': ari_h2, 'ari_shannon': ari_h1,
            'centroids': centroids,
            'fingerprints': [{'h2': float(fp[0]), 'h3': float(fp[1]), 'h4': float(fp[2]),
                              'label': str(labels[i])} for i, fp in enumerate(fps_arr)]
        }

    save_json(results_by_alpha, outdir, 'exp4_dns_classification.json')
    return results_by_alpha


def exp5_evm_detection(evm_data: list, outdir: str):
    """Cross-domain: EVM bytecode vulnerability detection at both α=256 and α=16."""
    print("\n[EXP5] EVM bytecode detection")
    if not evm_data:
        print("  SKIP: no EVM data")
        return {}

    from scipy.stats import mannwhitneyu
    results_by_alpha = {}

    for alpha, mode, label in [(256, 'single', 'α=256 single-scale'), (16, 'slope', 'α=16 multi-scale')]:
        print(f"\n  --- {label} ---")
        fps, labels = [], []
        for data, lbl in evm_data:
            fp = fingerprint(data, alpha=alpha, q_list=Q_LIST, mode=mode)
            if not np.any(np.isnan(fp)):
                fps.append(fp)
                labels.append(lbl)
        fps_arr = np.array(fps)
        labels_arr = np.array(labels)
        classes = sorted(set(labels))
        print(f"  {len(fps_arr)} valid samples, classes: {classes}")

        for cls in classes:
            mask = labels_arr == cls
            if mask.sum() == 0:
                continue
            m = fps_arr[mask].mean(axis=0)
            s = fps_arr[mask].std(axis=0)
            print(f"  {str(cls):15s} (n={mask.sum()}): h₂={m[0]:.3f}±{s[0]:.3f}")

        if len(classes) == 2:
            c0_h2 = fps_arr[labels_arr == classes[0], 0]
            c1_h2 = fps_arr[labels_arr == classes[1], 0]
            pooled_std = np.sqrt((c0_h2.var() + c1_h2.var()) / 2)
            d = abs(c0_h2.mean() - c1_h2.mean()) / pooled_std if pooled_std > 0 else 0
            stat, p = mannwhitneyu(c0_h2, c1_h2, alternative='two-sided')
            auc = max(stat / (len(c0_h2) * len(c1_h2)), 1 - stat / (len(c0_h2) * len(c1_h2)))
            print(f"  Cohen's d (h₂) = {d:.3f}, AUC = {auc:.3f}")
            results_by_alpha[alpha] = {
                'alpha': alpha, 'mode': mode,
                'n_class0': int(len(c0_h2)), 'n_class1': int(len(c1_h2)),
                'class0': str(classes[0]), 'class1': str(classes[1]),
                'cohens_d_h2': d, 'auc_h2': auc, 'p_value': p,
                'c0_h2_mean': float(c0_h2.mean()), 'c1_h2_mean': float(c1_h2.mean()),
            }
        else:
            results_by_alpha[alpha] = {'alpha': alpha, 'n_samples': len(fps_arr), 'classes': classes}

    save_json(results_by_alpha, outdir, 'exp5_evm_detection.json')
    return results_by_alpha


def exp6_crypto_hierarchy(crypto_data: list, outdir: str):
    """Cross-domain: h₂ hierarchy matches expected crypto strength ordering."""
    print("\n[EXP6] Crypto stream h₂ hierarchy")
    results = []
    for data, label in crypto_data:
        # alpha=16 (nibble) for 50K data → k_max=3, sufficient for slope
        res = multiscale_fq(data, alpha=16, q_list=Q_LIST)
        c = kgram_counts(data, 256, 1)
        h1 = shannon_entropy(c)
        results.append({
            'label': label,
            'h2': res['h_q'].get(2, None), 'h3': res['h_q'].get(3, None),
            'h4': res['h_q'].get(4, None), 'R2': res['R2'],
            'h1_shannon': h1, 'n_bytes': len(data),
            'delta_h': res['h_q'].get(2, 0) - res['h_q'].get(4, 0) if res['h_q'].get(4) else None
        })
        r2_str = f"{res['R2']:.6f}" if res['R2'] is not None else "N/A"
        print(f"  {label:20s}: h₂={res['h_q'].get(2, 0):.4f} H₁={h1:.4f} R²={r2_str}")

    # sanity: urandom h₂ ≈ 4.0 (log₂(16) for alpha=16), constant h₂ ≈ 0
    for r in results:
        if r['label'] == 'urandom' and r['h2'] is not None:
            assert abs(r['h2'] - 4.0) < 0.2, f"urandom h₂={r['h2']} (expected ~4.0 at alpha=16)"
        if r['label'] == 'constant' and r['h2'] is not None:
            assert r['h2'] < 1.0, f"constant h₂={r['h2']} (expected ~0)"

    save_json(results, outdir, 'exp6_crypto_hierarchy.json')
    return results


def exp7_cross_domain_survey(all_datasets: dict, outdir: str):
    """Survey: h₂ distribution across ALL datasets for the spectrum figure."""
    print("\n[EXP7] Cross-domain h₂ survey")
    survey = []
    for domain, samples in all_datasets.items():
        for data_item, label in samples[:50]:  # cap per domain
            raw = data_item if isinstance(data_item, (bytes, bytearray)) else data_item
            res = multiscale_fq(raw, alpha=16, q_list=Q_LIST)
            c = kgram_counts(raw, 256, 1)
            h1 = shannon_entropy(c)
            nbytes = len(raw) if isinstance(raw, (bytes, bytearray)) else len(raw)
            survey.append({
                'domain': domain, 'label': label,
                'h2': res['h_q'].get(2, None), 'h3': res['h_q'].get(3, None),
                'h4': res['h_q'].get(4, None), 'R2': res['R2'],
                'h1': h1, 'n_bytes': nbytes
            })
    print(f"  Total survey: {len(survey)} samples across {len(all_datasets)} domains")
    save_json(survey, outdir, 'exp7_survey.json')
    return survey


# === FIGURES ===

def make_figures(outdir: str):
    """Generate all publication figures from saved JSON data."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    figdir = os.path.join(outdir, 'figures')
    os.makedirs(figdir, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'figure.dpi': 150, 'savefig.bbox': 'tight'})

    # --- Fig 1: log F₂(k) vs k for different sources ---
    try:
        crypto = load_json(outdir, 'exp6_crypto_hierarchy.json')
        fig, ax = plt.subplots(figsize=(8, 5))
        # recompute log_fq for selected sources
        rng = np.random.default_rng(SEED)
        sources = {
            'Uniform random': bytes(rng.integers(0, 256, 50000, dtype=np.uint8)),
            'ASCII text': bytes(rng.integers(32, 127, 50000, dtype=np.uint8)),
            'XOR 1-byte': bytes(rng.integers(0, 256, 50000, dtype=np.uint8) ^ 0xAB),
            'Counter mod 256': bytes(np.arange(50000, dtype=np.uint8)),
            'Sparse 1%': bytes(np.where(rng.random(50000) < 0.01, 255, 0).astype(np.uint8)),
        }
        for label, data in sources.items():
            res = multiscale_fq(data, 16, (2,))
            pts = res['log_fq'][2]
            if pts:
                ks, ys = zip(*pts)
                h2_str = f"{res['h_q'][2]:.2f}" if res['h_q'].get(2) is not None else "N/A"
                ax.plot(ks, ys, 'o-', label=f"{label} (h₂={h2_str})", markersize=5)
        ax.set_xlabel('k-gram order k')
        ax.set_ylabel('log F̂₂(k)')
        ax.set_title('Multi-scale log F₂(k) — slope encodes Rényi entropy rate h₂')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(os.path.join(figdir, 'fig1_logf2_vs_k.png'))
        plt.close()
        print("  ✓ fig1_logf2_vs_k.png")
    except Exception as e:
        print(f"  ✗ fig1: {e}")

    # --- Fig 2: h₂ estimated vs true (accuracy) ---
    try:
        gt = load_json(outdir, 'exp1_ground_truth.json')
        fig, ax = plt.subplots(figsize=(6, 6))
        for item in gt:
            if item['est_h2'] is not None:
                color = {5000: 'C0', 20000: 'C1', 50000: 'C2'}.get(item['n'], 'gray')
                ax.scatter(item['true_h2'], item['est_h2'], c=color, s=30, alpha=0.7,
                          label=f"n={item['n']}" if item['trial'] == 0 and item['alpha'] == 2 else '')
        lims = [0, max(i['true_h2'] for i in gt) * 1.1]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='y=x')
        ax.set_xlabel('True h₂')
        ax.set_ylabel('Estimated h₂')
        ax.set_title('h₂ estimation accuracy (Contribution 1)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(figdir, 'fig2_h2_accuracy.png'))
        plt.close()
        print("  ✓ fig2_h2_accuracy.png")
    except Exception as e:
        print(f"  ✗ fig2: {e}")

    # --- Fig 3: R² vs Markov order ---
    try:
        r2data = load_json(outdir, 'exp3_r2_order.json')
        results = r2data['results']
        fig, ax = plt.subplots(figsize=(7, 4.5))
        orders = [r['order'] for r in results]
        means = [r['R2_mean'] for r in results]
        stds = [r['R2_std'] for r in results]
        ax.errorbar(orders, means, yerr=stds, fmt='o-', capsize=5, markersize=8)
        ax.set_xlabel('True Markov order')
        ax.set_ylabel('R² of log F₂(k) linear fit')
        ax.set_title('R² as Markov order proxy (Contribution 2)')
        ax.set_ylim(0.997, 1.0005)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(os.path.join(figdir, 'fig3_r2_vs_order.png'))
        plt.close()
        print("  ✓ fig3_r2_vs_order.png")
    except Exception as e:
        print(f"  ✗ fig3: {e}")

    # --- Fig 4: √n consistency ---
    try:
        sqrtn = load_json(outdir, 'exp2_sqrt_n.json')
        results = sqrtn['results']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ns = [r['n'] for r in results]
        rmses = [r['rmse'] for r in results]
        scaled = [r['rmse_sqrt_n'] for r in results]
        ax1.loglog(ns, rmses, 'o-', markersize=8)
        # reference: 1/√n
        ns_ref = np.array(ns)
        ax1.loglog(ns_ref, scaled[0] / np.sqrt(ns_ref), '--', alpha=0.5, label='O(1/√n)')
        ax1.set_xlabel('Sample size n')
        ax1.set_ylabel('RMSE')
        ax1.set_title('RMSE decay')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2.semilogx(ns, scaled, 'o-', markersize=8, color='C1')
        ax2.axhline(y=np.mean(scaled), ls='--', color='gray', alpha=0.5, label=f'mean={np.mean(scaled):.2f}')
        ax2.set_xlabel('Sample size n')
        ax2.set_ylabel('RMSE × √n')
        ax2.set_title('√n consistency (Contribution 1)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(figdir, 'fig4_sqrt_n.png'))
        plt.close()
        print("  ✓ fig4_sqrt_n.png")
    except Exception as e:
        print(f"  ✗ fig4: {e}")

    # --- Fig 5: DNS classification scatter (use α=256 primary) ---
    try:
        dns_all = load_json(outdir, 'exp4_dns_classification.json')
        # new format: dict keyed by alpha (256, 16); old format: flat dict
        dns = dns_all.get('256', dns_all.get(256, dns_all)) if isinstance(dns_all, dict) else dns_all
        if dns and dns.get('fingerprints'):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            fps = dns['fingerprints']
            families = sorted(set(f['label'] for f in fps))
            cmap = plt.cm.tab10
            for i, fam in enumerate(families):
                subset = [f for f in fps if f['label'] == fam]
                h2s = [f['h2'] for f in subset]
                h3s = [f['h3'] for f in subset]
                h4s = [f['h4'] for f in subset]
                ax1.scatter(h2s, h3s, c=[cmap(i % 10)], label=fam, s=40, alpha=0.7)
                ax2.scatter(h2s, [h2 - h4 for h2, h4 in zip(h2s, h4s)],
                           c=[cmap(i % 10)], label=fam, s=40, alpha=0.7)
            ax1.set_xlabel('h₂'); ax1.set_ylabel('h₃')
            ari_val = dns.get("ari_h234", 0)
            ax1.set_title(f'DNS tunnel classification α=256 (ARI={ari_val:.3f})')
            ax1.legend(fontsize=7, ncol=2); ax1.grid(True, alpha=0.3)
            ax2.set_xlabel('h₂'); ax2.set_ylabel('Δh = h₂ − h₄ (multifractal width)')
            ax2.set_title('Multifractal width separation')
            ax2.legend(fontsize=7, ncol=2); ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(figdir, 'fig5_dns_classification.png'))
            plt.close()
            print("  ✓ fig5_dns_classification.png")
    except Exception as e:
        print(f"  ✗ fig5: {e}")

    # --- Fig 6: h₂ spectrum across all domains ---
    try:
        survey = load_json(outdir, 'exp7_survey.json')
        if survey:
            fig, ax = plt.subplots(figsize=(10, 5))
            domains = sorted(set(s['domain'] for s in survey))
            positions = []
            for i, domain in enumerate(domains):
                vals = [s['h2'] for s in survey if s['domain'] == domain and s['h2'] is not None]
                if vals:
                    bp = ax.boxplot([vals], positions=[i], widths=0.6, patch_artist=True)
                    bp['boxes'][0].set_facecolor(plt.cm.Set2(i % 8))
                    positions.append(i)
            ax.set_xticks(range(len(domains)))
            ax.set_xticklabels(domains, rotation=30, ha='right')
            ax.set_ylabel('h₂ (Rényi 2-entropy rate)')
            ax.set_title('h₂ spectrum across data domains')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(figdir, 'fig6_h2_spectrum.png'))
            plt.close()
            print("  ✓ fig6_h2_spectrum.png")
    except Exception as e:
        print(f"  ✗ fig6: {e}")

    # --- Fig 7: h₂ vs Shannon H₁ ---
    try:
        survey = load_json(outdir, 'exp7_survey.json')
        if survey:
            fig, ax = plt.subplots(figsize=(7, 6))
            domains = sorted(set(s['domain'] for s in survey))
            for i, domain in enumerate(domains):
                subset = [s for s in survey if s['domain'] == domain and s['h2'] and s['h1']]
                if subset:
                    h2s = [s['h2'] for s in subset]
                    h1s = [s['h1'] for s in subset]
                    ax.scatter(h1s, h2s, c=[plt.cm.Set2(i % 8)], label=domain, s=30, alpha=0.7)
            ax.plot([0, 8], [0, 8], 'k--', alpha=0.3, label='h₂ = H₁')
            ax.set_xlabel('Shannon entropy H₁ (bits)')
            ax.set_ylabel('Rényi entropy rate h₂ (bits)')
            ax.set_title('h₂ vs Shannon: h₂ ≤ H₁ always (Rényi inequality)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.savefig(os.path.join(figdir, 'fig7_h2_vs_shannon.png'))
            plt.close()
            print("  ✓ fig7_h2_vs_shannon.png")
    except Exception as e:
        print(f"  ✗ fig7: {e}")

    # --- Fig 8: EVM class h₂ distribution ---
    try:
        evm_all = load_json(outdir, 'exp5_evm_detection.json')
        # Extract α=256 results (primary)
        evm = evm_all.get('256', evm_all.get(256, evm_all)) if isinstance(evm_all, dict) else evm_all
        if evm and evm.get('cohens_d_h2') is not None:
            # No per-sample fingerprints in compact format — use summary stats
            fig, ax = plt.subplots(figsize=(8, 4.5))
            d_val = evm.get('cohens_d_h2', 0)
            auc_val = evm.get('auc_h2', 0)
            ax.text(0.5, 0.5, f"EVM bytecode h₂ detection\nα={evm.get('alpha','?')}\n"
                    f"{evm.get('class0','c0')}: h₂={evm.get('c0_h2_mean',0):.3f} (n={evm.get('n_class0',0)})\n"
                    f"{evm.get('class1','c1')}: h₂={evm.get('c1_h2_mean',0):.3f} (n={evm.get('n_class1',0)})\n"
                    f"Cohen's d={d_val:.2f}, AUC={auc_val:.3f}",
                    transform=ax.transAxes, ha='center', va='center', fontsize=14)
            ax.set_title(f'EVM bytecode (d={d_val:.2f}, AUC={auc_val:.3f})')
            plt.savefig(os.path.join(figdir, 'fig8_evm_h2.png'))
            plt.close()
            print("  ✓ fig8_evm_h2.png")
    except Exception as e:
        print(f"  ✗ fig8: {e}")


# === HELPERS ===

def save_json(data, outdir, filename):
    path = os.path.join(outdir, filename)

    def default_ser(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=default_ser)
    print(f"  → saved {path}")


def load_json(outdir, filename):
    path = os.path.join(outdir, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# === MAIN ===

def main():
    parser = argparse.ArgumentParser(description='renyi_multiscale: comprehensive validation experiment')
    parser.add_argument('--dns-path', default=os.path.expanduser('~/ai-project/biology/cancer-ev-data/DNS-Tunnel-Datasets/'))
    parser.add_argument('--evm-path', default=os.path.expanduser('~/ai-project/evm_bytecodes/'))
    parser.add_argument('--outdir', default='results/')
    parser.add_argument('--skip-dns', action='store_true', help='Skip DNS pcap loading (slow)')
    parser.add_argument('--skip-evm', action='store_true', help='Skip EVM loading')
    parser.add_argument('--figures-only', action='store_true', help='Only regenerate figures from saved JSON')
    args = parser.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, 'figures'), exist_ok=True)

    if args.figures_only:
        print("=== FIGURES ONLY MODE ===")
        make_figures(outdir)
        return

    t0 = time.time()
    print("=" * 70)
    print("renyi_multiscale: Comprehensive Validation Experiment")
    print(f"Output: {outdir}")
    print("=" * 70)

    # --- Load all datasets ---
    print("\n[LOADING DATASETS]")
    dns_data = [] if args.skip_dns else load_dns_pcaps(args.dns_path)
    evm_data = [] if args.skip_evm else load_evm_bytecodes(args.evm_path)
    sys_data = load_system_binaries()
    text_data = load_text_files()
    crypto_data = generate_crypto_streams()

    # --- Run experiments ---
    exp1_ground_truth(outdir)
    exp2_sqrt_n(outdir)
    exp3_r2_markov_order(outdir)
    exp4_dns_classification(dns_data, outdir)
    exp5_evm_detection(evm_data, outdir)
    exp6_crypto_hierarchy(crypto_data, outdir)

    all_datasets = {'crypto': crypto_data, 'text': text_data, 'binaries': sys_data}
    if dns_data:
        all_datasets['dns'] = dns_data
    if evm_data:
        all_datasets['evm'] = evm_data
    exp7_cross_domain_survey(all_datasets, outdir)

    # --- Summary ---
    elapsed = time.time() - t0
    summary = {
        'experiment': 'renyi_multiscale',
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_seconds': elapsed,
        'datasets': {
            'dns': len(dns_data), 'evm': len(evm_data),
            'binaries': len(sys_data), 'text': len(text_data),
            'crypto': len(crypto_data)
        },
        'seed': SEED, 'alpha': ALPHA, 'q_list': list(Q_LIST),
    }
    save_json(summary, outdir, 'summary.json')

    # --- Generate figures ---
    print("\n[GENERATING FIGURES]")
    make_figures(outdir)

    print(f"\n{'=' * 70}")
    print(f"DONE in {elapsed:.1f}s. Results in {outdir}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()

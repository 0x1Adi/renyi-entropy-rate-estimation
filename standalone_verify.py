#!/usr/bin/env python3
"""
STANDALONE VERIFIER — Zero imports from our code.
Pure numpy. Every line readable. No helper functions.
Computes ARI for all methods on DNS data from scratch.

Run: python standalone_verify.py --dns-path ~/ai-project/biology/cancer-ev-data/DNS-Tunnel-Datasets/
"""
import numpy as np, argparse, os, sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dns-path', required=True)
    args = parser.parse_args()

    # ========================================
    # STEP 1: Load DNS pcaps (dpkt)
    # ========================================
    import dpkt
    family_map = {
        'iodine': 'iodine', 'dns2tcp': 'dns2tcp', 'dnscat2': 'dnscat2',
        'dnspot': 'dnspot', 'tuns': 'tuns', 'dns-shell': 'dns_shell',
        'DNS-shell': 'dns_shell', 'cobaltstrike': 'cobaltstrike',
        'ozymandns': 'ozymandns', 'tcp-over-dns': 'tcp_over_dns',
        'normal': 'benign', 'wildcard': 'benign',
    }
    samples = []  # list of (raw_bytes, family_label)
    base = Path(args.dns_path)
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
                        except:
                            continue
            except:
                continue
            if len(qbytes) > 500:
                samples.append((bytes(qbytes), family))

    print(f"Loaded {len(samples)} pcaps")
    families = sorted(set(s[1] for s in samples))
    print(f"Families ({len(families)}): {families}")

    # ========================================
    # STEP 2: Core functions — ALL INLINE, no helpers
    # ========================================

    def to_nibbles(data):
        """Byte → two nibbles. 0xAB → [0xA, 0xB]."""
        raw = np.frombuffer(data, dtype=np.uint8)
        out = np.empty(2 * len(raw), dtype=np.uint8)
        out[0::2] = raw >> 4
        out[1::2] = raw & 0x0F
        return out

    def count_kgrams(seq, alpha, k):
        """Count k-grams. Returns array of length alpha^k."""
        n = len(seq)
        d = alpha ** k
        idx = np.zeros(n - k + 1, dtype=np.int64)
        for j in range(k):
            idx += seq[j:n - k + 1 + j].astype(np.int64) * (alpha ** (k - 1 - j))
        counts = np.zeros(d, dtype=np.int64)
        np.add.at(counts, idx, 1)
        return counts

    def F2_debiased(counts):
        """Debiased F₂ = (Σ n_i(n_i-1)) / (n(n-1)). Unbiased for multinomial."""
        n = int(counts.sum())
        if n < 2: return np.nan
        num = float(np.sum(counts.astype(np.float64) * (counts - 1)))
        den = float(n) * (n - 1)
        return num / den

    def F3_debiased(counts):
        """Debiased F₃ = Σ n_i(n_i-1)(n_i-2) / n(n-1)(n-2)."""
        n = int(counts.sum())
        if n < 3: return np.nan
        c = counts.astype(np.float64)
        num = float(np.sum(c * np.maximum(counts - 1, 0) * np.maximum(counts - 2, 0)))
        den = float(n) * (n - 1) * (n - 2)
        return num / den

    def F4_debiased(counts):
        """Debiased F₄ = Σ n_i^{(4)} / n^{(4)}."""
        n = int(counts.sum())
        if n < 4: return np.nan
        c = counts.astype(np.float64)
        num = float(np.sum(c * np.maximum(counts-1,0) * np.maximum(counts-2,0) * np.maximum(counts-3,0)))
        den = float(n) * (n-1) * (n-2) * (n-3)
        return num / den

    def shannon_H1(counts):
        """-Σ p log₂ p. Plug-in, biased."""
        n = counts.sum()
        if n == 0: return 0.0
        p = counts[counts > 0] / n
        return float(-np.sum(p * np.log2(p)))

    def shannon_H1_mm(counts):
        """Miller-Madow bias-corrected Shannon entropy."""
        n = int(counts.sum())
        if n == 0: return 0.0
        p = counts[counts > 0] / n
        h = float(-np.sum(p * np.log2(p)))
        S = int((counts > 0).sum())
        return h + (S - 1) / (2 * n * np.log(2))

    def ari(true, pred):
        """Adjusted Rand Index. Matches sklearn exactly."""
        n = len(true)
        ct_map = {}
        for t, p in zip(true, pred):
            ct_map[(t, p)] = ct_map.get((t, p), 0) + 1
        a_map, b_map = {}, {}
        for (t, p), v in ct_map.items():
            a_map[t] = a_map.get(t, 0) + v
            b_map[p] = b_map.get(p, 0) + v
        c2 = lambda x: x * (x - 1) // 2
        sum_ct = sum(c2(v) for v in ct_map.values())
        sum_a = sum(c2(v) for v in a_map.values())
        sum_b = sum(c2(v) for v in b_map.values())
        cn = c2(n)
        if cn == 0: return 0.0
        exp = sum_a * sum_b / cn
        mx = (sum_a + sum_b) / 2
        den = mx - exp
        if abs(den) < 1e-12: return 1.0 if abs(sum_ct - exp) < 1e-12 else 0.0
        return (sum_ct - exp) / den

    def loo_nc(fps, labels):
        """Leave-one-out nearest-centroid classification → ARI."""
        classes = sorted(set(labels))
        preds = []
        for i in range(len(fps)):
            # compute centroids excluding i
            centroids = []
            for c in classes:
                members = [fps[j] for j in range(len(fps)) if j != i and labels[j] == c]
                if members:
                    centroids.append((c, np.mean(members, axis=0)))
            # nearest centroid
            best_c, best_d = classes[0], np.inf
            for c, cent in centroids:
                d = np.linalg.norm(fps[i] - cent)
                if d < best_d:
                    best_c, best_d = c, d
            preds.append(best_c)
        return ari(labels, preds)

    # ========================================
    # STEP 3: Compute ALL methods
    # ========================================
    print("\n" + "=" * 70)
    print("STANDALONE VERIFICATION — 9 methods, all from scratch")
    print("=" * 70)

    # Precompute nibble sequences
    nib_seqs = {}
    for i, (data, fam) in enumerate(samples):
        nib_seqs[i] = to_nibbles(data)

    # ---------- Method 1: Shannon H₁ @ byte (α=256, k=1) ----------
    fps = [[shannon_H1(count_kgrams(np.frombuffer(d, np.uint8), 256, 1))] for d, _ in samples]
    labels = [s[1] for s in samples]
    a = loo_nc(np.array(fps), labels)
    print(f"  1. Shannon H₁ @byte          (1D): ARI = {a:.3f}")

    # ---------- Method 2: Rényi H₂ @ byte (α=256, k=1, single-scale) ----------
    fps = []
    for d, _ in samples:
        c = count_kgrams(np.frombuffer(d, np.uint8), 256, 1)
        f2 = F2_debiased(c)
        fps.append([float(-np.log2(f2))] if f2 > 0 else [np.nan])
    valid = [(fp, l) for fp, l in zip(fps, labels) if not np.isnan(fp[0])]
    a = loo_nc(np.array([v[0] for v in valid]), [v[1] for v in valid])
    print(f"  2. Rényi H₂ @byte single     (1D): ARI = {a:.3f}")

    # ---------- Helper: compute slope from k=1..K ----------
    def slope_from_points(ks, ys):
        if len(ks) < 2: return None
        ks, ys = np.array(ks, dtype=float), np.array(ys, dtype=float)
        A = np.vstack([ks, np.ones_like(ks)]).T
        coeff, _, _, _ = np.linalg.lstsq(A, ys, rcond=None)
        return coeff[0]

    # ---------- Method 3a: Shannon rate @α=16 plug-in ----------
    fps, lbls = [], []
    for i, (data, fam) in enumerate(samples):
        seq = nib_seqs[i]
        pts = []
        for k in range(1, 4):
            d = 16 ** k
            nk = len(seq) - k + 1
            if d / nk > 0.5: break
            c = count_kgrams(seq, 16, k)
            pts.append((k, shannon_H1(c)))
        s = slope_from_points([p[0] for p in pts], [p[1] for p in pts])
        if s is not None:
            fps.append([s]); lbls.append(fam)
    a = loo_nc(np.array(fps), lbls)
    print(f"  3a. Shannon rate @nib plugin  (1D): ARI = {a:.3f}")

    # ---------- Method 3b: Shannon rate @α=16 Miller-Madow debiased ----------
    fps, lbls = [], []
    for i, (data, fam) in enumerate(samples):
        seq = nib_seqs[i]
        pts = []
        for k in range(1, 4):
            d = 16 ** k
            nk = len(seq) - k + 1
            if d / nk > 0.5: break
            c = count_kgrams(seq, 16, k)
            pts.append((k, shannon_H1_mm(c)))
        s = slope_from_points([p[0] for p in pts], [p[1] for p in pts])
        if s is not None:
            fps.append([s]); lbls.append(fam)
    a = loo_nc(np.array(fps), lbls)
    print(f"  3b. Shannon rate @nib MM      (1D): ARI = {a:.3f}  *** BIAS CONFOUND ***")

    # ---------- Method 4: Rényi h₂ @α=16 slope ----------
    fps, lbls = [], []
    for i, (data, fam) in enumerate(samples):
        seq = nib_seqs[i]
        pts = []
        for k in range(1, 4):
            d = 16 ** k
            nk = len(seq) - k + 1
            if d / nk > 0.5: break
            c = count_kgrams(seq, 16, k)
            f2 = F2_debiased(c)
            if f2 > 0:
                pts.append((k, float(np.log(f2))))
        s = slope_from_points([p[0] for p in pts], [p[1] for p in pts])
        if s is not None:
            h2 = float(-s / np.log(2))  # h₂ = -slope / ln(2)
            fps.append([h2]); lbls.append(fam)
    a = loo_nc(np.array(fps), lbls)
    print(f"  4. Rényi h₂ @nib slope        (1D): ARI = {a:.3f}")

    # ---------- Method 5: Rényi (h₂,h₃,h₄) @α=16 slope ----------
    fps, lbls = [], []
    for i, (data, fam) in enumerate(samples):
        seq = nib_seqs[i]
        hqs = []
        for q, Fq_fn in [(2, F2_debiased), (3, F3_debiased), (4, F4_debiased)]:
            pts = []
            for k in range(1, 4):
                d = 16 ** k
                nk = len(seq) - k + 1
                if d / nk > 0.5: break
                c = count_kgrams(seq, 16, k)
                fq = Fq_fn(c)
                if fq is not None and fq > 0:
                    pts.append((k, float(np.log(fq))))
            s = slope_from_points([p[0] for p in pts], [p[1] for p in pts])
            if s is not None:
                hqs.append(float(-s / ((q - 1) * np.log(2))))
            else:
                hqs.append(np.nan)
        if not any(np.isnan(h) for h in hqs):
            fps.append(hqs); lbls.append(fam)
    a = loo_nc(np.array(fps), lbls)
    print(f"  5. Rényi (h₂,h₃,h₄) @nib     (3D): ARI = {a:.3f}")

    # ---------- Method 6a: Shannon 3D @α=16 plug-in ----------
    fps, lbls = [], []
    for i, (data, fam) in enumerate(samples):
        seq = nib_seqs[i]
        hvals = []
        for k in range(1, 4):
            d = 16 ** k
            nk = len(seq) - k + 1
            if d / nk > 0.5:
                hvals.append(np.nan)
                continue
            c = count_kgrams(seq, 16, k)
            hvals.append(shannon_H1(c))
        if not any(np.isnan(h) for h in hvals):
            fps.append(hvals); lbls.append(fam)
    a = loo_nc(np.array(fps), lbls)
    print(f"  6a. Shannon 3D @nib plugin    (3D): ARI = {a:.3f}")

    # ---------- Method 6b: Shannon 3D @α=16 MM debiased ----------
    fps, lbls = [], []
    for i, (data, fam) in enumerate(samples):
        seq = nib_seqs[i]
        hvals = []
        for k in range(1, 4):
            d = 16 ** k
            nk = len(seq) - k + 1
            if d / nk > 0.5:
                hvals.append(np.nan)
                continue
            c = count_kgrams(seq, 16, k)
            hvals.append(shannon_H1_mm(c))
        if not any(np.isnan(h) for h in hvals):
            fps.append(hvals); lbls.append(fam)
    a = loo_nc(np.array(fps), lbls)
    print(f"  6b. Shannon 3D @nib MM        (3D): ARI = {a:.3f}  *** BIAS CONFOUND ***")

    # ---------- Method 7: Rényi (h₂,h₃,h₄) @byte single-scale ----------
    fps, lbls = [], []
    for i, (data, fam) in enumerate(samples):
        raw = np.frombuffer(data, np.uint8)
        c = count_kgrams(raw, 256, 1)
        f2 = F2_debiased(c)
        f3 = F3_debiased(c)
        f4 = F4_debiased(c)
        if f2 and f3 and f4 and f2 > 0 and f3 > 0 and f4 > 0:
            h2 = float(-np.log2(f2))
            h3 = float(np.log2(f3) / (1 - 3))
            h4 = float(np.log2(f4) / (1 - 4))
            fps.append([h2, h3, h4]); lbls.append(fam)
    a = loo_nc(np.array(fps), lbls)
    print(f"  7. Rényi (h₂,h₃,h₄) @byte    (3D): ARI = {a:.3f}")

    # ========================================
    # STEP 4: Cross-check ARI with sklearn
    # ========================================
    print("\n" + "=" * 70)
    print("SKLEARN CROSS-CHECK")
    print("=" * 70)
    try:
        from sklearn.metrics import adjusted_rand_score
        # Recompute method 1 and check
        fps = [shannon_H1(count_kgrams(np.frombuffer(d, np.uint8), 256, 1)) for d, _ in samples]
        labels = [s[1] for s in samples]
        # Simple nearest-centroid LOO
        preds = []
        for i in range(len(fps)):
            classes = sorted(set(labels))
            centroids = {}
            for c in classes:
                vals = [fps[j] for j in range(len(fps)) if j != i and labels[j] == c]
                if vals:
                    centroids[c] = np.mean(vals)
            best_c = min(centroids.keys(), key=lambda c: abs(fps[i] - centroids[c]))
            preds.append(best_c)
        our_ari = ari(labels, preds)
        sk_ari = adjusted_rand_score(labels, preds)
        print(f"  Method 1 ARI: ours={our_ari:.6f}, sklearn={sk_ari:.6f}, match={abs(our_ari-sk_ari)<1e-10}")
    except ImportError:
        print("  sklearn not available — install to cross-check: pip install scikit-learn")

    print("\nDONE. If numbers match mega_validation.py → code is consistent.")
    print("If they differ → one of the implementations has a bug.")


if __name__ == '__main__':
    main()

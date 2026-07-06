"""CQT-chromagram + DTW alignment of Korean originals to Mandarin 填詞 covers.

Feature space: HPSS-harmonic Constant-Q chromagram (timbre/vocal-invariant).
Alignment: subsequence DTW (open begin+end) — matches the shorter clip as a
contiguous segment of the longer one, so MV intros/outros are skipped cleanly.
"""
import os, csv, warnings
import numpy as np
import librosa

warnings.filterwarnings("ignore")  # quiet librosa m4a/audioread deprecation noise

BASE = os.environ.get("KPOP_DATA", "data")   # dataset root (set KPOP_DATA to your copy)
FEATDIR = os.path.join(BASE, "features")
ALIGNDIR = os.path.join(BASE, "alignments")
SR, HOP, BPO = 22050, 1024, 36          # 22.05 kHz, ~46 ms/frame, 36 bins/octave CQT
FPS = SR / HOP


def load_pairs(base=BASE):
    """Return matched (idx, name, cover_path, original_path) from manifest.csv."""
    pairs = []
    with open(os.path.join(base, "manifest.csv")) as f:
        for r in csv.DictReader(f):
            cov, orig = r["primary_cover"], r["original_file"]
            if orig and orig != "NOT DOWNLOADED" and cov not in ("(none in covers/)", ""):
                pairs.append((int(r["idx"]), r["original"],
                              os.path.join(base, "covers", cov),
                              os.path.join(base, "originals", orig)))
    return pairs


def compute_chroma(path):
    """Load audio -> mono 22.05k -> HPSS harmonic -> CQT chroma (12 x T), L2-normalized."""
    y, sr = librosa.load(path, sr=SR, mono=True)
    y_h, _ = librosa.effects.hpss(y)
    C = librosa.feature.chroma_cqt(y=y_h, sr=sr, hop_length=HOP, bins_per_octave=BPO)
    C = librosa.util.normalize(C + 1e-6, axis=0)   # +eps: silent frames -> uniform, avoids NaN cosine
    return C.astype(np.float32)


def chroma_cached(idx, side, path, featdir=FEATDIR):
    """Chroma for one file, cached at features/{idx:03d}_{side}.npy."""
    os.makedirs(featdir, exist_ok=True)
    fp = os.path.join(featdir, f"{idx:03d}_{side}.npy")
    if os.path.exists(fp):
        return np.load(fp)
    C = compute_chroma(path)
    np.save(fp, C)
    return C


def frames_to_sec(f):
    return np.asarray(f) * HOP / SR


def align(cover_chroma, orig_chroma, transpose=True):
    """Subsequence DTW (open begin+end). Returns dict with warping path + cost.

    Query = shorter sequence (fully matched); reference = longer (partially matched).
    If transpose=True, first corrects a global key difference (some covers are sung
    in a different key, e.g. "降调版") by rolling the cover chroma to the semitone
    offset that best matches the original (optimal transposition index).
    """
    from dtw import dtw
    def _san(C):  # defend against zero/NaN frames (cosine distance is undefined for zero vectors)
        C = np.nan_to_num(np.asarray(C, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        return librosa.util.normalize(C + 1e-6, axis=0)
    Cc, Co = _san(cover_chroma), _san(orig_chroma)
    semis = 0
    if transpose:
        hc, ho = Cc.mean(axis=1), Co.mean(axis=1)   # global chroma profiles
        semis = int(np.argmax([float(np.dot(np.roll(hc, s), ho)) for s in range(12)]))
        if semis:
            Cc = np.roll(Cc, semis, axis=0)
    cover_is_query = Cc.shape[1] <= Co.shape[1]
    Q, R = (Cc, Co) if cover_is_query else (Co, Cc)
    al = dtw(Q.T, R.T, dist_method="cosine", step_pattern="asymmetric",
             open_begin=True, open_end=True, distance_only=False)
    qi, ri = np.asarray(al.index1), np.asarray(al.index2)
    # map query/ref path indices back to (cover_frame, orig_frame)
    cov_f, orig_f = (qi, ri) if cover_is_query else (ri, qi)
    return {
        "norm_distance": float(al.normalizedDistance),
        "cover_idx": cov_f, "orig_idx": orig_f,          # aligned frame indices (paired)
        "cover_sec": frames_to_sec(cov_f), "orig_sec": frames_to_sec(orig_f),
        "cover_is_query": cover_is_query, "transpose_semitones": semis,
        "cover_span_sec": (float(frames_to_sec(cov_f.min())), float(frames_to_sec(cov_f.max()))),
        "orig_span_sec": (float(frames_to_sec(orig_f.min())), float(frames_to_sec(orig_f.max()))),
    }


def run_alignments(pairs=None, save=True):
    """Align every pair from cached chroma. Saves per-pair .npz warping paths and a
    summary CSV under alignments/. Returns a list of summary dicts."""
    if pairs is None:
        pairs = load_pairs()
    os.makedirs(ALIGNDIR, exist_ok=True)
    summary = []
    for idx, name, cov, orig in pairs:
        Cc = chroma_cached(idx, "cover", cov)
        Co = chroma_cached(idx, "orig", orig)
        r = align(Cc, Co)
        if save:
            np.savez_compressed(
                os.path.join(ALIGNDIR, f"{idx:03d}.npz"),
                cover_idx=r["cover_idx"], orig_idx=r["orig_idx"],
                cover_sec=r["cover_sec"], orig_sec=r["orig_sec"],
                norm_distance=r["norm_distance"], transpose_semitones=r["transpose_semitones"],
                name=name)
        summary.append({
            "idx": idx, "song": name, "norm_distance": round(r["norm_distance"], 4),
            "transpose": r["transpose_semitones"],
            "cover_dur": round(Cc.shape[1] / FPS, 1), "orig_dur": round(Co.shape[1] / FPS, 1),
            "cover_span": f"{r['cover_span_sec'][0]:.0f}-{r['cover_span_sec'][1]:.0f}s",
            "orig_span":  f"{r['orig_span_sec'][0]:.0f}-{r['orig_span_sec'][1]:.0f}s",
            "cover_is_query": r["cover_is_query"],
        })
    if save:
        import csv as _csv
        with open(os.path.join(ALIGNDIR, "alignment_summary.csv"), "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader(); w.writerows(summary)
    return summary


def _rdp_keep(pts, eps):
    """Ramer–Douglas–Peucker: indices of points to keep so the polyline stays
    within eps of the original. Iterative (safe for long DTW paths)."""
    n = len(pts)
    keep = np.zeros(n, bool); keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        p0, d = pts[i], pts[j] - pts[i]
        L = float(np.hypot(*d))
        seg = pts[i + 1:j] - p0
        dist = (np.hypot(*seg.T) if L == 0
                else np.abs(seg[:, 0] * d[1] - seg[:, 1] * d[0]) / L)
        k = int(np.argmax(dist))
        if dist[k] > eps:
            m = i + 1 + k; keep[m] = True
            stack.append((i, m)); stack.append((m, j))
    return np.where(keep)[0]


def segments(cover_sec, orig_sec, tol=1.5, gap=1.5):
    """Convert a dense warping path into matching time intervals.

    Returns a list of dicts, each a segment with corresponding KR (original) and
    ZH (cover) spans. `kind`:
      - 'aligned'    : both cover the same passage (tempo_ratio = zh_dur / kr_dur ~ 1)
      - 'cover_extra': ZH spans much more time than the matched KR (cover-only material
                       or a passage the original compresses/omits)
      - 'orig_extra' : KR spans much more time than the matched ZH (e.g. a rap/interlude
                       the cover skips or compresses)
    `tol` = piecewise-linear fit tolerance (s); larger = fewer, coarser segments.
    `gap` = min duration difference (s) to call a segment lopsided.
    """
    c = np.asarray(cover_sec, float); o = np.asarray(orig_sec, float)
    order = np.argsort(c, kind="stable"); c, o = c[order], o[order]
    pts = np.column_stack([c, o])
    idx = _rdp_keep(pts, tol)
    segs = []
    for a, b in zip(idx[:-1], idx[1:]):
        zs, ze, ks, ke = c[a], c[b], o[a], o[b]
        dz, dk = ze - zs, ke - ks
        if dz - dk >= gap and dz >= 2 * dk:
            kind = "cover_extra"      # ZH span >> KR span
        elif dk - dz >= gap and dk >= 2 * dz:
            kind = "orig_extra"       # KR span >> ZH span
        else:
            kind = "aligned"
        segs.append({
            "kind": kind,
            "kr_start": round(float(ks), 2), "kr_end": round(float(ke), 2),
            "zh_start": round(float(zs), 2), "zh_end": round(float(ze), 2),
            "kr_dur": round(float(dk), 2), "zh_dur": round(float(dz), 2),
            "tempo_ratio": round(float(dz / dk), 3) if dk > 0.05 else None,
        })
    return segs


def save_segments(pairs=None, tol=1.5):
    """Write matching intervals for every pair: segments/{idx:03d}.csv + segments.json."""
    import csv as _csv, json as _json
    if pairs is None:
        pairs = load_pairs()
    segdir = os.path.join(BASE, "segments")
    os.makedirs(segdir, exist_ok=True)
    combined = {}
    for idx, name, cov, orig in pairs:
        z = np.load(os.path.join(ALIGNDIR, f"{idx:03d}.npz"), allow_pickle=True)
        segs = segments(z["cover_sec"], z["orig_sec"], tol=tol)
        combined[f"{idx:03d}"] = {"song": name, "segments": segs}
        with open(os.path.join(segdir, f"{idx:03d}.csv"), "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(segs[0].keys()))
            w.writeheader(); w.writerows(segs)
    with open(os.path.join(segdir, "segments.json"), "w") as f:
        _json.dump(combined, f, ensure_ascii=False, indent=1)
    return combined


def _worker(args):
    idx, name, cov, orig = args
    try:
        chroma_cached(idx, "cover", cov)
        chroma_cached(idx, "orig", orig)
        return (idx, "ok")
    except Exception as e:
        return (idx, f"ERR {type(e).__name__}: {e}")


if __name__ == "__main__":
    # Full batch: parallel-cache chroma -> align every pair -> write matching intervals.
    import multiprocessing as mp
    pairs = load_pairs()
    print(f"caching chroma for {len(pairs)} pairs ({2*len(pairs)} files) -> {FEATDIR}", flush=True)
    with mp.Pool(min(12, os.cpu_count())) as pool:
        for i, (idx, status) in enumerate(pool.imap_unordered(_worker, pairs), 1):
            print(f"  [{i:02d}/{len(pairs)}] pair {idx:03d}: {status}", flush=True)
    print(f"aligning {len(pairs)} pairs -> {ALIGNDIR}", flush=True)
    summary = run_alignments(pairs, save=True)
    save_segments(pairs)
    import statistics
    print(f"done. median DTW cost {statistics.median(s['norm_distance'] for s in summary):.4f}; "
          f"wrote alignment_summary.csv, {{idx}}.npz, segments/", flush=True)

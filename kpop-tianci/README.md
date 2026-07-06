# Korean ↔ Mandarin K-pop 填詞 bitext

Tools for building a **Korean-original ↔ Mandarin-cover** lyric dataset from Bilibili
**填詞** (fan lyric-rewrite) covers of K-pop songs. For each cover we align it to the
Korean original, read its burned-in Mandarin subtitles, and produce line-level
Korean↔Mandarin pairs enriched with romanization, Traditional Chinese, and English.

The **audio/video data is not included** (it isn't ours to redistribute). Point the code
at your own copy with the `KPOP_DATA` environment variable:

```bash
export KPOP_DATA=/path/to/bilibili_kpop_tianci
```

Expected layout under `$KPOP_DATA`: `covers/`, `originals/`, `covers_meta.json`,
`manifest.csv`, and (produced by the pipeline) `features/`, `alignments/`, `segments/`,
`ocr/`, `kr_lyrics/`, `bitext/`.

Run scripts from this folder (or with `$KPOP_DATA` set — they don't depend on the repo's
working directory otherwise). See the [repo root README](../README.md) for the shared
conda environments.

## Pipeline

| Step | Script | Env | Output |
|------|--------|-----|--------|
| 1. Audio alignment | `align_lib.py` | music-translation | `alignments/`, `segments/` (CQT chromagram + optimal-transposition + subsequence DTW → warping path & matching intervals) |
| 2. Lyric OCR | `vlm_lyrics.py` | qwen | `ocr/lyrics/<idx>.jsonl` (Qwen3-VL reads the burned-in Mandarin subtitles) |
| 3. Map to KR timeline | `map_lyrics_to_kr.py` | music-translation | `ocr/lyrics_aligned/<idx>.csv` (Mandarin lines projected onto the Korean timeline) |
| 4. Fetch original lyrics | `scrape_lyrics.py` | music-translation | `kr_lyrics/<idx>.txt` (original song lyrics from Genius — canonical-URL + search fallback) |
| 5. KR lyric forced-align | `align_kr_lyrics.py` | qwen | `kr_lyrics/<idx>.jsonl` (lyric text → per-line timings on the Korean audio) |
| 6. Build bitext | `build_bitext.py` | music-translation | `bitext/<idx>.csv` + combined `bitext.csv` (korean, korean_rr, zh_hans, zh_hant, english) |

`dataset_walkthrough.ipynb` walks through the whole pipeline on one song end-to-end (outputs
are stripped in the repo — run it locally to render). To run step 1 over the whole set,
`python align_lib.py` caches chroma, aligns every pair, and writes the matching intervals.

## Notes

- Only genuine Mandarin-sung 中文/填詞 covers are used; English- or Korean-language covers in
  the source playlist are flagged (`language` column in `manifest.csv`) and set aside.
- Lyric text is used for research/alignment; the source recordings are not redistributed here.

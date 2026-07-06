# Lyric-to-lyric translation alignment

Tools for turning song collections into time-aligned, line-level translation bitext:
align a recording to a reference (or to another recording of the same song), read or
fetch the lyrics, forced-align them to the audio timeline, and emit line-level
translation pairs. The methodology isn't tied to one language pair or genre — this repo
currently hosts two sub-projects that share it.

## Sub-projects

- **[`kpop-tianci/`](kpop-tianci/)** — builds a **Korean-original ↔ Mandarin-cover**
  bitext from Bilibili **填詞** (fan lyric-rewrite) covers of K-pop songs: audio
  alignment, burned-in-subtitle OCR, forced alignment, and bitext construction.
- **[`expand-lyrics-audio/`](expand-lyrics-audio/)** — takes existing lyric datasets
  that are missing audio (received from other researchers, or pulled from sources like
  Hugging Face) and locates/downloads a matching recording for each entry so they can be
  run through the same alignment pipeline.

See each folder's README for pipeline details, expected data layout, and environments.

## Environments

- **`music-translation`** — `librosa`, `dtw-python`, `numpy`, `pandas`, `matplotlib`,
  `opencc-python-reimplemented`, `korean-romanizer` (alignment + join, CPU).
- **`qwen`** — `torch`, `transformers`, `ctc-forced-aligner` (Qwen3-VL OCR and forced
  alignment, GPU).

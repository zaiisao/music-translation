# Expand existing lyric datasets to include audio

Several public/shared lyric datasets ship text (and sometimes translations) without a
matching recording for every entry. This sub-project locates or downloads audio for
those entries so the datasets can be run through the same align → OCR/lyrics →
forced-align → bitext pipeline used in [`../kpop-tianci/`](../kpop-tianci/).

None of the datasets below are redistributed here (see `.gitignore`) — each folder is
local-only.

## Datasets

- **`dataset_share/`** — 1,000-song lyric dataset shared by another researcher (Haven
  Kim), mostly K-pop with some Disney/musical/J-pop/hymn/anime entries. `meta.csv` has
  one row per song (`LID, Artist, English, Korean, is_official, url1, url2, Genre`);
  `lyrics/<LID><lang>.txt` holds the lyric text per language. Audio is sparse —
  `url1`/`url2` are empty or non-YouTube/SoundCloud for a large fraction of rows.
- **`aligned/`** — a second received dataset: 162 EN/JP/KR trilingual songs (identified
  from their lyrics).
- **`aligned_to_dataset_share_mapping.csv`** — best-match mapping between `aligned/` and
  `dataset_share/` entries (same underlying song, different dataset), with a match score.
- **`mandarin_official_versions.md`** — analysis of which songs in `dataset_share/` and
  `aligned/` have an official Mandarin version (Disney dub, K-pop act's own Mandarin
  release, or licensed stage production).
- **`mavl/`** — the [Noename/MAVL](https://huggingface.co/datasets/Noename/MAVL) dataset
  from Hugging Face (CC-BY-NC-4.0): 228 Disney songs with dubbed versions across
  US/ES/FR/JP/KR, each with a `youtube_url` and `lyrics_url` per language. Not yet run
  through any script here — it's a candidate for the same audio-resolution treatment.

## Scripts

| Script | Purpose |
|--------|---------|
| `resolve_missing_audio.py` | For `dataset_share/meta.csv` rows with no usable audio URL, searches YouTube (`yt-dlp ytsearch`, metadata only) and scores candidates by title/channel/duration match. Writes `dataset_share/audio_resolution.csv` for human review — does not download anything. |
| `download_dataset_share_audio.py` | Downloads audio for `dataset_share/meta.csv` rows that already have (or were resolved to have) a YouTube/SoundCloud URL, into `dataset_share/audio/<LID>.<ext>`. Safe to re-run; skips rows already downloaded. Logs to `dataset_share/download_log.csv`. |

`resolve_missing_audio.py` only proposes candidates (`auto-accept` / `needs-review` /
`no-match`) — review `audio_resolution.csv` before feeding accepted URLs back into
`meta.csv` / `download_dataset_share_audio.py`.

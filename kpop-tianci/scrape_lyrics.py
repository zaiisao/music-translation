#!/usr/bin/env python3
"""Fetch each cover song's ORIGINAL lyrics from Genius into kr_lyrics/<idx>.txt
(one sung line per line; Korean in Hangul, in-song English kept as-is).

These feed align_kr_lyrics.py -> build_bitext.py. The English translation is NOT
scraped — fan translations don't line up 1:1 with the sung lines, and build_bitext
treats english as optional (generate .en.txt separately, e.g. by per-line MT).

Resolution: Genius has no lyrics API, so we (1) try the canonical song URL built from
artist+title, and (2) fall back to Genius' public search JSON — deriving the original page
from a romanization/translation "bot" hit when the original isn't surfaced directly. Lyrics
come from the on-page data-lyrics-container. Be considerate: this hits a third-party site —
it is rate-limited and caches raw HTML; use it for research and mind robots.txt / ToS.

Run:  conda run -n music-translation python scrape_lyrics.py            # Chinese covers missing lyrics
      conda run -n music-translation python scrape_lyrics.py --idx 51   # one song
      conda run -n music-translation python scrape_lyrics.py --overwrite
"""
import os, re, csv, json, time, argparse, difflib
import requests
from bs4 import BeautifulSoup

BASE = os.environ.get("KPOP_DATA", "data")
LYR = os.path.join(BASE, "kr_lyrics")
CACHE = os.path.join(LYR, "_html_cache")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
           "Accept-Language": "ko,en;q=0.9"}
# romanization / translation / alt-version markers (multiple languages)
_ALT = re.compile(r"romaniz|instrumental|cappella|sped up|remix|inst\.|"
                  r"translat|traduc|traduz|übersetz|ubersetz|перевод|翻[譯译]|"
                  r"japanese|chinese|español|espanol|deutsch|fran[çc]ais|português|portugues|italiano", re.I)

def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())

def _get(url):
    return requests.get(url, headers=HEADERS, timeout=25)

def _canonical(artist, title):
    """Best-guess canonical Genius URL, e.g. 'Red Velvet','Cosmic' -> Red-velvet-cosmic-lyrics."""
    s = re.sub(r"\(feat[^)]*\)", "", f"{artist} {title}", flags=re.I).replace("&", "and").replace("'", "")
    s = re.sub(r"[^0-9A-Za-z가-힣]+", " ", s).strip()
    s = re.sub(r"\s+", "-", s)
    return f"https://genius.com/{s[:1].upper() + s[1:].lower()}-lyrics"

def _derive_original(bot_url):
    """From a romanization/translation page URL, recover the original song URL."""
    slug = bot_url.rstrip("/").rsplit("/", 1)[-1]
    slug = re.sub(r"-lyrics$", "", slug)
    slug = re.sub(r"^genius-(romanizations|traducciones-al-espanol|[a-z-]+-translations)-", "", slug)
    slug = re.sub(r"-(romanized|english-translation|traduccion-al-espanol|[a-z-]+-translation)$", "", slug)
    return f"https://genius.com/{slug[:1].upper() + slug[1:]}-lyrics"

def _search_url(artist, title):
    """Best Genius URL from the search JSON (prefers the real original; else derives it)."""
    bare = re.sub(r"\(feat[^)]*\)", "", title, flags=re.I).strip()
    j = _get(f"https://genius.com/api/search/multi?q={requests.utils.quote(artist + ' ' + bare)}").json()
    best, best_score, best_is_bot = None, -9.0, False
    for sec in j["response"]["sections"]:
        for h in sec.get("hits", []):
            if h["type"] != "song":
                continue
            res = h["result"]
            a, t = res["primary_artist"]["name"], res["title"]
            is_bot = "genius" in a.lower() or bool(_ALT.search(t) or _ALT.search(a))
            score = (difflib.SequenceMatcher(None, _norm(a), _norm(artist)).ratio()
                     + difflib.SequenceMatcher(None, _norm(t), _norm(title)).ratio()
                     - (0.4 if is_bot else 0.0))       # mild penalty: keep bots as a derivable fallback
            if score > best_score:
                best_score, best, best_is_bot = score, res, is_bot
    if not best:
        return None
    return _derive_original(best["url"]) if best_is_bot else best["url"]

def resolve(artist, title):
    """Return (html, url) for the original lyrics page, or (None, None)."""
    for url in (_canonical(artist, title), _search_url(artist, title)):
        if not url:
            continue
        r = _get(url)
        if r.status_code == 200 and 'data-lyrics-container="true"' in r.text:
            return r.text, url
    return None, None

def extract_lyrics(html):
    """Pull clean lyric lines from a Genius page's data-lyrics-container(s)."""
    soup = BeautifulSoup(html, "html.parser")
    parts = []
    for b in soup.select('[data-lyrics-container="true"]'):
        for br in b.find_all("br"):
            br.replace_with("\n")
        parts.append(b.get_text())
    text = "\n".join(parts)
    text = re.sub(r"^.*?Lyrics(\[|\n)", r"\1", text, count=1, flags=re.S)   # strip contributor/header preamble
    out = []
    for ln in text.split("\n"):
        ln = re.sub(r"\s+", " ", ln).strip()
        ln = re.sub(r"^\[[^\]]*\]\s*", "", ln)          # inline [Section] label
        ln = re.sub(r"\d*Embed$", "", ln).strip()       # trailing Genius "…Embed"
        if not ln or re.match(r"^\[.*\]$", ln):
            continue
        if "Contributors" in ln or ln.startswith("You might also like") or ln.isdigit():
            continue
        out.append(ln)
    return out

def scrape(idx, song):
    artist, _, title = song.partition(" - ")
    title = title or song
    html, url = resolve(artist, title)
    if not html:
        return {"idx": idx, "ok": False, "msg": "could not resolve Genius page"}
    os.makedirs(CACHE, exist_ok=True)
    open(os.path.join(CACHE, f"{idx:03d}.html"), "w").write(html)
    lines = extract_lyrics(html)
    if not lines:
        return {"idx": idx, "ok": False, "msg": "no lyrics parsed", "url": url}
    os.makedirs(LYR, exist_ok=True)
    open(os.path.join(LYR, f"{idx:03d}.txt"), "w").write("\n".join(lines) + "\n")
    han = sum(bool(re.search(r"[가-힣]", l)) for l in lines)
    r = {"idx": idx, "ok": True, "lines": len(lines), "hangul_lines": han, "url": url}
    if han < max(3, len(lines) * 0.15):
        r["flag"] = "few Hangul lines — verify match (or the original is English)"
    return r

def target_songs():
    """Chinese-cover idx -> 'Artist - Title', from covers_lang.csv (fallback: all aligned)."""
    lp = os.path.join(BASE, "covers_lang.csv")
    if os.path.exists(lp):
        return {int(r["idx"]): r["original"] for r in csv.DictReader(open(lp)) if r["language"] == "chinese"}
    ap = os.path.join(BASE, "alignments", "alignment_summary.csv")
    return {int(r["idx"]): r["song"] for r in csv.DictReader(open(ap))}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, help="single song index")
    ap.add_argument("--overwrite", action="store_true", help="re-fetch songs that already have kr_lyrics/<idx>.txt")
    ap.add_argument("--sleep", type=float, default=1.5, help="seconds between songs (be polite)")
    args = ap.parse_args()
    songs = target_songs()
    targets = [args.idx] if args.idx else sorted(songs)
    flagged, done, fail = [], 0, 0
    for i in targets:
        if i not in songs:
            print(json.dumps({"idx": i, "ok": False, "msg": "not a Chinese cover"})); continue
        if not args.overwrite and os.path.exists(os.path.join(LYR, f"{i:03d}.txt")):
            print(json.dumps({"idx": i, "ok": True, "msg": "cached (skip)"})); continue
        try:
            r = scrape(i, songs[i])
        except Exception as e:
            r = {"idx": i, "ok": False, "error": f"{type(e).__name__}: {e}"}
        done += r.get("ok", False); fail += not r.get("ok", False)
        if r.get("flag"):
            flagged.append(i)
        print(json.dumps(r, ensure_ascii=False), flush=True)
        time.sleep(args.sleep)
    print(json.dumps({"done": done, "fail": fail, "flagged_for_review": flagged}), flush=True)

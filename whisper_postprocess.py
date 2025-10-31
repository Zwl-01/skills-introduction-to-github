import re
from pathlib import Path
from typing import List, Sequence, Tuple

from faster_whisper import WhisperModel
from tqdm import tqdm

# ========== 可调参数 ==========
AUDIO_DIR = r"D:\\whisper\\audios"
OUT_DIR = r"D:\\whisper\\srt"
MODEL = r"D:\\whisper\\models\\faster-whisper-medium"
LANG = "en"

# 字幕切分逻辑参数
GAP_THRESHOLD = 0.60  # 语气停顿阈值（秒）
MERGE_TOO_SHORT = 0.60  # 小于此时长的片段会合并
MIN_DURATION = 1.00  # 字幕最短时长（防闪）
MAX_CHARS_PER_LINE = 27
MAX_LINE_TOTAL = 48

# 文本清理与风格参数
STRIP_PUNCTUATION = True
TITLE_CASE_EACH_WORD = True
SMART_SPLIT_BY_PUNCT = True  # ✨语法级切分

# ====== NEW: 重复控制相关 ======
SKIP_IF_SRT_UP_TO_DATE = True  # 已有SRT且新于音频 => 跳过
DEDUP_ADJACENT_LINES = True  # 相邻重复字幕合并
# ============================

punct_re = re.compile(r"[，。？！；：“”‘’、…,.!?;:\"'—-]")


def format_time(t: float) -> str:
    ms = int(round((t - int(t)) * 1000))
    s = int(t)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _word_wrap(text: str, max_per_line: int) -> List[str]:
    """将文本按单词边界换行，避免把单词截断。"""
    words = text.split()
    if not words:
        return []

    lines: List[str] = []
    current: List[str] = []
    for word in words:
        tentative = " ".join(current + [word]) if current else word
        if current and len(tentative) > max_per_line:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def wrap_lines(text: str, max_per_line: int = MAX_CHARS_PER_LINE) -> List[str]:
    text = text.strip()
    if len(text) <= max_per_line:
        return [text]

    wrapped = _word_wrap(text, max_per_line)
    # 如果单词太长导致长度仍超限，最后再 fallback 按字符截断。
    normalized = []
    for line in wrapped:
        if len(line) <= max_per_line:
            normalized.append(line)
            continue
        normalized.extend([line[i : i + max_per_line] for i in range(0, len(line), max_per_line)])
    return normalized


def normalize_text(text: str) -> str:
    """最小化清理：仅压缩空白，保留标点以便后续比较与切分。"""

    return re.sub(r"\s+", " ", text).strip()


def stylize_text(text: str) -> str:
    """输出阶段的风格处理，可去除标点或首字母大写。"""

    t = normalize_text(text)
    if not t:
        return ""
    if STRIP_PUNCTUATION:
        t = punct_re.sub("", t)
    if TITLE_CASE_EACH_WORD:
        def tc(word: str) -> str:
            return word.capitalize() if re.match(r"^[A-Za-z][A-Za-z\-']*$", word) else word

        t = " ".join(tc(w) for w in t.split(" "))
    return t


def merge_split_words(words: Sequence[dict], max_gap: float = 0.10) -> List[dict]:
    """把同一单词被分裂的碎片在极短间隔内合并"""
    merged: List[dict] = []
    for w in words:
        if not merged:
            merged.append(w)
            continue
        prev = merged[-1]
        if not prev["word"].endswith(" ") and not w["word"].startswith(" "):
            if (w["start"] - prev["end"]) < max_gap:
                prev["word"] += w["word"]
                prev["end"] = w["end"]
                continue
        merged.append(w)
    return merged


def prosody_segments(words: Sequence[dict], gap: float = GAP_THRESHOLD) -> List[Tuple[float, float, str]]:
    """根据语气停顿切分（切之前先合并被拆开的单词）"""
    words = merge_split_words(words)
    segs: List[Tuple[float, float, str]] = []
    buf: List[dict] = []
    for w in words:
        if not buf:
            buf.append(w)
            continue
        delta = w["start"] - buf[-1]["end"]
        if delta > gap:
            text = "".join(x["word"] for x in buf)
            segs.append((buf[0]["start"], buf[-1]["end"], text))
            buf = [w]
        else:
            buf.append(w)
    if buf:
        text = "".join(x["word"] for x in buf)
        segs.append((buf[0]["start"], buf[-1]["end"], text))
    return segs


def _allocate_times(chunks: List[str], start: float, end: float) -> List[Tuple[float, float]]:
    duration = end - start
    char_lengths = [len(chunk) for chunk in chunks]
    total_chars = sum(char_lengths)
    if total_chars == 0:
        return [(start, end)] * len(chunks)

    allocated: List[Tuple[float, float]] = []
    cursor = start
    for i, length in enumerate(char_lengths):
        if i == len(chunks) - 1:
            allocated.append((cursor, end))
            break
        portion = duration * (length / total_chars)
        next_cursor = cursor + portion
        allocated.append((cursor, next_cursor))
        cursor = next_cursor
    return allocated


def _split_with_punct(text: str, start: float, end: float, max_total: int) -> List[Tuple[float, float, str]]:
    puncts = [".", ",", "!", "?", ";", "，", "。", "！", "？", "；"]
    split_points = [i for i, c in enumerate(text) if c in puncts and i > max_total * 0.4]
    if not split_points:
        return []

    split_idx = min(split_points, key=lambda x: abs(x - max_total))
    left = text[: split_idx + 1].strip()
    right = text[split_idx + 1 :].strip()
    if not left or not right:
        return []

    chunks = [left, right]
    timings = _allocate_times(chunks, start, end)
    return [(timings[i][0], timings[i][1], chunk) for i, chunk in enumerate(chunks)]


def split_by_total_chars(seg_text: str, start: float, end: float, max_total: int) -> List[Tuple[float, float, str]]:
    """
    ✨ 优先按语法标点切分，若无合适标点再按单词分块，避免截断单词
    """
    text = seg_text.strip()
    if len(text) <= max_total:
        return [(start, end, text)]

    if SMART_SPLIT_BY_PUNCT:
        punct_split = _split_with_punct(text, start, end, max_total)
        if punct_split:
            return punct_split

    words = text.split()
    if len(words) <= 1:
        # 单词异常长，只能硬拆
        mid = start + (end - start) / 2
        return [(start, mid, text[:max_total]), (mid, end, text[max_total:])]

    chunks: List[str] = []
    current: List[str] = []
    for word in words:
        tentative = " ".join(current + [word]) if current else word
        if current and len(tentative) > max_total:
            chunks.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        chunks.append(" ".join(current))

    timings = _allocate_times(chunks, start, end)
    return [(timings[i][0], timings[i][1], chunk) for i, chunk in enumerate(chunks)]


def merge_too_short(segs: Sequence[Tuple[float, float, str]], min_dur: float = MIN_DURATION) -> List[Tuple[float, float, str]]:
    if not segs:
        return []
    out: List[Tuple[float, float, str]] = [segs[0]]
    for s, e, t in segs[1:]:
        if (e - s) < min_dur and out:
            ps, pe, pt = out[-1]
            out[-1] = (ps, e, (pt + " " + t).strip())
        else:
            out.append((s, e, t))
    return out


# ====== NEW: 相邻重复去重 ======
def dedupe_adjacent(segs: Sequence[Tuple[float, float, str]]) -> List[Tuple[float, float, str]]:
    """
    相邻两条清洗后的文本相同 -> 合并时间窗
    """
    if not segs:
        return []
    out: List[Tuple[float, float, str]] = []
    for s, e, t in segs:
        current_norm = normalize_text(t)
        if out and current_norm and normalize_text(out[-1][2]) == current_norm:
            ps, pe, pt = out[-1]
            out[-1] = (ps, e, pt)
        else:
            out.append((s, e, t))
    return out


def postprocess_and_write(segments: Sequence[Tuple[float, float, str]], out_path: Path) -> None:
    tmp: List[Tuple[float, float, str]] = []
    for s, e, t in segments:
        normalized = normalize_text(t)
        if not normalized:
            continue
        parts = split_by_total_chars(normalized, s, e, MAX_LINE_TOTAL)
        tmp.extend(parts)

    tmp = merge_too_short(tmp, MIN_DURATION)

    if DEDUP_ADJACENT_LINES:
        tmp = dedupe_adjacent(tmp)

    final_entries: List[Tuple[float, float, str]] = []
    for s, e, t in tmp:
        styled = stylize_text(t)
        if not styled:
            continue
        final_entries.append((s, e, styled))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, (s, e, t) in enumerate(final_entries, 1):
            lines = wrap_lines(t)
            f.write(f"{i}\n{format_time(s)} --> {format_time(e)}\n")
            for line in lines:
                f.write(line.strip() + "\n")
            f.write("\n")


def process_file(model: WhisperModel, audio_path: Path, out_dir: Path) -> None:
    out_path = out_dir / (audio_path.stem + ".srt")

    if SKIP_IF_SRT_UP_TO_DATE and out_path.exists():
        try:
            srt_mtime = out_path.stat().st_mtime
            audio_mtime = audio_path.stat().st_mtime
            if srt_mtime >= audio_mtime and out_path.stat().st_size > 0:
                print(f"[→] Skip (up-to-date): {audio_path.name}")
                return
        except Exception:
            pass

    print(f"[+] Transcribing: {audio_path.name}")
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=LANG,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
    )

    all_words: List[dict] = []
    segments_list = []
    total_duration = getattr(info, "duration", None)
    progress = tqdm(total=total_duration or 100, unit="s", desc=f"{audio_path.name}")
    last_t = 0.0

    for seg in segments_iter:
        segments_list.append(seg)
        if total_duration:
            progress.update(max(0, seg.end - last_t))
        last_t = seg.end
        if seg.words:
            for w in seg.words:
                if w.start is not None and w.end is not None:
                    all_words.append({"word": w.word, "start": w.start, "end": w.end})

    progress.close()

    if not all_words:
        segments = [(seg.start, seg.end, seg.text) for seg in segments_list]
    else:
        ps = prosody_segments(all_words, GAP_THRESHOLD)
        segments = [(s, e, t) for (s, e, t) in ps]

    postprocess_and_write(segments, out_path)
    print(f"[✓] Saved: {out_path}")


def main() -> None:
    model = WhisperModel(MODEL, device="cpu", compute_type="int8")
    in_dir, out_dir = Path(AUDIO_DIR), Path(OUT_DIR)
    exts = {".mp3", ".wav", ".m4a", ".mp4", ".flac", ".aac", ".ogg", ".wma", ".webm"}
    files = [p for p in in_dir.glob("*") if p.suffix.lower() in exts]
    if not files:
        print(f"[!] No audio found in {AUDIO_DIR}")
        return
    for p in files:
        process_file(model, p, out_dir)


if __name__ == "__main__":
    main()

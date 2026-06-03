#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pii_fixer_pipeline.py — script chạy trực tiếp: python pii_fixer_pipeline.py
# Cần: pip install openai pydantic ; điền GEMINI_API_KEY = key vilao.ai trong phần Cấu hình.

# Chỉ cần khi chạy TẦNG D (gọi LLM qua cổng OpenAI-compatible). Bỏ ghi chú để cài:
# %pip install -q openai pydantic
import os

# --- Đường dẫn ---
INPUT_DIR   = "data/in"                                   # thư mục các shard .json gốc
OUTPUT_DIR  = "data/out"                                  # nơi ghi shard đã sửa + audit
INPUT_GLOB  = "records_p8_s0of2_f4200-4500_gemma-3-4b-it.json"                            # mẫu tên shard
SAMPLE_FILE = "records_p1_s0of2_f0-200_Qwen3-8B.json"     # shard mẫu để self-test (đặt cạnh notebook)

# --- Mô hình LLM (gọi qua cổng trung gian OpenAI-compatible: vilao.ai) ---
API_BASE_URL   = "https://api.vilao.ai/v1"               # cổng trung gian (OpenAI-compatible)
GEMINI_MODEL   = "ts/gemini-3.1-flash-lite"              # id model trên vilao (giữ tiền tố "ts/")
GEMINI_API_KEY = "sk-16edc9fe49bf4e284835215c6b5a8d28b67aa590bc40d5c1b6dea10d94b976dd"                    # <-- điền API key của vilao.ai vào đây

# --- Điều khiển chi phí / hành vi ---
USE_GEMINI           = bool(GEMINI_API_KEY)  # tự bật nếu có key; nếu tắt → escalation chỉ được ghi log
BATCH_SIZE           = 1       # số hồ sơ gộp trong 1 lời gọi LLM
MAX_CONCURRENCY      = 20      # số request song song tối đa
ESCALATE_ALL         = True    # True = gửi MỌI hồ sơ cho LLM (rất đắt! chỉ để so sánh / QA toàn bộ)
ESCALATE_SAMPLE_RATE = 0.0      # vd 0.02 = thêm 2% hồ sơ "sạch" cho LLM để QA chất lượng luật
USE_CONTEXT_CACHE    = True     # cache system prompt (best-effort; tự fallback nếu prompt quá ngắn)
THINKING_BUDGET      = 0        # tắt token "suy nghĩ" có tính phí
CHECKPOINT_FILE      = os.path.join(OUTPUT_DIR, "_llm_checkpoint.jsonl")

os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Gemini:", "BẬT" if USE_GEMINI else "TẮT (chỉ tầng xác định)", "| model:", GEMINI_MODEL)
"""
pii_fixer_core.py  —  Deterministic, span-aware detection + safe auto-fix
=========================================================================
The "free" (no-LLM) tier of the cost funnel. It catches high-confidence
label<->value mismatches in the *prose* of Vietnamese PII/SPII records and
fixes only the cases that are provably safe, routing everything ambiguous to
an `escalate` list for the Gemini stage.

KEY INSIGHT (validated on the sample shard): the generator's mistake is almost
always in the PROSE LABEL, while the annotated SPAN is correct. Example:
prose says "số điện thoại 035048152083" but 035048152083 is a 12-digit CCCD and
the span at that position is correctly labelled `cccd`.  Therefore:

  * RELABEL  (preferred when a span already annotates the value): rewrite the
             wrong label word(s) to match the value's true field. The value and
             its (correct) span are left untouched -> zero data loss, zero span
             breakage.
  * VALUE-FIX (only for a stray token that NO span covers): replace the
             out-of-place value with the ground-truth profile_fake[label] value.
  * ESCALATE everything else (contained-in-phrase edits, three-way
             disagreements, ambiguous/low-confidence, missing profile value).

No third-party deps. Pure Python.
"""
import re
import json
import unicodedata
from collections import defaultdict

# ---------------------------------------------------------------------------
# 1) Vietnamese closed vocabularies  (lowercased, NFC-normalized)
# ---------------------------------------------------------------------------
GENDERS = {"nam", "nữ"}
MARITAL = {"độc thân", "đã kết hôn", "ly hôn", "ly thân", "góa", "goá", "góa bụa"}
NATIONALITIES = {"việt nam"}
RELIGIONS = {
    "không tôn giáo", "phật giáo", "công giáo", "tin lành", "cao đài",
    "hòa hảo", "hoà hảo", "hồi giáo", "bà la môn", "tịnh độ cư sĩ",
    "minh sư đạo", "minh lý đạo", "bửu sơn kỳ hương", "tứ ân hiếu nghĩa",
}
# 54 official Vietnamese ethnic groups (lowercased). Enough to classify reliably.
ETHNICITIES = {
    "kinh", "tày", "thái", "mường", "khmer", "hoa", "nùng", "mông", "h'mông",
    "dao", "gia rai", "gia-rai", "ê đê", "ê-đê", "ba na", "ba-na", "xơ đăng",
    "xơ-đăng", "sán chay", "cơ ho", "cơ-ho", "chăm", "sán dìu", "hrê",
    "ra glai", "ra-glai", "mnông", "m'nông", "thổ", "stiêng", "khơ mú",
    "khơ-mú", "bru vân kiều", "cơ tu", "cơ-tu", "giáy", "tà ôi", "tà-ôi",
    "mạ", "co", "chơ ro", "chơ-ro", "xinh mun", "hà nhì", "chu ru", "chu-ru",
    "lào", "kháng", "la chí", "phù lá", "la hủ", "la ha", "pà thẻn", "lự",
    "ngái", "chứt", "lô lô", "mảng", "cơ lao", "bố y", "cống", "si la",
    "pu péo", "rơ măm", "brâu", "ơ đu", "giẻ triêng", "giẻ-triêng",
}

CLOSED = {
    "gender": GENDERS,
    "marital_status": MARITAL,
    "ethnicity": ETHNICITIES,
    "religion": RELIGIONS,
    "nationality": NATIONALITIES,
}

# Generic words that share spelling with a vocab entry but are usually NOT the
# attribute value (e.g. "khác"=other, "không"=no). Excluded from the matcher.
AMBIGUOUS_TOKENS = {"khác", "không"}

# Closed-vocab tokens too short / too collision-prone to value-fix safely when
# they have no span (e.g. "nam" collides with "Việt Nam"/"miền Nam"). These are
# still detected, but a *free* value-fix on them is escalated to the LLM.
RISKY_FREE_TOKENS = {"nam", "nữ", "co", "mạ", "lự", "dao", "hoa", "thổ", "lào"}

# ---------------------------------------------------------------------------
# 2) Strict value formats. Two roles:
#    - classify_value(): identify what a *bare token* looks like.
#    - span_format check (tight fields only): is a span's value well-formed?
# ---------------------------------------------------------------------------
STRICT_REGEX = {
    "dob": re.compile(r"^\d{2}/\d{2}/\d{4}$"),
    "phone": re.compile(r"^0[3-9]\d{8}$"),
    "cccd": re.compile(r"^0\d{11}$"),
    "passport_number": re.compile(r"^[A-Z]\d{7}$"),
    "email": re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$"),
    "vehicle_plate": re.compile(r"^\d{2}[A-Z]{1,2}[-\s]?\d{3,5}(\.\d{2,3})?$"),
    "driver_license": re.compile(r"^GPLX\d{6,12}$"),
    "eid_credentials": re.compile(r"^VNeID-\d{6,}$"),
    "biometric": re.compile(r"^BIO-\d{4,}$"),
    # bank_account is a bare 12-digit number (generator convention) and thus
    # OVERLAPS cccd (^0\d{11}$). Including it here means a 12-digit value
    # classifies as BOTH, so a "số tài khoản ngân hàng <12 digits>" is NOT
    # falsely flagged, while a span still pins the true field for relabelling.
    "bank_account": re.compile(r"^\d{12}$"),
}

# Only these fields get a span-value FORMAT check. Phrase-type fields
# (biometric, eid_credentials, driver_license, bank_account, family_relations,
# political_view, private_life, health_status, ...) are NEVER format-checked,
# because their spans legitimately contain descriptive text.
TIGHT_FORMAT_FIELDS = {"dob", "phone", "cccd", "passport_number", "email", "vehicle_plate"}

# Fields whose LABEL, when followed by a structured value of a *different* field,
# signals a prose error. tax_code is a FORM-only label (not a PII field) included
# to catch "mã số thuế là <biển số xe>".
STRUCTURED_LABEL_FIELDS = {
    "phone", "cccd", "passport_number", "email", "dob", "vehicle_plate",
    "driver_license", "bank_account", "eid_credentials", "biometric", "tax_code",
}
TAX_CODE_RE = re.compile(r"^\d{10}(\d{3})?$")

# Canonical Vietnamese label used when RELABELLING the prose to match a field.
CANONICAL_LABELS = {
    "phone": "số điện thoại",
    "cccd": "số căn cước công dân",
    "passport_number": "số hộ chiếu",
    "vehicle_plate": "biển số xe",
    "driver_license": "số giấy phép lái xe",
    "bank_account": "số tài khoản ngân hàng",
    "dob": "ngày sinh",
    "email": "địa chỉ email",
    "ethnicity": "dân tộc",
    "religion": "tôn giáo",
    "gender": "giới tính",
    "marital_status": "tình trạng hôn nhân",
    "nationality": "quốc tịch",
    "political_view": "quan điểm chính trị",
    "eid_credentials": "tài khoản định danh điện tử VNeID",
    "tax_code": "mã số thuế",
}


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", nfc(s).strip().lower())


def classify_value(tok: str):
    """Return the SET of canonical fields whose strict format `tok` matches.
    Empty set => unknown / free-text."""
    t = tok.strip().strip("\"'“”‘’.,;:()[]")
    out = set()
    for field, rx in STRICT_REGEX.items():
        if rx.match(t):
            out.add(field)
    low = norm(t)
    if low and low not in AMBIGUOUS_TOKENS:
        for field, vals in CLOSED.items():
            if low in vals:
                out.add(field)
    return out


# ---------------------------------------------------------------------------
# 3) Label lexicon: Vietnamese surface phrase -> canonical field.
#    Sorted longest-first so the most specific phrase wins.
# ---------------------------------------------------------------------------
LABEL_LEXICON = [
    ("tình trạng hôn nhân", "marital_status"),
    ("quan điểm chính trị", "political_view"),
    ("thành phần chính trị", "political_view"),
    ("tinh thần chính trị", "political_view"),
    ("khuynh hướng chính trị", "political_view"),
    ("số giấy phép lái xe", "driver_license"),
    ("giấy phép lái xe", "driver_license"),
    ("bằng lái", "driver_license"),
    ("gplx", "driver_license"),
    ("số tài khoản ngân hàng", "bank_account"),
    ("tài khoản ngân hàng", "bank_account"),
    ("số tài khoản", "bank_account"),
    ("tài khoản định danh điện tử", "eid_credentials"),
    ("định danh điện tử", "eid_credentials"),
    ("số định danh cá nhân", "cccd"),
    ("định danh cá nhân", "cccd"),
    ("số định danh", "cccd"),
    ("căn cước công dân", "cccd"),
    ("số căn cước", "cccd"),
    ("căn cước", "cccd"),
    ("số cccd", "cccd"),
    ("cccd", "cccd"),
    ("số cmnd", "cccd"),
    ("cmnd", "cccd"),
    ("chứng minh nhân dân", "cccd"),
    ("chứng minh thư nhân dân", "cccd"),
    ("chứng minh thư", "cccd"),
    ("chứng minh", "cccd"),
    ("số hộ chiếu", "passport_number"),
    ("hộ chiếu", "passport_number"),
    ("passport", "passport_number"),
    ("số điện thoại", "phone"),
    ("điện thoại", "phone"),
    ("telephone", "phone"),
    ("biển số xe", "vehicle_plate"),
    ("biển kiểm soát", "vehicle_plate"),
    ("đăng ký xe", "vehicle_plate"),
    ("biển số", "vehicle_plate"),
    ("ngày tháng năm sinh", "dob"),
    ("ngày sinh", "dob"),
    ("năm sinh", "dob"),
    ("sinh năm", "dob"),
    ("sinh ngày", "dob"),
    ("địa chỉ email", "email"),
    ("địa chỉ thư điện tử", "email"),
    ("thư điện tử", "email"),
    ("e-mail", "email"),
    ("email", "email"),
    ("mã số thuế", "tax_code"),
    ("tôn giáo", "religion"),
    ("dân tộc", "ethnicity"),
    ("quốc tịch", "nationality"),
    ("giới tính", "gender"),
    ("hôn nhân", "marital_status"),
]
LABEL_LEXICON.sort(key=lambda kv: -len(kv[0]))

_VOCAB_SORTED = sorted(
    [(v, f) for f, vals in CLOSED.items() for v in vals],
    key=lambda kv: -len(kv[0]),
)

_WORD_BOUNDARY = re.compile(r"[0-9a-zA-ZÀ-ỹ]")


# ---------------------------------------------------------------------------
# 3b) VALUE-DRIVEN (span-anchored) detection support
#     Catches a correctly-spanned, identifiable VALUE that sits under a WRONG or
#     UNTRACKED prose label (e.g. "tên vợ là 36A-91523" where 36A-91523 has a
#     vehicle_plate span). The label-driven scan (section A) can't see these
#     because the prose label ("tên vợ", "số nhà", "giấy phép kinh doanh") is not
#     a tracked field alias. Here we trust the span and fix the prose label.
# ---------------------------------------------------------------------------
# Format-identifiable fields whose span pins the true type with high confidence.
VALUE_DRIVEN_FORMAT_FIELDS = {
    "vehicle_plate", "passport_number", "driver_license",
    "eid_credentials", "cccd", "phone", "bank_account", "dob", "email",
}
# Aliases that BOTH cccd and the VNeID e-ID legitimately share ("định danh").
# When such a label precedes an eid_credentials value we ESCALATE instead of
# auto-relabelling, because "số định danh cá nhân trong hệ thống VNeID" is
# genuinely ambiguous.
CCCD_EID_SHARED_ALIASES = {
    "định danh cá nhân", "số định danh cá nhân", "số định danh", "định danh",
}

# A genuine decoy label is a short NOUN phrase. It must start with one of these
# label-leading tokens and must not contain verb/function words (which would mean
# the region grabbed a sentence fragment, not a label).
_LABEL_START_TOKENS = {
    "số", "mã", "tên", "tài", "giấy", "biển", "địa", "ngày",
    "chứng", "thông", "hồ", "email", "e-mail",
}
_LABEL_BLOCK_WORDS = (
    "được", "cung cấp", "đề nghị", "ghi nhận", "gán", "của", "cũng",
    "còn", "như", "giữa", "khác biệt", "thêm", "nộp", "nhằm", "vì",
)

_CLAUSE_SEP = re.compile(r"[,;.:()\[\]\n]")
_LEAD_CONNECTORS = (
    "bao gồm", "gồm có", "gồm", "bên cạnh đó", "ngoài ra", "cùng với",
    "kèm theo", "kèm", "trong đó", "đồng thời", "và", "với", "cùng", "hoặc",
)
_LEAD_STRIP = re.compile(r"^(?:\s*(?:và|bản sao|bản chính|bản)\s+)+", re.IGNORECASE)
_TRAIL_STRIP = re.compile(r"(?:\s*(?:là|số|mã số|mã|gồm|:|=|-)\s*)+$", re.IGNORECASE)


def _governing_label(text: str, low: str, s: int, floor: int = 0, maxlen: int = 70):
    """Return (region_text, start, end, had_connector): the prose label phrase
    that governs the value starting at `s`, bounded to its own clause and never
    reaching before `floor` (e.g. the end of the preceding value span).
    had_connector is True when a real connector ('là'/'số'/'mã'/':'/'=') sat
    between the label and the value (a strong signal the phrase is a label)."""
    lo = max(0, floor, s - maxlen)
    start = lo
    for mm in _CLAUSE_SEP.finditer(text[lo:s]):          # last clause separator
        start = lo + mm.end()
    win_low = low[lo:s]
    for conn in _LEAD_CONNECTORS:                        # ...or last connector word
        k = win_low.rfind(conn + " ")
        if k != -1:
            cand = lo + k + len(conn) + 1
            if cand > start:
                start = cand
    end = s
    lm = _LEAD_STRIP.match(text[start:end])              # strip leading fillers
    if lm:
        start += lm.end()
    had_connector = False
    tm = _TRAIL_STRIP.search(text[start:end])            # strip trailing connectors
    if tm:
        if re.search(r"(là|số|mã|:|=)", tm.group(), re.IGNORECASE):
            had_connector = True
        end = start + tm.start()
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return text[start:end], start, end, had_connector


def _is_boundary(s: str, idx: int) -> bool:
    if idx >= len(s):
        return True
    return _WORD_BOUNDARY.match(s[idx]) is None


def value_constraint(field: str):
    if field in STRICT_REGEX:
        return ("regex", STRICT_REGEX[field])
    if field in CLOSED:
        return ("vocab", CLOSED[field])
    return None


def value_ok_for_field(value: str, field: str) -> bool:
    c = value_constraint(field)
    if c is None:
        return True
    v = value.strip().strip("\"'“”.,;:()[]")
    if c[0] == "regex":
        return c[1].match(v) is not None
    return norm(v) in c[1]


# ---------------------------------------------------------------------------
# helpers for span <-> position relations
# ---------------------------------------------------------------------------
def _spans_covering(rec, p0, p1):
    """Spans whose [start,end) intersects [p0,p1). Each item: (idx, span)."""
    out = []
    for k, sp in enumerate(rec.get("spans", [])):
        s, e = sp.get("start"), sp.get("end")
        if s is None or e is None:
            continue
        if not (e <= p0 or s >= p1):
            out.append((k, sp))
    return out


def _straddles_any_span(rec, p0, p1):
    """True if [p0,p1) partially overlaps a span boundary (crosses start or end
    without being fully contained). Such edits sit on a malformed source span
    and are unsafe to auto-apply."""
    for (k, sp) in _spans_covering(rec, p0, p1):
        s, e = sp["start"], sp["end"]
        contained = s <= p0 and p1 <= e          # edit fully inside this span
        contains = p0 <= s and e <= p1           # edit fully wraps this span
        if not (contained or contains):
            return True
    return False


# ---------------------------------------------------------------------------
# 4) Detector
# ---------------------------------------------------------------------------
def detect(rec: dict):
    """Return a list of candidate-issue dicts for one record.

    label_value candidate keys:
      kind='label_value', subtype, label_field, label_pos, label_len,
      label_surface, value, value_fields[list], vpos, vlen, confidence
    span_format candidate keys:
      kind='span_format', span_index, span_field, value, value_fields, confidence
    """
    text = rec.get("text", "")
    low = nfc(text).lower()           # same length as text (NFC 1:1 for this data)
    cands = []
    seen_pos = set()

    # ---- A) label -> value mismatch in the narrative text ----
    for phrase, lab_field in LABEL_LEXICON:
        start = 0
        plen = len(phrase)
        while True:
            i = low.find(phrase, start)
            if i == -1:
                break
            start = i + plen
            if i > 0 and _WORD_BOUNDARY.match(low[i - 1]):
                continue
            j = i + plen
            cm = re.match(r"[\s:\"'“”‘’\-–—]*", text[j:])
            vpos = j + cm.end()
            la = re.match(r"là[\s:]+", text[vpos:], re.IGNORECASE)
            if la:
                vpos += la.end()
            if vpos in seen_pos:
                continue

            # A1) closed-vocab value sitting right after the label
            matched_val = matched_field = None
            for val, vfield in _VOCAB_SORTED:
                if low.startswith(val, vpos) and _is_boundary(low, vpos + len(val)):
                    matched_val, matched_field = val, vfield
                    break
            if (matched_val and matched_field != lab_field
                    and matched_val not in AMBIGUOUS_TOKENS):
                seen_pos.add(vpos)
                cands.append({
                    "kind": "label_value", "subtype": "closed",
                    "label_field": lab_field, "label_pos": i, "label_len": plen,
                    "label_surface": text[i:i + plen],
                    "value": text[vpos:vpos + len(matched_val)],
                    "value_fields": [matched_field],
                    "vpos": vpos, "vlen": len(matched_val),
                    "confidence": "high",
                })
                continue

            # A2) structured value (regex) of a DIFFERENT field after the label
            if lab_field in STRUCTURED_LABEL_FIELDS:
                # token regex excludes quotes so we never capture a trailing "
                tok_m = re.match(r"[^\s,;.)\]\u2026\"'“”‘’]+", text[vpos:])
                if tok_m:
                    tok = tok_m.group()
                    vf = classify_value(tok)
                    if lab_field == "tax_code":
                        if vf and not TAX_CODE_RE.match(tok.strip("\"'.,;:")):
                            seen_pos.add(vpos)
                            cands.append({
                                "kind": "label_value", "subtype": "regex",
                                "label_field": lab_field, "label_pos": i,
                                "label_len": plen, "label_surface": text[i:i + plen],
                                "value": tok, "value_fields": sorted(vf),
                                "vpos": vpos, "vlen": len(tok), "confidence": "high",
                            })
                            continue
                    elif vf and lab_field not in vf:
                        seen_pos.add(vpos)
                        cands.append({
                            "kind": "label_value", "subtype": "regex",
                            "label_field": lab_field, "label_pos": i,
                            "label_len": plen, "label_surface": text[i:i + plen],
                            "value": tok, "value_fields": sorted(vf),
                            "vpos": vpos, "vlen": len(tok),
                            "confidence": "high" if len(vf) == 1 else "medium",
                        })
                        continue

    # ---- B) span whose value violates its own field's format ----
    #   tight-format field  -> flag any malformed value (cross-category = high)
    #   closed-vocab field  -> flag ONLY genuine cross-category (value is a
    #                          DIFFERENT closed field); typos are ignored
    #   phrase field        -> never checked
    for k, sp in enumerate(rec.get("spans", [])):
        f = sp.get("field_name")
        v = sp.get("text", "")
        vf = classify_value(v)
        if f in TIGHT_FORMAT_FIELDS:
            if not value_ok_for_field(v, f):
                cross = sorted(x for x in vf if x != f)
                cands.append({
                    "kind": "span_format", "span_index": k, "span_field": f,
                    "value": v, "value_fields": cross,
                    "confidence": "high" if cross else "low",
                })
        elif f in CLOSED:
            cross = sorted(x for x in vf if x in CLOSED and x != f)
            if cross and norm(v) not in CLOSED[f]:
                cands.append({
                    "kind": "span_format", "span_index": k, "span_field": f,
                    "value": v, "value_fields": cross, "confidence": "high",
                })
        # phrase fields: skip

    # ---- C) VALUE-DRIVEN: an identifiable, correctly-spanned value sitting
    #         under a WRONG or UNTRACKED prose label. Trust the span; fix label.
    for k, sp in enumerate(rec.get("spans", [])):
        f = sp.get("field_name")
        if f not in VALUE_DRIVEN_FORMAT_FIELDS:
            continue
        s0, e0 = sp.get("start"), sp.get("end")
        if s0 is None or e0 is None or e0 <= s0 or (e0 - s0) > 45:
            continue
        if s0 in seen_pos:                       # already handled by section A
            continue
        v = sp.get("text", "") or text[s0:e0]
        if text[s0:e0] != v:                     # span offsets must be intact
            continue
        # the value must really match the field (extra safety beyond the span)
        if not value_ok_for_field(v, f) and f not in classify_value(v):
            continue

        # don't let the governing-label search reach back past a preceding value
        floor = 0
        for sp2 in rec.get("spans", []):
            e2 = sp2.get("end")
            if e2 is not None and e2 <= s0 and e2 > floor:
                floor = e2
        region, rs, re_, had_conn = _governing_label(text, low, s0, floor=floor)
        if not region or not (2 <= len(region) <= 45):
            continue
        rl = region.lower()
        canon = CANONICAL_LABELS.get(f, "")
        if canon and canon.lower() in rl:        # label already correct (verbose)
            continue

        # is there a known field alias inside the governing label?
        f_label = a_pos = a_len = a_surface = matched = None
        for phrase, lab in LABEL_LEXICON:
            jj = rl.find(phrase)
            if jj != -1 and _is_boundary(rl, jj + len(phrase)):
                f_label, matched = lab, phrase
                a_pos, a_len = rs + jj, len(phrase)
                a_surface = text[a_pos:a_pos + a_len]
                break

        if f_label is not None:
            if f_label == f:                     # label already names this field
                continue
            if f == "eid_credentials" and matched in CCCD_EID_SHARED_ALIASES:
                continue                          # "định danh" is shared w/ VNeID
            seen_pos.add(s0)
            cands.append({
                "kind": "label_value", "subtype": "value_anchored_alias",
                "label_field": f_label, "label_pos": a_pos, "label_len": a_len,
                "label_surface": a_surface, "value": v, "value_fields": [f],
                "vpos": s0, "vlen": e0 - s0, "confidence": "high",
            })
        else:
            # UNTRACKED decoy label. Conservative gates to avoid relabelling a
            # legitimate-but-unlisted label or a mid-sentence fragment:
            #   - a real connector ('là'/'số'/':') must separate label & value
            #   - the phrase must look like a label: <=6 words, <=32 chars, no digit
            #   - it must START with a noun-label token and contain no verb words
            #   - it must not sit over / straddle any span
            if not had_conn:
                continue
            if len(region) > 32 or len(region.split()) > 6 or any(c.isdigit() for c in region):
                continue
            toks = rl.split()
            if not toks or toks[0] not in _LABEL_START_TOKENS:
                continue
            if any(b in rl for b in _LABEL_BLOCK_WORDS):
                continue
            if f == "eid_credentials" and "vneid" in rl:   # already an e-ID label
                continue
            if _spans_covering(rec, rs, re_) or _straddles_any_span(rec, rs, re_):
                continue
            seen_pos.add(s0)
            cands.append({
                "kind": "label_value", "subtype": "value_anchored_decoy",
                "label_field": None, "label_pos": rs, "label_len": re_ - rs,
                "label_surface": region, "value": v, "value_fields": [f],
                "vpos": s0, "vlen": e0 - s0, "confidence": "high",
            })
    return cands


# ---------------------------------------------------------------------------
# 5) Triage -> relabel edits + value-fix edits + escalate
# ---------------------------------------------------------------------------
def plan_fixes(rec: dict, cands: list, strategy: str = "auto"):
    """Return (edits, escalate_items).

    edits: list of splice dicts {pos, length, old, new, op, field, reason}
           op in {'relabel','value_fix'}.  Applied left->right by apply_edits.
    escalate_items: candidate dicts that need the LLM.

    strategy:
      'auto'      (default) relabel coincidence cases + value-fix free cases
      'relabel'   only relabel coincidence cases
      'value_fix' only value-fix free cases
    """
    pf = rec.get("profile_fake", {}) or {}
    edits, escalate = [], []

    for c in cands:
        if c["kind"] != "label_value":
            escalate.append(c)
            continue

        lab = c["label_field"]
        vpos, vlen = c["vpos"], c["vlen"]
        p0, p1 = vpos, vpos + vlen
        value_fields = c.get("value_fields", [])
        covering = _spans_covering(rec, p0, p1)

        # ---- Case 1: value EXACTLY covered by a span whose field is the value's
        #              true type -> the span is right, only the label is wrong.
        #              RELABEL the prose. (handles cccd/passport/plate swaps too)
        exact_correct = None
        for (k, sp) in covering:
            if sp.get("start") == p0 and sp.get("end") == p1:
                sf = sp.get("field_name")
                if sf in value_fields and sf != lab:
                    exact_correct = sf
                    break
        if exact_correct and strategy in ("auto", "relabel"):
            new_label = CANONICAL_LABELS.get(exact_correct)
            lp0, lp1 = c["label_pos"], c["label_pos"] + c["label_len"]
            if new_label and norm(new_label) != norm(c["label_surface"]) \
                    and not _straddles_any_span(rec, lp0, lp1):
                edits.append({
                    "pos": c["label_pos"], "length": c["label_len"],
                    "old": c["label_surface"], "new": new_label,
                    "op": "relabel", "field": exact_correct,
                    "reason": f"prose_label_{lab}_but_span_is_{exact_correct}",
                })
                continue
            # label already canonical-equivalent; nothing to do
            escalate.append(c)
            continue

        # ---- Case 2: stray value with NO span covering it -> VALUE-FIX with the
        #              ground-truth profile value (safe: touches no annotation).
        if not covering and strategy in ("auto", "value_fix"):
            correct = pf.get(lab)
            single = len(value_fields) == 1
            clean = (c["value"] == c["value"].strip()
                     and not c["value"].endswith(('"', '”', '’', "'")))
            not_risky = norm(c["value"]) not in RISKY_FREE_TOKENS and len(c["value"].strip()) >= 4
            if (correct and isinstance(correct, str) and correct.strip()
                    and lab in pf and single and clean and not_risky
                    and value_ok_for_field(correct, lab)
                    and norm(correct) != norm(c["value"])
                    and c["confidence"] == "high"):
                edits.append({
                    "pos": vpos, "length": vlen,
                    "old": c["value"], "new": correct,
                    "op": "value_fix", "field": lab,
                    "reason": f"{lab}_slot_had_{value_fields[0]}_value_no_span",
                })
                continue

        # ---- everything else -> LLM ----
        escalate.append(c)

    # de-dup edits on identical (pos) keeping first; sort left->right
    edits.sort(key=lambda e: e["pos"])
    deduped, used = [], set()
    for e in edits:
        if e["pos"] in used:
            continue
        used.add(e["pos"])
        deduped.append(e)
    return deduped, escalate


# ---------------------------------------------------------------------------
# 6) Apply text edits and remap span offsets safely
# ---------------------------------------------------------------------------
def apply_edits(text: str, edits: list):
    """Apply splice edits (each {pos,length,old,new}) left->right.
    Skips any edit whose `old` no longer matches text[pos:pos+length]
    (offset-sanity guard). Returns (new_text, deltas, applied) where
    deltas is a sorted list of (orig_pos, orig_end, cumulative_shift)."""
    edits = sorted(edits, key=lambda e: e["pos"])
    out, cur, shift, deltas, applied = [], 0, 0, [], []
    for e in edits:
        p, ln, new = e["pos"], e["length"], e["new"]
        if p < cur:
            continue                                   # overlapping -> skip
        if "old" in e and text[p:p + ln] != e["old"]:  # sanity guard
            continue
        out.append(text[cur:p])
        out.append(new)
        cur = p + ln
        shift += len(new) - ln
        deltas.append((p, p + ln, shift))
        applied.append(e)
    out.append(text[cur:])
    return "".join(out), deltas, applied


def _remap_pos(pos: int, deltas: list) -> int:
    shift = 0
    for (p, end, s) in deltas:
        if end <= pos:
            shift = s
        else:
            break
    return pos + shift


def remap_spans(spans: list, new_text: str, deltas: list, applied_edits: list):
    """Update each span's start/end/text after edits.

    Handles: edit before span (shift), edit after span (no-op), edit strictly
    INSIDE a span (shift end + splice the span's text), edit exactly equal to a
    span (adopt new text). Partial boundary overlap -> needs_review.
    Returns (new_spans, n_needs_review)."""
    new_spans, needs_review = [], 0
    for sp in spans:
        sp = dict(sp)
        os, oe = sp.get("start"), sp.get("end")
        if os is None or oe is None:
            new_spans.append(sp)
            continue
        otext = sp.get("text", "")
        ns = _remap_pos(os, deltas)

        # find an applied edit that touches this span
        touching = [e for e in applied_edits
                    if not (e["pos"] + e["length"] <= os or e["pos"] >= oe)]
        if not touching:
            ne = _remap_pos(oe, deltas)
            sp["start"], sp["end"] = ns, ne
            if new_text[ns:ne] != otext:                # safety relocate
                ns, ne, ok = _relocate(new_text, ns, otext)
                sp["start"], sp["end"] = ns, ne
                if not ok:
                    needs_review += 1
                    sp["_needs_review"] = True
            new_spans.append(sp)
            continue

        # rebuild span text by applying each touching edit in local coords
        new_otext = otext
        ok = True
        for e in sorted(touching, key=lambda x: x["pos"]):
            ep, el, enew = e["pos"], e["length"], e["new"]
            if os <= ep and ep + el <= oe:              # strictly inside / exact
                rel = ep - os
                if otext[rel:rel + el] != e.get("old", otext[rel:rel + el]):
                    ok = False
                    break
                # recompute against current new_otext using offset bookkeeping
            else:
                ok = False                              # straddles boundary
                break
        if ok:
            # splice all inside-edits (compute on original otext, then shift)
            inside = sorted(touching, key=lambda x: x["pos"])
            buf, cur = [], 0
            for e in inside:
                rel = e["pos"] - os
                buf.append(otext[cur:rel])
                buf.append(e["new"])
                cur = rel + e["length"]
            buf.append(otext[cur:])
            new_otext = "".join(buf)
            ne = ns + len(new_otext)
            sp["text"], sp["start"], sp["end"] = new_otext, ns, ne
            if new_text[ns:ne] != new_otext:
                ns2, ne2, ok2 = _relocate(new_text, ns, new_otext)
                sp["start"], sp["end"] = ns2, ne2
                if not ok2:
                    needs_review += 1
                    sp["_needs_review"] = True
        else:
            needs_review += 1
            sp["_needs_review"] = True
            ne = _remap_pos(oe, deltas)
            sp["start"], sp["end"] = ns, ne
        new_spans.append(sp)
    return new_spans, needs_review


def _relocate(new_text: str, approx: int, target: str, window: int = 300):
    """Find `target` near `approx` in new_text. Returns (start,end,ok)."""
    lo = max(0, approx - window)
    w = new_text[lo: approx + window + len(target)]
    k = w.find(target)
    if k != -1:
        s = lo + k
        return s, s + len(target), True
    k = new_text.find(target)
    if k != -1:
        return k, k + len(target), True
    return approx, approx + len(target), False


def fix_record(rec: dict, strategy: str = "auto"):
    """Full deterministic pass on one record.
    Returns (fixed_record, report)."""
    cands = detect(rec)
    edits, escalate = plan_fixes(rec, cands, strategy=strategy)
    fixed = dict(rec)
    nrev = 0
    applied = []
    if edits:
        new_text, deltas, applied = apply_edits(rec["text"], edits)
        new_spans, nrev = remap_spans(rec.get("spans", []), new_text, deltas, applied)
        fixed["text"] = new_text
        fixed["spans"] = new_spans
        # edits that failed the sanity guard -> escalate them
        applied_pos = {(e["pos"], e["length"]) for e in applied}
        for e in edits:
            if (e["pos"], e["length"]) not in applied_pos:
                escalate.append({
                    "kind": "label_value", "subtype": "skipped_edit",
                    "label_field": e.get("field"), "value": e.get("old"),
                    "value_fields": [], "confidence": "low",
                    "note": "deterministic edit skipped by sanity guard",
                })
    residual = detect(fixed)
    n_relabel = sum(1 for e in applied if e["op"] == "relabel")
    n_value = sum(1 for e in applied if e["op"] == "value_fix")
    report = {
        "record_id": rec.get("record_id"),
        "n_candidates": len(cands),
        "n_edits": len(applied),
        "n_relabel": n_relabel,
        "n_value_fix": n_value,
        "n_escalate": len(escalate),
        "edits": applied,
        "escalate": escalate,
        "needs_review": nrev,
        "residual_after": len(residual),
    }
    return fixed, report

import json

# Self-test trên shard mẫu — minh hoạ "phát hiện then chốt" bằng số liệu
try:
    sample = json.load(open(SAMPLE_FILE, encoding="utf-8"))
    agg = dict(flag=0, cand=0, relabel=0, value_fix=0, escalate=0, broken=0, residual=0)
    for rec in sample:
        _, rep = fix_record(rec, strategy="auto")
        agg["flag"]      += 1 if rep["n_candidates"] else 0
        agg["cand"]      += rep["n_candidates"]
        agg["relabel"]   += rep["n_relabel"]
        agg["value_fix"] += rep["n_value_fix"]
        agg["escalate"]  += rep["n_escalate"]
        agg["broken"]    += 1 if rep["needs_review"] else 0
        agg["residual"]  += rep["residual_after"]
    print(f"{len(sample)} hồ sơ | gắn cờ {agg['flag']} | ứng viên lỗi {agg['cand']}")
    print(f"  -> relabel {agg['relabel']} | value-fix {agg['value_fix']} | escalate {agg['escalate']}")
    print(f"  -> span bị vỡ {agg['broken']} | lỗi còn lại sau sửa (= số escalate) {agg['residual']}")
    auto = agg['relabel'] + agg['value_fix']
    print("  (~{:.0f}% lỗi do luật tự xử lý; LLM chỉ chạm phần còn lại)".format(
        100 * auto / max(1, auto + agg['escalate'])))
except FileNotFoundError:
    print(f"(Bỏ qua self-test: không thấy {SAMPLE_FILE} — đặt shard mẫu cạnh notebook để xem số liệu)")
# --- Ước tính chi phí bằng số đo THỰC (tỉ lệ escalate trên shard mẫu) ---
N_TOTAL = 15000
PRICE = {"gemini-2.5-flash-lite": (0.10, 0.40), "gemini-2.5-flash": (0.30, 2.50)}
in_p, out_p = PRICE.get(GEMINI_MODEL, (0.10, 0.40))

try:
    esc_recs = sum(1 for r in sample if plan_fixes(r, detect(r), "auto")[1])
    esc_rate = esc_recs / len(sample)
except Exception:
    esc_rate = 0.03

naive_in = N_TOTAL * 9000
naive_cost = naive_in / 1e6 * in_p
calls = (N_TOTAL * esc_rate) / max(1, BATCH_SIZE)
fun_in, fun_out = calls * 1500, calls * 300
fun_cost = fun_in / 1e6 * in_p + fun_out / 1e6 * out_p

print(f"Tỉ lệ escalate đo được   : {esc_rate*100:.1f}%  (~{int(N_TOTAL*esc_rate)} / {N_TOTAL} hồ sơ)")
print(f"Gửi thẳng full record    : ~{naive_in/1e6:.0f}M token vào -> ~${naive_cost:,.2f} (chỉ input)")
print(f"Phễu lọc (notebook này)  : ~{fun_in/1e3:.0f}K in + {fun_out/1e3:.0f}K out -> ~${fun_cost:,.3f}")
print(f"=> tiết kiệm ~{(1 - fun_cost/max(naive_cost,1e-9))*100:.1f}%")
from typing import List, Literal
try:
    from pydantic import BaseModel
    _HAS_PYDANTIC = True
except Exception:
    BaseModel = object
    _HAS_PYDANTIC = False  # cài pydantic trước khi chạy tầng D

# ---- Lược đồ kết quả structured output ----
class LLMEdit(BaseModel):
    issue_id: int
    action: Literal["relabel", "value_fix", "none"]
    find: str
    replace: str
    field: str
    reason: str

class RecordEdits(BaseModel):
    id: str
    edits: List[LLMEdit]

class BatchResponse(BaseModel):
    records: List[RecordEdits]

WIN = 45  # ký tự đệm mỗi bên của cửa sổ nghi vấn

def _window(text, a, b, pad=WIN):
    lo, hi = max(0, a - pad), min(len(text), b + pad)
    return ("…" if lo > 0 else "") + text[lo:hi] + ("…" if hi < len(text) else "")

def build_issue_payload(rec):
    """
    Đóng gói dữ liệu tối giản siêu tiết kiệm token cho LLM.
    Chỉ gửi nguyên văn 'text' và một từ điển ánh xạ {Giá trị : Nhãn đúng}.
    """
    text = rec.get("text", "")
    spans = rec.get("spans", [])

    # Tạo từ điển map nhanh để LLM dễ đối chiếu, tự động loại bỏ các span trùng lặp
    # Kết quả sẽ có dạng: {"36A-91523": "vehicle_plate", "Vũ Bá Thịnh": "full_name"}
    truth_map = {}
    for s in spans:
        val = s.get("text")
        field = s.get("field_name")
        if val and field:
            truth_map[val] = field

    return {
        "id": rec.get("record_id"),
        "text": text,
        "truth_map": truth_map
    }
import asyncio, json, random

CANON_HINT = "\n".join(f"- {k}: {v}" for k, v in CANONICAL_LABELS.items())

SYSTEM_INSTRUCTION = f"""Bạn là chuyên gia tinh chỉnh và làm sạch dữ liệu PII/SPII hành chính.
Input: "text" (văn bản) và "truth_map" (từ điển {{"Giá trị": "Nhãn đúng bản chất"}}).

NHIỆM VỤ: Quét TỪ ĐẦU ĐẾN CUỐI văn bản cho MỖI GIÁ TRỊ trong truth_map. Một giá trị có thể xuất hiện NHIỀU LẦN và bị gắn nhãn sai bằng nhiều cụm từ/ngữ cảnh khác nhau. BẠN PHẢI TÌM VÀ TẠO LỆNH SỬA CHO TẤT CẢ CÁC LẦN XUẤT HIỆN ĐÓ.

ĐỊNH DẠNG ĐẦU RA BẮT BUỘC (QUAN TRỌNG NHẤT):
- Trả về DUY NHẤT một MẢNG JSON (JSON Array) chứa các object. Ví dụ: [{{...}}, {{...}}].
- TUYỆT ĐỐI KHÔNG bọc kết quả trong markdown (KHÔNG dùng ký hiệu ```json hay ```).
- Nếu văn bản đã hoàn toàn đúng hoặc không có lỗi, trả về mảng rỗng: []

QUY TẮC TẠO 'find' VÀ 'replace' (CỰC KỲ NGHIÊM NGẶT):
- Cả 'find' và 'replace' BẮT BUỘC phải chứa đoạn văn bản bao trùm cả Nhãn + Giá Trị.
- Không được phép trả về 'find' chỉ chứa mỗi giá trị.
- Tuyệt đối giữ nguyên vẹn Giá Trị (Value) trong chuỗi 'replace'.

CÁC VÍ DỤ CHUẨN (HÃY HỌC THEO ĐÚNG CẤU TRÚC MẢNG NÀY):

Ví dụ 1 (Lỗi đơn lẻ):
- text: "...người đó có tên vợ là 36A-91523, sinh sống tại..."
- truth_map: {{"36A-91523": "vehicle_plate"}}
=> Output ĐÚNG:
[
    {{"action": "relabel", "find": "tên vợ là 36A-91523", "replace": "biển số xe là 36A-91523", "field": "vehicle_plate"}}
]

Ví dụ 2 (Giá trị xuất hiện NHIỀU LẦN - Phải sinh NHIỀU LỆNH SỬA):
- text: "...mã hồ sơ là P0136744. Sau đó cung cấp mã hồ sơ nghiệp vụ là P0136744..."
- truth_map: {{"P0136744": "passport_number"}}
=> Output ĐÚNG:
[
    {{"action": "relabel", "find": "mã hồ sơ là P0136744", "replace": "số hộ chiếu là P0136744", "field": "passport_number"}},
    {{"action": "relabel", "find": "mã hồ sơ nghiệp vụ là P0136744", "replace": "số hộ chiếu là P0136744", "field": "passport_number"}}
]

Ví dụ 3 (Hai giá trị nằm sát nhau):
- text: "...tài khoản ngân hàng 046556455641 và 046556455640..."
- truth_map: {{"046556455641": "cccd", "046556455640": "cccd"}}
=> Output ĐÚNG:
[
    {{"action": "relabel", "find": "tài khoản ngân hàng 046556455641", "replace": "số căn cước công dân 046556455641", "field": "cccd"}},
    {{"action": "relabel", "find": "và 046556455640", "replace": "và số căn cước công dân 046556455640", "field": "cccd"}}
]

Ví dụ 4 (XUNG ĐỘT NGỮ CẢNH - Bẻ lại văn bản xung quanh để khớp với bản chất giá trị):
- text: "...thực hiện phương thức thanh toán thông qua việc sử dụng 22A-45375 cho giao dịch..."
- truth_map: {{"22A-45375": "vehicle_plate"}}
=> Output ĐÚNG:
[
    {{"action": "relabel", "find": "phương thức thanh toán thông qua việc sử dụng 22A-45375", "replace": "phương tiện di chuyển là xe ô tô biển số 22A-45375", "field": "vehicle_plate"}}
]

VÍ DỤ SAI (TUYỆT ĐỐI KHÔNG LÀM):
- SAI: find: "020583090106" -> (Lỗi: Chỉ bắt mỗi giá trị).
- SAI: find: "tên vợ là " -> (Lỗi: Thiếu giá trị ở đuôi).
- SAI: {{ "action": "relabel"... }} -> (Lỗi: Trả về object đơn lẻ thay vì mảng [{{...}}]).

CẢNH BÁO TỐI THƯỢNG: Việc bạn tìm thấy 1 lỗi rồi lười biếng bỏ qua các lần xuất hiện tiếp theo của giá trị đó sẽ bị đánh giá là LỖI NGHIÊM TRỌNG. Bắt buộc phải quét cạn kiệt văn bản.

Nhãn chuẩn của hệ thống:
{CANON_HINT}
"""

def _genai_client():
    from openai import AsyncOpenAI
    return AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=API_BASE_URL)

def _make_cache(client):
    # Cổng OpenAI-compatible (vilao) không có "context cache" tường minh như Gemini -> bỏ qua.
    # System prompt vẫn được gửi kèm mỗi lời gọi (nhiều cổng tự cache prefix giống nhau).
    return None

def _load_checkpoint():
    done = {}
    if os.path.exists(CHECKPOINT_FILE):
        for line in open(CHECKPOINT_FILE, encoding="utf-8"):
            line = line.strip()
            if line:
                o = json.loads(line); done[o["id"]] = o["edits"]
    return done

def _append_checkpoint(rows):
    with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], "edits": r["edits"]}, ensure_ascii=False) + "\n")

def _parse_llm_rows(content, batch):
    """Chuẩn hoá đầu ra của cổng về [{id, edits}], chấp nhận mọi hình dạng JSON:
    {"records":[{id,edits}]}, một object {id,edits}, hoặc mảng edit thuần (gán cho hồ sơ trong lô)."""
    s = (content or "").strip()
    if s.startswith("```"):                       # gỡ rào markdown nếu lỡ có
        s = s.strip("`")
        nl = s.find("\n")
        s = (s[nl + 1:] if nl != -1 else s).strip().rstrip("`").strip()
    try:
        data = json.loads(s)
    except Exception:
        import re as _re
        m = _re.search(r"(\{.*\}|\[.*\])", s, _re.DOTALL)
        if not m:
            return []
        data = json.loads(m.group(1))
    if isinstance(data, dict) and "records" in data:
        return [{"id": r.get("id"), "edits": r.get("edits", [])} for r in data["records"]]
    if isinstance(data, dict) and "edits" in data:
        rid = data.get("id") or (batch[0].get("record_id") if batch else None)
        return [{"id": rid, "edits": data.get("edits", [])}]
    if isinstance(data, list) and batch:          # mảng edit thuần -> hồ sơ đầu của lô (đặt BATCH_SIZE=1)
        return [{"id": batch[0].get("record_id"), "edits": data}]
    return []

async def _process_batch(client, cache_name, batch, sem):
    payloads = [p for p in (build_issue_payload(r) for r in batch) if p]
    if not payloads:
        return []
    user_msg = ("Rà soát các hồ sơ sau (JSON đầu vào). Trả về JSON đúng lược đồ, không kèm văn bản thừa:\n"
                + json.dumps(payloads, ensure_ascii=False))
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": user_msg}]
    use_json_mode = True   # tự tắt nếu cổng không hỗ trợ response_format
    async with sem:
        for attempt in range(4):
            try:
                kwargs = dict(model=GEMINI_MODEL, messages=messages, temperature=0)
                if use_json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = await client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content
                rows = _parse_llm_rows(content, batch)
                _append_checkpoint(rows)
                return rows
            except Exception as e:
                msg = str(e)
                if use_json_mode and "response_format" in msg:   # cổng không nhận -> bỏ JSON mode, thử lại
                    use_json_mode = False
                    print("  (cổng không hỗ trợ response_format -> dùng prompt JSON thuần)")
                    continue
                wait = 2 ** attempt + random.random()
                print(f"  ! lỗi lần {attempt+1}: {msg[:90]} -> chờ {wait:.1f}s")
                await asyncio.sleep(wait)
        return []

async def run_gemini(records):
    """Trả về dict: record_id -> list[edit do LLM đề xuất]."""
    if not GEMINI_API_KEY:
        print("Không có GEMINI_API_KEY -> bỏ qua tầng LLM.")
        return {}
    to_send = records
    done = _load_checkpoint()
    pending = [r for r in to_send if r.get("record_id") not in done]
    print(f"LLM: {len(to_send)} hồ sơ cần xét | {len(done)} đã có checkpoint | {len(pending)} còn lại")
    if not pending:
        return done
    client = _genai_client()
    cache_name = _make_cache(client)
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    batches = [pending[i:i+BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)]
    results = await asyncio.gather(*[_process_batch(client, cache_name, b, sem) for b in batches])
    for rows in results:
        for row in rows:
            done[row["id"]] = row["edits"]
    print(f"  hoàn tất {len(batches)} lô.")
    return done
def _locate(text, find, approx=None, win=400):
    if not find:
        return -1
    if approx is not None:
        k = text.find(find, max(0, approx - win), approx + win + len(find))
        if k != -1:
            return k
    return text.find(find)

def _llm_to_edits(rec, llm_edits):
    text = rec.get("text", "")
    spans = rec.get("spans", []) or []

    def _touches_span(p0, p1):
        return [sp for sp in spans
                if sp.get("start") is not None and sp.get("end") is not None
                and not (sp["end"] <= p0 or sp["start"] >= p1)]

    out = []
    for e in (llm_edits or []):
        op = e.get("action") or e.get("op")
        if op not in ("relabel", "value_fix"):
            continue

        find_full = e.get("find", "").strip()
        replace_full = e.get("replace", "").strip()

        if not find_full or replace_full == find_full:
            continue

        # ==============================================================
        # 1. BẮT BUỘC: Tìm vị trí của TOÀN BỘ chuỗi trước để chốt tọa độ
        # ==============================================================
        pos = text.find(find_full)
        if pos < 0:
            continue

        # ==============================================================
        # 2. Toán học bóc tách: Tính số ký tự dư thừa ở hai đầu
        # ==============================================================
        pref = 0
        while pref < len(find_full) and pref < len(replace_full) and find_full[pref] == replace_full[pref]:
            pref += 1

        suff = 0
        while suff < len(find_full) - pref and suff < len(replace_full) - pref and find_full[-(suff+1)] == replace_full[-(suff+1)]:
            suff += 1

        # Lấy phần lõi thực sự thay đổi
        core_find = find_full[pref : len(find_full)-suff] if suff > 0 else find_full[pref:]
        core_replace = replace_full[pref : len(replace_full)-suff] if suff > 0 else replace_full[pref:]

        if not core_find.strip():
            continue

        # Tính lại tọa độ P0, P1 siêu chuẩn xác
        p0 = pos + pref
        p1 = pos + len(find_full) - suff

        hit = _touches_span(p0, p1)

        # -------------------------------------------------------------
        # 3. Lớp khiên bảo vệ (Chống nuốt dữ liệu)
        # -------------------------------------------------------------
        if op == "relabel":
            if len(core_find) > 120:
                continue

            is_safe = True
            for sp in hit:
                # Tránh lỗi chia cho 0 hoặc rỗng
                if not core_replace and not core_find:
                    continue
                # Nếu đè lên span, bắt buộc span phải còn tồn tại sau khi sửa
                if sp.get("text") not in core_replace and sp.get("text") not in text[p0:p1].replace(core_find, core_replace):
                    is_safe = False
                    break

            if not is_safe:
                print(f"⚠️ Chặn LLM làm mất PII tại: '{find_full}'")
                continue

        else:
            if not any(sp["start"] == p0 and sp["end"] == p1 for sp in hit):
                continue

        out.append({
            "pos": p0,
            "length": p1 - p0, # Dùng độ dài tính bằng toán học
            "old": core_find,
            "new": core_replace,
            "op": op,
            "field": e.get("field"),
            "reason": "LLM_Override: " + (e.get("reason") or ""),
            "source": "LLM"
        })

    return out

def fix_record_full(rec, llm_edits=None):
    """Áp dụng 100% bằng LLM, vứt bỏ det_edits của tầng Rule-base"""
    text = rec.get("text", "")

    # Chỉ lấy các edit do LLM sinh ra
    all_edits = sorted(_llm_to_edits(rec, llm_edits), key=lambda e: e["pos"])

    fixed, applied, nrev = dict(rec), [], 0
    if all_edits:
        new_text, deltas, applied = apply_edits(text, all_edits)
        new_spans, nrev = remap_spans(rec.get("spans", []), new_text, deltas, applied)
        fixed["text"], fixed["spans"] = new_text, new_spans

    audit = {
        "record_id": rec.get("record_id"),
        "n_rule": 0,
        "n_llm":  len(applied),
        "edits": [{k: e.get(k) for k in ("pos", "op", "old", "new", "field", "reason", "source")}
                  for e in applied],
        "spans_needs_review": nrev,
    }
    return fixed, audit

def _overlap(a, b):
    return not (a["pos"] + a["length"] <= b["pos"] or a["pos"] >= b["pos"] + b["length"])

import glob, time

def process_shard(path, llm_map):
    recs = json.load(open(path, encoding="utf-8"))
    fixed_recs, audits = [], []
    for r in recs:
        fr, au = fix_record_full(r, llm_map.get(r.get("record_id")))
        fixed_recs.append(fr); audits.append(au)
    out_path = os.path.join(OUTPUT_DIR, os.path.basename(path).replace(".json", ".fixed.json"))
    json.dump(fixed_recs, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return out_path, audits

async def run_pipeline():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, INPUT_GLOB)))
    if not files:
        print(f"Không thấy shard nào khớp {INPUT_DIR}/{INPUT_GLOB}.")
        return
    print(f"Tìm thấy {len(files)} shard.")
    all_recs = []
    for f in files:
        with open(f, "r", encoding="utf-8") as file:
            data = json.load(file)
            if isinstance(data, list):
                all_recs.extend(data)
            elif isinstance(data, dict):
                all_recs.append(data)

    llm_map = await run_gemini(all_recs) if (USE_GEMINI or ESCALATE_ALL) else {}

    all_audit, t0 = [], time.time()
    for f in files:
        out_path, audits = process_shard(f, llm_map)
        all_audit.extend(audits)
        nr = sum(a["n_rule"] for a in audits); nl = sum(a["n_llm"] for a in audits)
        rs = sum(a.get("residual_after", 0) for a in audits); bk = sum(a.get("spans_needs_review", 0) for a in audits)
        print(f"  ✓ {os.path.basename(f)} -> {os.path.basename(out_path)} | luật {nr}, LLM {nl}, span vỡ {bk}, lỗi còn lại {rs}")

    json.dump(all_audit, open(os.path.join(OUTPUT_DIR, "audit_combined.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    sent = len(llm_map)
    print(f"\nXong trong {time.time()-t0:.1f}s. Audit: {os.path.join(OUTPUT_DIR, 'audit_combined.json')}")
    print(f"Hồ sơ đã gửi LLM: {sent} (~{sent/max(1,len(all_recs))*100:.1f}% tổng {len(all_recs)}).")

if __name__ == "__main__":
    asyncio.run(run_pipeline())

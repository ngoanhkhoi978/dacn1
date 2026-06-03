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

"""Serialise one scoring row of features.parquet into SLM text (docs/serialisation.md).

Pure Python, no GPU and no model. Three pieces:

- prompt(row): the session as fixed lines, in a fixed order with fixed wording;
- target(row, label): "Label: <yes|no>" then "Reason: <one sentence>", the training text;
- scoring_prompt(row): prompt(row) + "\nLabel:", what the scorer feeds the model.

The prompt reads only the 44 feature columns in configs/features.yaml. Labels never enter
the prompt or the reason: target() takes the label as an argument and uses it for the
"Label:" word only. The reason is templated from the features alone (the largest positive
deviations from the user's 30-day norm), so it is the same for a positive and a negative
row with the same features.
"""
import math

# Columns of features.parquet that are not model features (configs/features.yaml says so):
# identifiers, the scoring timestamp, and is_final (metadata only, never a feature).
EXCLUDED = {
    "row_id": "identifier",
    "session_id": "identifier",
    "user": "identifier; the model must not memorise users",
    "pc": "identifier",
    "score_time": "metadata; time-of-day enters only through off_hours and after_hours_emails",
    "is_final": "metadata only, never a feature (decisions log)",
}

# The z-score features: (column, name in the reason, short name in the prompt), in this order.
DEVIATIONS = [
    ("usb_connects_z", "USB connects", "USB"),
    ("file_events_z", "file copies", "files"),
    ("emails_sent_z", "emails sent", "emails"),
    ("external_recipients_z", "external recipients", "external"),
    ("http_job_search_z", "job-search visits", "job"),
    ("http_cloud_storage_z", "cloud-storage visits", "cloud"),
    ("http_filesharing_leak_z", "file-sharing or leak visits", "leak"),
]
REASON_Z = 2.0        # a deviation counts for the reason at z >= 2
REASON_TOP = 2        # the reason names at most this many deviations
SUFFIX = "\nLabel:"

HEADER = "Is this session malicious? [n] = last hour."


def z(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{float(value):.1f}".replace("-0.0", "0.0")


def kilobytes(value):
    return f"{int(round(int(value) / 1024))}"


def last_hour(row, name):
    return f"{int(row[name])} [{int(row[name + '_1h'])}]"


# Fixed order. Each line: (label, function of the row, the feature columns it reads).
LINES = [
    ("Session", lambda r: (f"hour {int(r['session_minutes_so_far']) // 60}, "
                           f"{'off-hours' if int(r['off_hours']) else 'working hours'}, "
                           f"{'weekend' if int(r['weekend']) else 'weekday'}, "
                           f"{'new PC' if int(r['new_pc']) else 'known PC'}"),
     ["session_minutes_so_far", "off_hours", "weekend", "new_pc"]),
    ("User", lambda r: f"{r['role']}; {r['department']}; {r['functional_unit']}",
     ["role", "department", "functional_unit"]),
    ("Logons", lambda r: (f"unlocks {int(r['unlocks_so_far'])}, "
                          f"today {int(r['logon_events_today_so_far'])}"),
     ["unlocks_so_far", "logon_events_today_so_far"]),
    ("USB", lambda r: (f"{last_hour(r, 'usb_connects')}, "
                       f"today {int(r['usb_connects_today_so_far'])}"),
     ["usb_connects", "usb_connects_1h", "usb_connects_today_so_far"]),
    ("Files", lambda r: (f"{last_hour(r, 'file_events')}; "
                         + " ".join(f"{e} {int(r[f'file_ext_{e}'])}"
                                    for e in ("doc", "pdf", "zip", "exe", "jpg", "txt"))
                         + f"; mismatch {int(r['file_header_mismatch'])}"),
     ["file_events", "file_events_1h", "file_ext_doc", "file_ext_pdf", "file_ext_zip",
      "file_ext_exe", "file_ext_jpg", "file_ext_txt", "file_header_mismatch"]),
    ("Email", lambda r: (f"sent {last_hour(r, 'emails_sent')}, after hours "
                         f"{int(r['after_hours_emails'])}; external "
                         f"{last_hour(r, 'external_recipients')}; attachments "
                         f"{last_hour(r, 'attachments_count')}; {kilobytes(r['email_bytes'])} KB"),
     ["emails_sent", "emails_sent_1h", "after_hours_emails", "external_recipients",
      "external_recipients_1h", "attachments_count", "attachments_count_1h", "email_bytes"]),
    ("Web", lambda r: (f"visits {int(r['http_visits'])}; job "
                       f"{last_hour(r, 'http_job_search')}; cloud "
                       f"{last_hour(r, 'http_cloud_storage')}; leak "
                       f"{last_hour(r, 'http_filesharing_leak')}; hacking "
                       f"{int(r['http_hacking_tools'])}"),
     ["http_visits", "http_job_search", "http_job_search_1h", "http_cloud_storage",
      "http_cloud_storage_1h", "http_filesharing_leak", "http_filesharing_leak_1h",
      "http_hacking_tools"]),
    ("30-day z", lambda r: " ".join(f"{short} {z(r[col])}" for col, _, short in DEVIATIONS),
     [col for col, _, _ in DEVIATIONS]),
]

USED = [col for _, _, cols in LINES for col in cols]


def prompt(row):
    """The session text. `row` is any mapping with the feature columns; other keys are ignored."""
    return "\n".join([HEADER] + [f"{label}: {text(row)}" for label, text, _ in LINES])


def scoring_prompt(row):
    return prompt(row) + SUFFIX


def reason(row):
    """One sentence from the largest positive deviations (z >= REASON_Z); no labels."""
    found = []
    for col, name, _ in DEVIATIONS:
        value = row[col]
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        if float(value) >= REASON_Z:
            found.append((-round(float(value), 1), name))
    found = sorted(found)[:REASON_TOP]        # ties broken by name: deterministic
    if not found:
        return f"no 30-day z is {REASON_Z:g} or more."
    parts = [f"{name} {-neg:+.1f}" for neg, name in found]
    return f"highest 30-day z: {', '.join(parts)}."


def target(row, label):
    """Training completion after the scoring prompt: " yes|no" then the reason line."""
    return f" {'yes' if label else 'no'}\nReason: {reason(row)}"


def training_text(row, label):
    """The full training example: scoring prompt followed by its completion."""
    return scoring_prompt(row) + target(row, label)


# ---------- Comparison 2: text block (runbook 6.2) ----------
#
# A block of the session's own email, http and file events is placed between the structured
# prompt and "\nLabel:", so scoring still reads the answer token after "Label:" and the
# target is unchanged. Only events with ts <= score_time are eligible (the feature rule).
# Truncation: events are taken in priority order (TEXT_TIERS, then newest first, then
# event_id), whole events only; an event that does not fit the remaining budget is skipped
# and the next one tried. The selected events are shown oldest first. The budget is in
# model tokens of the whole block, header included, measured by the caller's tokenizer.

TEXT_HEADER = "\nText:"
TEXT_NONE = " none"
TEXT_BUDGET = 256
# 0: http to a job-search, cloud-storage, leak or hacking host (the feature keyword lists);
# 1: email; 2: file; 3: any other http.
TEXT_TIERS = {"http_flagged": 0, "email": 1, "file": 2, "http": 3}


def event_tier(event):
    if event["source"] == "http" and event.get("flagged"):
        return TEXT_TIERS["http_flagged"]
    return TEXT_TIERS[event["source"]]


def event_line(event):
    """One event as one line: '<source> <HH:MM> <meta>: <content>'."""
    head = f"{event['source']} {event['ts']:%H:%M}"
    if event.get("meta"):
        head += f" {event['meta']}"
    content = " ".join(str(event.get("content") or "").split())
    return f"\n{head}: {content}" if content else f"\n{head}:"


def eligible(events, score_time):
    """Leakage rule: only events at or before the scoring time."""
    return [e for e in events if e["ts"] <= score_time]


def priority_order(events):
    """Tier, then newest first, then event_id: a total, deterministic order."""
    by_id = sorted(events, key=lambda e: str(e["event_id"]))
    by_time = sorted(by_id, key=lambda e: e["ts"], reverse=True)
    return sorted(by_time, key=event_tier)


def text_block(events, score_time, count_tokens, budget=TEXT_BUDGET, line_tokens=None):
    """The text block for one scoring row and a summary of the truncation.

    count_tokens(str) -> int is the model tokenizer (no special tokens). line_tokens, if
    given, maps event_id to count_tokens(event_line(e)), to avoid re-tokenizing.
    Returns (block, info): block starts with TEXT_HEADER and fits the budget.
    """
    pool = eligible(events, score_time)
    lines = {e["event_id"]: event_line(e) for e in pool}
    cost = {i: (line_tokens[i] if line_tokens and i in line_tokens else count_tokens(lines[i]))
            for i in lines}
    header = count_tokens(TEXT_HEADER)
    full = header + sum(cost.values())
    remaining = budget - header
    chosen = []
    for e in priority_order(pool):
        if cost[e["event_id"]] <= remaining:
            chosen.append(e)
            remaining -= cost[e["event_id"]]

    def assemble(selected):
        if not selected:
            return TEXT_HEADER + TEXT_NONE
        shown = sorted(selected, key=lambda e: (e["ts"], str(e["event_id"])))
        return TEXT_HEADER + "".join(lines[e["event_id"]] for e in shown)

    block = assemble(chosen)
    # Line costs are measured separately; drop the lowest-priority event if joining them
    # changes the count enough to break the budget (rare).
    while chosen and count_tokens(block) > budget:
        chosen.pop()
        block = assemble(chosen)
    info = {"eligible_events": len(pool), "selected_events": len(chosen),
            "tokens_untruncated": full, "tokens": count_tokens(block),
            "truncated": len(chosen) < len(pool)}
    return block, info


def text_scoring_prompt(row, block):
    """Structured prompt + text block, still ending in "\\nLabel:"."""
    return prompt(row) + block + SUFFIX

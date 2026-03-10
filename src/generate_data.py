import random
import csv
import datetime as dt
from typing import List, Tuple
import os

# 1) Canonical month names
MONTHS_LONG = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
MONTHS_SHORT = [m[:3] for m in MONTHS_LONG]

# 2) OCR confusions mapping for synthetic noise
OCR_CONFUSIONS = {
    "0": ["O", "o"],
    "1": ["l", "I"],
    "2": ["Z"],
    "3": ["B"],
    "5": ["S"],
    "6": ["G"],
    "8": ["B"],
    "O": ["0"],
    "o": ["0"],
    "l": ["1", "I"],
    "I": ["1", "l"],
    "S": ["5"],
    "B": ["8", "3"],
    "Z": ["2"],
}

SEPARATORS = [
    "/",
    "-",
    ".",
    " ",
    "'",
    "",
]  # include empty to simulate missing separators
WHITESPACE_VARIANTS = [" ", "  ", "\t", ""]


def rand_date(
    start_year=2000,
    end_year=2026,
) -> dt.date:
    """Generate a random valid date."""
    year = random.randint(start_year, end_year)
    month = random.randint(1, 12)
    # Days per month
    if month == 2:
        # simple leap-year rule
        is_leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        day = random.randint(1, 29 if is_leap else 28)
    elif month in [4, 6, 9, 11]:
        day = random.randint(1, 30)
    else:
        day = random.randint(1, 31)
    return dt.date(year, month, day)


def iso(d: dt.date) -> str:
    return d.strftime("%Y-%m-%d")


def with_prob(p: float) -> bool:
    return random.random() < p


def maybe_confuse_char(c: str, p=0.08) -> str:
    """Randomly replace character with an OCR confusion."""
    if with_prob(p) and c in OCR_CONFUSIONS:
        return random.choice(OCR_CONFUSIONS[c])
    return c


def reinforce_zero_O_confusion(s: str) -> str:
    # Increase chance of O<->0 swap specifically when near digits
    # e.g., "30" -> "3O"; "2021" -> "2O21" sometimes
    chars = list(s)
    for i, ch in enumerate(chars):
        if ch == "0" and with_prob(0.12):  # higher prob than general corrupt_string
            chars[i] = "O"
        elif ch == "O" and with_prob(0.12):
            chars[i] = "0"
    return "".join(chars)


def maybe_zero_like_day(day: int) -> str:
    # For days that end with 0, sometimes inject 'O'
    d = f"{day}"
    if d.endswith("0") and with_prob(0.4):
        return d[:-1] + "O"
    return d


def corrupt_string(s: str, char_p=0.08, drop_p=0.02, dup_p=0.02) -> str:
    """Apply character-level noise: replacements, drops, and occasional duplicates."""
    out = []
    for ch in s:
        # drop
        if with_prob(drop_p):
            continue
        # replace
        ch2 = maybe_confuse_char(ch, p=char_p)
        out.append(ch2)
        # duplicate
        if with_prob(dup_p):
            out.append(ch2)
    return "".join(out)


def maybe_pad_spaces(parts: List[str]) -> str:
    """Randomize spacing (including no-space) between parts to mimic OCR joining/splitting."""
    sep = random.choice(WHITESPACE_VARIANTS)
    return sep.join(parts)


def fmt_with_month_name(d: dt.date, short=False, format_type="day_first") -> str:
    month = MONTHS_SHORT[d.month - 1] if short else MONTHS_LONG[d.month - 1]
    day_str = maybe_zero_like_day(d.day)
    year2 = f"{d.year%100:02d}"
    year4 = f"{d.year}"

    if format_type == "day_first":
        y = year4 if with_prob(0.5) else year2
        order = random.choice([[day_str, month, y], [day_str.zfill(2), month, y]])
    elif format_type == "year_first":
        # Always full year for year-first
        order = [year4, month, day_str]
    else:
        raise ValueError("Unknown format_type")

    return maybe_pad_spaces(order)


def fmt_numeric(d: dt.date, format_type="day_first") -> str:
    day = f"{d.day}"
    month = f"{d.month}"
    dd = day if with_prob(0.5) else day.zfill(2)
    mm = month if with_prob(0.5) else month.zfill(2)
    yyyy = f"{d.year}"
    yy = f"{d.year%100:02d}"
    y = yyyy if with_prob(0.5) else yy

    s1 = random.choice(SEPARATORS)
    s2 = random.choice(SEPARATORS)

    if format_type == "day_first":
        parts = [dd, mm, y]
    elif format_type == "year_first":
        parts = [yyyy, mm, dd]  # Always full year for year-first
    else:
        raise ValueError("Unknown format_type")

    return f"{parts[0]}{s1}{parts[1]}{s2}{parts[2]}"


def maybe_add_noise_tokens(s: str) -> str:
    """
    Add surrounding noise like words and currency to mimic receipt context lines.
    E.g., 'DATE: 05Ju1'22', 'Txn 05-07-22', etc.
    """
    prefixes = ["DATE", "Dt", "Txn", "On", "Issued", "D:", ""]
    suffixes = ["", "  at 10:23", "  Total:", "  VAT:", "  Ref: 12345"]
    pre = random.choice(prefixes)
    suf = random.choice(suffixes)
    parts = []
    if pre:
        parts.append(pre + random.choice([":", " =", " -", " "]))

    parts.append(s)

    if suf:
        parts.append(suf)

    return maybe_pad_spaces(parts)


def inject_month_typos(s: str) -> str:
    repls = [
        ("Jan", "J an"),
        ("Feb", "Fe b"),
        ("Mar", "M ar"),
        ("Apr", "A pr"),
        ("May", "M ay"),
        ("Jun", "J un"),
        ("Jul", "Ju1"),
        ("Aug", "Au9"),
        ("Sep", "5ep"),
        ("Oct", "0ct"),
        ("Nov", "N ov"),
        ("Dec", "D ec"),
        ("jan", "j an"),
        ("feb", "fe b"),
        ("mar", "m ar"),
        ("apr", "a pr"),
        ("may", "m ay"),
        ("jun", "j un"),
        ("jul", "ju1"),
        ("aug", "au9"),
        ("sep", "5ep"),
        ("oct", "0ct"),
        ("nov", "n ov"),
        ("dec", "d ec"),
    ]
    if with_prob(0.45):  # slightly higher coverage
        for a, b in repls:
            if a in s and with_prob(0.5):
                s = s.replace(a, b)
    return s


def generate_example() -> Tuple[str, str]:
    d = rand_date()
    target = iso(d)

    # Decide format type
    if with_prob(0.10):
        format_type = "year_first"
    else:
        format_type = "day_first"

    fmt = random.choice(["month_name", "numeric"])

    if fmt == "month_name":
        short = with_prob(0.6)
        s = fmt_with_month_name(d, short=short, format_type=format_type)
        if with_prob(0.4):
            s = s.lower()
        elif with_prob(0.4):
            s = s.title()
        s = inject_month_typos(s)
    else:
        s = fmt_numeric(d, format_type=format_type)

    if with_prob(0.2):
        s = s.replace(str(d.year % 100).zfill(2), "'" + str(d.year % 100).zfill(2))

    if with_prob(0.3):
        s = " ".join(s.split())
    if with_prob(0.15):
        s = s.replace(" ", "")

    if with_prob(0.4):
        s = maybe_add_noise_tokens(s)

    if with_prob(0.5):
        s = reinforce_zero_O_confusion(s)

    s = corrupt_string(s, char_p=0.06, drop_p=0.01, dup_p=0.01)

    return s, target


def generate_dataset(n: int) -> List[Tuple[str, str]]:
    seen = set()
    pairs = []
    while len(pairs) < n:
        src, tgt = generate_example()
        key = (src, tgt)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return pairs


def write_csv(path: str, rows: List[Tuple[str, str]]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["input", "target"])
        for src, tgt in rows:
            w.writerow([src, tgt])


def main():
    try:
        random.seed(42)
        N = int(os.getenv("N_SAMPLES", "200000"))
        print(f"[generate_data] Generating {N} samples...")
        pairs = generate_dataset(N)
        random.shuffle(pairs)
        n_train = int(N * 0.9)
        n_val = int(N * 0.05)
        train = pairs[:n_train]
        val = pairs[n_train : n_train + n_val]
        test = pairs[n_train + n_val :]
        write_csv("data/train.csv", train)
        write_csv("data/val.csv", val)
        write_csv("data/test.csv", test)
        print(
            f"[generate_data] Done. Train={len(train)} Val={len(val)} Test={len(test)} -> written to data/"
        )
    except Exception as e:
        print("[generate_data] ERROR:", e)
        traceback.print_exc(file=sys.stdout)


if __name__ == "__main__":
    main()

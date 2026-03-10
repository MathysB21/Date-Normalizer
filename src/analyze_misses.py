import pandas as pd
import re
from collections import Counter
from datetime import datetime

# Path to your test results CSV
CSV_PATH = "data/test_with_preds.csv"


def classify_error(row):
    """
    Classify the type of error between target and prediction.
    """
    target = str(row["target"]).strip()
    pred = str(row["pred"]).strip()

    # Exact match check (shouldn't be here if ok==0, but just in case)
    if target == pred:
        return "Exact match (should be ok)"

    # Try parsing both as dates
    try:
        t_date = datetime.strptime(target, "%Y-%m-%d")
    except ValueError:
        t_date = None
    try:
        p_date = datetime.strptime(pred, "%Y-%m-%d")
    except ValueError:
        p_date = None

    # If prediction is not a valid date
    if p_date is None:
        return "Invalid date format"

    # If target is valid but pred is valid too, check differences
    if t_date and p_date:
        if t_date.year != p_date.year:
            return "Year mismatch"
        elif t_date.month != p_date.month:
            return "Month mismatch"
        elif t_date.day != p_date.day:
            return "Day mismatch"
        else:
            return "Formatting mismatch"

    # Fallback
    return "Other mismatch"


def detect_common_patterns(df):
    """
    Look for common OCR-like issues in the wrong predictions.
    """
    patterns = {
        "O_as_0": re.compile(r"[O]"),
        "0_as_O": re.compile(r"0"),
        "l_as_1": re.compile(r"[lI]"),
        "missing_zero_pad": re.compile(r"-\d-|\d-\d-"),
    }

    pattern_counts = Counter()

    for _, row in df.iterrows():
        pred = str(row["pred"])
        for name, regex in patterns.items():
            if regex.search(pred):
                pattern_counts[name] += 1

    return pattern_counts


def main():
    df = pd.read_csv(CSV_PATH)

    # Filter misses
    misses = df[df["ok"] == 0].copy()

    print(f"Total test samples: {len(df)}")
    print(f"Total misses: {len(misses)}")
    print(f"Accuracy: {100 * (1 - len(misses) / len(df)):.2f}%\n")

    # Classify error types
    misses["error_type"] = misses.apply(classify_error, axis=1)
    error_counts = misses["error_type"].value_counts()

    print("=== Error Type Breakdown ===")
    for err_type, count in error_counts.items():
        print(f"{err_type}: {count} ({count/len(misses)*100:.1f}%)")
    print()

    # Detect common OCR-like patterns in predictions
    pattern_counts = detect_common_patterns(misses)
    print("=== Common Pattern Matches in Predictions ===")
    for pattern, count in pattern_counts.items():
        print(f"{pattern}: {count} ({count/len(misses)*100:.1f}%)")
    print()

    # Show a few sample misses for manual inspection
    print("=== Sample Misses ===")
    print(
        misses[["input", "target", "pred", "error_type"]]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()

"""Rule-based baseline for AEGIS risk scoring, using the team's shared contract.

Input per image:
{
  "ocr_pii_flag": bool,            # OCR + NER found personal info
  "pii_types": list[str],          # e.g. ["address", "phone"]
  "landmark_flag": bool,           # landmark recognised
  "landmark_confidence": float     # 0.0 to 1.0
}
Output: (score 0-100, label "low" | "medium" | "high")
"""

def rule_based_risk(f: dict) -> tuple[int, str]:
    score = 0
    if f["ocr_pii_flag"]:
        score += 20
        score += min(len(f["pii_types"]), 3) * 10     # more PII types, more risk
        if "address" in f["pii_types"]:
            score += 20                               # an address is the worst text leak
    if f["landmark_flag"]:
        score += int(30 * f["landmark_confidence"])   # confident landmark = location clue
    score = min(score, 100)

    label = "high" if score >= 60 else "medium" if score >= 30 else "low"
    return score, label


if __name__ == "__main__":
    samples = [
        {"ocr_pii_flag": False, "pii_types": [], "landmark_flag": False, "landmark_confidence": 0.0},
        {"ocr_pii_flag": False, "pii_types": [], "landmark_flag": True, "landmark_confidence": 0.9},
        {"ocr_pii_flag": True, "pii_types": ["address", "phone"], "landmark_flag": True, "landmark_confidence": 0.8},
    ]
    for s in samples:
        print(s, "->", rule_based_risk(s))
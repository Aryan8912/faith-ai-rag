# moderation/safety.py
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModerationResult:
    allowed: bool
    level: str
    flag: Optional[str]
    reason: Optional[str]


HARD_BLOCK_PATTERNS = [
    (r"rewrite\s+(the\s+)?(bible|scripture|verse|gospel).*(support|justify|promote)", "scripture_manipulation"),
    (r"make\s+(the\s+)?(bible|god|jesus).*(say|support).*(hate|kill|harm|racist)", "scripture_manipulation"),
    (r"bible\s+verse\s+(to\s+justify|supporting)\s+(violence|hate|racism|terrorism)", "scripture_manipulation"),
    (r"\b(kill|murder|harm)\s+(all\s+)?(jews|muslims|christians|atheists|lgbt)\b", "hate_speech"),
    (r"god\s+(wants|commands)\s+(us\s+to\s+)?(kill|hate|destroy)\s+", "hate_speech"),
    (r"christian\s+(terrorism|genocide|ethnic\s+cleansing)", "extremism"),
    (r"\b(child|minor|underage)\b.*(sexual|nude|naked|explicit)", "csam"),
    (r"use\s+(the\s+)?bible\s+to\s+(prove|justify)\s+(white\s+supremac|nazism|fascism)", "ideology_injection"),
]

SOFT_FLAG_PATTERNS = [
    (r"why\s+does\s+god\s+(allow|permit|cause)\s+(evil|suffering|pain|death)", "theodicy"),
    (r"(is\s+hell\s+real|eternal\s+(damnation|punishment)|do\s+people\s+go\s+to\s+hell)", "eschatology"),
    (r"\b(lgbt|gay|homosexual|transgender)\b.*(bible|christian|sin|church)", "sensitive_social"),
    (r"(abortion|contraception).*(bible|christian|sin|church)", "sensitive_social"),
    # Fixed: broader pattern for denomination rivalry
    (r"which\s+(denomination|church|religion)\s+is\s+(right|true|correct|best|correct one)", "denomination_rivalry"),
    (r"(correct|right|true)\s+(denomination|church|religion)", "denomination_rivalry"),
    (r"was\s+(the\s+)?(crusades|inquisition|slavery).*(christian|justified)", "historical_atrocities"),
    (r"(prove|disprove)\s+(god\s+exists|christianity\s+is\s+(true|false))", "apologetics"),
    (r"(fake|made.?up|false|wrong)\s+(bible\s+verse|scripture|passage)", "verse_verification"),
    (r"did\s+jesus\s+(really\s+)?(exist|rise|perform\s+miracles)", "historicity"),
]

IMAGE_BLOCK_PATTERNS = [
    r"jesus.*(nazi|kkk|white\s+supremac)",
    r"(sexual|nude|naked|erotic).*(jesus|mary|god|angel|saint)",
    r"(demonic|satanic|occult).*(cross|church|bible|jesus)",
    r"mock(ing)?\s+(jesus|god|mary|bible|church)",
    r"(racist|nazi|kkk).*christian",
    r"graphic.*(crucif|wound|gore|blood)",
]

HALLUCINATION_PATTERNS = [
    (r"\b4\s+(kings?|john|corinthians?|peter|acts)\b", "nonexistent_book"),
    (r"\b5\s+(john|peter|acts|kings?)\b", "nonexistent_book"),
    (r"\b3\s+corinthians?\b", "nonexistent_book"),
    (r"revelation\s+2[3-9]\b", "chapter_out_of_bounds"),
    # Fixed: match Psalms 151-999
    (r"psalm[s]?\s+(1[5-9][1-9]|[2-9]\d{2})\b", "chapter_out_of_bounds"),
    (r"genesis\s+5[1-9]\b", "chapter_out_of_bounds"),
    (r"john\s+2[2-9]\b", "chapter_out_of_bounds"),
    (r"philippians\s+[5-9]\b", "chapter_out_of_bounds"),
]

DENOMINATION_PATTERNS = {
    "catholic": r"\b(catholic|pope|vatican|mass|rosary|purgatory|transubstantiation|catechism|confession)\b",
    "orthodox": r"\b(orthodox|patriarch|theosis|iconostasis|byzantine|coptic|greek\s+orthodox|russian\s+orthodox)\b",
    # Fixed: added more protestant keywords
    "protestant": r"\b(baptist|calvinist|reformed|presbyterian|methodist|lutheran|anglican|evangelical|pentecostal|protestant|sola\s+scriptura|reformation)\b",
    "fringe": r"\b(mormon|lds|latter.?day|jehovah.?witness|seventh.?day\s+adventist)\b",
}


def moderate_text(text: str) -> ModerationResult:
    text_lower = text.lower()
    for pattern, flag in HARD_BLOCK_PATTERNS:
        if re.search(pattern, text_lower):
            return ModerationResult(allowed=False, level="hard_block", flag=flag,
                reason="This request violates our content guidelines and cannot be processed.")
    for pattern, flag in SOFT_FLAG_PATTERNS:
        if re.search(pattern, text_lower):
            return ModerationResult(allowed=True, level="soft_flag", flag=flag, reason=None)
    return ModerationResult(allowed=True, level="clean", flag=None, reason=None)


def moderate_image_prompt(prompt: str) -> ModerationResult:
    prompt_lower = prompt.lower()
    for pattern in IMAGE_BLOCK_PATTERNS:
        if re.search(pattern, prompt_lower):
            return ModerationResult(allowed=False, level="hard_block", flag="image_policy",
                reason="This image request cannot be generated. Please request respectful Christian imagery.")
    return ModerationResult(allowed=True, level="clean", flag=None, reason=None)


def detect_hallucination_attempt(text: str) -> Optional[str]:
    text_lower = text.lower()
    for pattern, reason in HALLUCINATION_PATTERNS:
        if re.search(pattern, text_lower):
            return f"⚠️ Suspicious reference detected ({reason}). This book/chapter may not exist in the Bible."
    return None


def detect_denomination(text: str) -> str:
    text_lower = text.lower()
    for denom, pattern in DENOMINATION_PATTERNS.items():
        if re.search(pattern, text_lower):
            return denom
    return "general"
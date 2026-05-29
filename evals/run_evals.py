# evals/run_evals.py
# DeepEval-based evaluation suite for Faith AI
# Tests: RAG quality, hallucination detection, moderation, denomination awareness

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRecallMetric,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase
from moderation.safety import moderate_text, moderate_image_prompt, detect_hallucination_attempt
from chain.qa_chain import ask

# ── Load test cases ───────────────────────────────────────────
with open(os.path.join(os.path.dirname(__file__), "test_cases.json")) as f:
    TEST_CASES = json.load(f)


# ── Moderation Tests (no LLM needed) ─────────────────────────
def run_moderation_tests():
    print("\n" + "="*60)
    print("MODERATION TESTS")
    print("="*60)

    passed = 0
    failed = 0

    for tc in TEST_CASES:
        if tc.get("image"):
            result = moderate_image_prompt(tc["input"])
        else:
            result = moderate_text(tc["input"])

        expected_block = tc.get("should_block", False)
        actual_block = not result.allowed

        if expected_block == actual_block:
            print(f"  ✅ {tc['id']} [{tc['category']}] - PASS")
            passed += 1
        else:
            print(f"  ❌ {tc['id']} [{tc['category']}] - FAIL")
            print(f"     Expected block={expected_block}, got block={actual_block}")
            print(f"     Input: {tc['input'][:60]}")
            failed += 1

    print(f"\n  Results: {passed} passed, {failed} failed out of {passed+failed} tests")
    return passed, failed


# ── Hallucination Guard Tests (no LLM needed) ─────────────────
def run_hallucination_tests():
    print("\n" + "="*60)
    print("HALLUCINATION GUARD TESTS")
    print("="*60)

    passed = 0
    failed = 0

    hallucination_cases = [tc for tc in TEST_CASES if tc.get("should_contain_hallucination_warning")]

    for tc in hallucination_cases:
        warning = detect_hallucination_attempt(tc["input"])
        if warning is not None:
            print(f"  ✅ {tc['id']} - PASS | Warning: {warning}")
            passed += 1
        else:
            print(f"  ❌ {tc['id']} - FAIL | No warning detected for: {tc['input']}")
            failed += 1

    # Also test that valid verses don't trigger warnings
    valid_cases = [
        "What does John 3:16 say?",
        "Tell me about Romans 8:28",
        "What is in Psalms 23?",
    ]
    for case in valid_cases:
        warning = detect_hallucination_attempt(case)
        if warning is None:
            print(f"  ✅ Valid verse no false positive - PASS | {case}")
            passed += 1
        else:
            print(f"  ❌ False positive - FAIL | {case} | Warning: {warning}")
            failed += 1

    print(f"\n  Results: {passed} passed, {failed} failed out of {passed+failed} tests")
    return passed, failed


# ── DeepEval RAG Quality Tests (needs LLM) ───────────────────
def run_deepeval_tests():
    print("\n" + "="*60)
    print("DEEPEVAL RAG QUALITY TESTS")
    print("="*60)

    # Only run on non-blocked, basic QA cases
    qa_cases = [tc for tc in TEST_CASES if not tc.get("should_block") and tc["category"] == "basic_qa"]

    deepeval_test_cases = []

    for tc in qa_cases:
        print(f"  Running {tc['id']}: {tc['input'][:50]}...")
        try:
            result = ask(tc["input"], session_id=tc["id"])
            from rag.retriever import retrieve_verses, format_context
            docs = retrieve_verses(tc["input"], k=4)
            context = [doc.page_content for doc in docs]

            test_case = LLMTestCase(
                input=tc["input"],
                actual_output=result["answer"],
                retrieval_context=context,
                expected_output=tc.get("expected_output_contains", ""),
            )
            deepeval_test_cases.append(test_case)
        except Exception as e:
            print(f"  ⚠️  Skipped {tc['id']} due to error: {e}")

    if not deepeval_test_cases:
        print("  ⚠️  No test cases ran (likely quota issue). Run again after quota resets.")
        return

    # Define metrics
    metrics = [
        AnswerRelevancyMetric(threshold=0.7, model="gpt-4o-mini"),
        FaithfulnessMetric(threshold=0.7, model="gpt-4o-mini"),
        HallucinationMetric(threshold=0.3, model="gpt-4o-mini"),
    ]

    print("\n  Running DeepEval metrics...")
    evaluate(deepeval_test_cases, metrics)


# ── Summary ───────────────────────────────────────────────────
def run_all():
    print("\n🔍 FAITH AI — EVALUATION SUITE")
    print("Using DeepEval for RAG quality metrics\n")

    mod_pass, mod_fail = run_moderation_tests()
    hall_pass, hall_fail = run_hallucination_tests()

    total_pass = mod_pass + hall_pass
    total_fail = mod_fail + hall_fail

    print("\n" + "="*60)
    print("FINAL SUMMARY (Rule-based tests)")
    print("="*60)
    print(f"  Moderation:  {mod_pass}/{mod_pass+mod_fail} passed")
    print(f"  Hallucination Guard: {hall_pass}/{hall_pass+hall_fail} passed")
    print(f"  Total: {total_pass}/{total_pass+total_fail} passed")

    print("\n" + "="*60)
    print("DeepEval RAG Tests (requires LLM quota)")
    print("="*60)
    try:
        run_deepeval_tests()
    except Exception as e:
        print(f"  ⚠️  DeepEval tests skipped: {e}")
        print("  Run again after Gemini quota resets or add OpenAI key for DeepEval metrics")


if __name__ == "__main__":
    run_all()
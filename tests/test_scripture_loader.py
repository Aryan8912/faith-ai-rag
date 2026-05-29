# tests/test_scripture_loader.py
# Tests for scripture_loader.py — no API needed

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.scripture_loader import load_documents, is_canonical_book, check_chapter_bounds

def test_load_documents():
    docs = load_documents()
    assert len(docs) > 0, "No documents loaded"
    assert docs[0].page_content != "", "First doc is empty"
    assert "ref" in docs[0].metadata
    assert "topic" in docs[0].metadata
    print(f"  ✅ Loaded {len(docs)} verse documents")

def test_canonical_books_valid():
    valid = ["Genesis", "John", "Psalms", "Revelation", "Matthew", "Romans"]
    for book in valid:
        assert is_canonical_book(book), f"{book} should be canonical"
    print(f"  ✅ Valid canonical books recognized")

def test_canonical_books_invalid():
    invalid = ["FakeBook", "4 Kings", "3 Corinthians", "5 John", "Hezekiah"]
    for book in invalid:
        assert not is_canonical_book(book), f"{book} should NOT be canonical"
    print(f"  ✅ Invalid books correctly rejected")

def test_deuterocanonical_books():
    # Catholic/Orthodox books
    deutero = ["Tobit", "Judith", "Sirach", "Wisdom", "1 Maccabees"]
    for book in deutero:
        assert is_canonical_book(book), f"{book} should be in deuterocanonical"
    print(f"  ✅ Deuterocanonical books recognized")

def test_chapter_bounds_valid():
    assert check_chapter_bounds("John", 3) == True
    assert check_chapter_bounds("Psalms", 150) == True
    assert check_chapter_bounds("Genesis", 50) == True
    assert check_chapter_bounds("Revelation", 22) == True
    print(f"  ✅ Valid chapter bounds pass")

def test_chapter_bounds_invalid():
    assert check_chapter_bounds("John", 99) == False
    assert check_chapter_bounds("Psalms", 200) == False
    assert check_chapter_bounds("Genesis", 51) == False
    assert check_chapter_bounds("Revelation", 23) == False
    print(f"  ✅ Invalid chapter bounds correctly rejected")

if __name__ == "__main__":
    print("\n=== test_scripture_loader.py ===")
    test_load_documents()
    test_canonical_books_valid()
    test_canonical_books_invalid()
    test_deuterocanonical_books()
    test_chapter_bounds_valid()
    test_chapter_bounds_invalid()
    print("\n✅ All scripture loader tests passed!")
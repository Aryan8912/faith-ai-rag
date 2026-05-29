# rag/scripture_loader.py
# Loads curated Bible verses as LangChain Documents for FAISS indexing

from langchain_core.documents import Document

VERSES = [
    # Salvation
    {"ref": "John 3:16", "text": "For God so loved the world that he gave his one and only Son, that whoever believes in him shall not perish but have eternal life.", "topic": "salvation"},
    {"ref": "Romans 10:9", "text": "If you declare with your mouth Jesus is Lord and believe in your heart that God raised him from the dead, you will be saved.", "topic": "salvation"},
    {"ref": "Ephesians 2:8-9", "text": "For it is by grace you have been saved, through faith and this is not from yourselves, it is the gift of God, not by works, so that no one can boast.", "topic": "salvation"},
    {"ref": "Acts 4:12", "text": "Salvation is found in no one else, for there is no other name under heaven given to mankind by which we must be saved.", "topic": "salvation"},

    # Love
    {"ref": "1 Corinthians 13:4-7", "text": "Love is patient, love is kind. It does not envy, it does not boast, it is not proud. It does not dishonor others, it is not self-seeking, it is not easily angered, it keeps no record of wrongs.", "topic": "love"},
    {"ref": "John 13:34", "text": "A new command I give you: Love one another. As I have loved you, so you must love one another.", "topic": "love"},
    {"ref": "Romans 8:38-39", "text": "Neither death nor life, neither angels nor demons, neither the present nor the future will be able to separate us from the love of God.", "topic": "love"},
    {"ref": "1 John 4:8", "text": "Whoever does not love does not know God, because God is love.", "topic": "love"},

    # Faith
    {"ref": "Hebrews 11:1", "text": "Now faith is confidence in what we hope for and assurance about what we do not see.", "topic": "faith"},
    {"ref": "James 2:17", "text": "In the same way, faith by itself, if it is not accompanied by action, is dead.", "topic": "faith"},
    {"ref": "Romans 10:17", "text": "Faith comes from hearing the message, and the message is heard through the word about Christ.", "topic": "faith"},
    {"ref": "Matthew 17:20", "text": "Truly I tell you, if you have faith as small as a mustard seed, you can say to this mountain move from here to there and it will move.", "topic": "faith"},

    # Prayer
    {"ref": "Matthew 6:9-13", "text": "Our Father in heaven, hallowed be your name, your kingdom come, your will be done, on earth as it is in heaven. Give us today our daily bread.", "topic": "prayer"},
    {"ref": "Philippians 4:6-7", "text": "Do not be anxious about anything, but in every situation, by prayer and petition, with thanksgiving, present your requests to God.", "topic": "prayer"},
    {"ref": "1 Thessalonians 5:17", "text": "Pray continually.", "topic": "prayer"},
    {"ref": "James 5:16", "text": "The prayer of a righteous person is powerful and effective.", "topic": "prayer"},

    # Hope
    {"ref": "Jeremiah 29:11", "text": "For I know the plans I have for you declares the Lord, plans to prosper you and not to harm you, plans to give you hope and a future.", "topic": "hope"},
    {"ref": "Romans 15:13", "text": "May the God of hope fill you with all joy and peace as you trust in him, so that you may overflow with hope by the power of the Holy Spirit.", "topic": "hope"},
    {"ref": "Lamentations 3:22-23", "text": "Because of the Lords great love we are not consumed, for his compassions never fail. They are new every morning; great is your faithfulness.", "topic": "hope"},

    # Forgiveness
    {"ref": "1 John 1:9", "text": "If we confess our sins, he is faithful and just and will forgive us our sins and purify us from all unrighteousness.", "topic": "forgiveness"},
    {"ref": "Ephesians 4:32", "text": "Be kind and compassionate to one another, forgiving each other, just as in Christ God forgave you.", "topic": "forgiveness"},
    {"ref": "Matthew 6:14", "text": "For if you forgive other people when they sin against you, your heavenly Father will also forgive you.", "topic": "forgiveness"},
    {"ref": "Colossians 3:13", "text": "Bear with each other and forgive one another if any of you has a grievance against someone. Forgive as the Lord forgave you.", "topic": "forgiveness"},

    # Suffering
    {"ref": "Romans 8:28", "text": "And we know that in all things God works for the good of those who love him, who have been called according to his purpose.", "topic": "suffering"},
    {"ref": "2 Corinthians 12:9", "text": "My grace is sufficient for you, for my power is made perfect in weakness.", "topic": "suffering"},
    {"ref": "Psalm 34:18", "text": "The Lord is close to the brokenhearted and saves those who are crushed in spirit.", "topic": "suffering"},
    {"ref": "James 1:2-3", "text": "Consider it pure joy whenever you face trials of many kinds, because you know that the testing of your faith produces perseverance.", "topic": "suffering"},

    # Wisdom
    {"ref": "Proverbs 3:5-6", "text": "Trust in the Lord with all your heart and lean not on your own understanding; in all your ways submit to him, and he will make your paths straight.", "topic": "wisdom"},
    {"ref": "James 1:5", "text": "If any of you lacks wisdom, you should ask God, who gives generously to all without finding fault, and it will be given to you.", "topic": "wisdom"},
    {"ref": "Proverbs 9:10", "text": "The fear of the Lord is the beginning of wisdom, and knowledge of the Holy One is understanding.", "topic": "wisdom"},

    # Grace
    {"ref": "Romans 6:14", "text": "Sin shall no longer be your master, because you are not under the law, but under grace.", "topic": "grace"},
    {"ref": "Titus 2:11", "text": "For the grace of God has appeared that offers salvation to all people.", "topic": "grace"},
    {"ref": "2 Corinthians 9:8", "text": "God is able to bless you abundantly, so that in all things at all times, having all that you need, you will abound in every good work.", "topic": "grace"},

    # Resurrection
    {"ref": "John 11:25-26", "text": "I am the resurrection and the life. The one who believes in me will live, even though they die; and whoever lives by believing in me will never die.", "topic": "resurrection"},
    {"ref": "1 Corinthians 15:20", "text": "But Christ has indeed been raised from the dead, the firstfruits of those who have fallen asleep.", "topic": "resurrection"},
    {"ref": "Romans 6:4", "text": "We were therefore buried with him through baptism into death in order that, just as Christ was raised from the dead through the glory of the Father, we too may live a new life.", "topic": "resurrection"},

    # Trinity
    {"ref": "Matthew 28:19", "text": "Therefore go and make disciples of all nations, baptizing them in the name of the Father and of the Son and of the Holy Spirit.", "topic": "trinity"},
    {"ref": "2 Corinthians 13:14", "text": "May the grace of the Lord Jesus Christ, and the love of God, and the fellowship of the Holy Spirit be with you all.", "topic": "trinity"},
    {"ref": "John 10:30", "text": "I and the Father are one.", "topic": "trinity"},

    # Peace
    {"ref": "John 14:27", "text": "Peace I leave with you; my peace I give you. I do not give to you as the world gives. Do not let your hearts be troubled and do not be afraid.", "topic": "peace"},
    {"ref": "Isaiah 26:3", "text": "You will keep in perfect peace those whose minds are steadfast, because they trust in you.", "topic": "peace"},
    {"ref": "Romans 5:1", "text": "Since we have been justified through faith, we have peace with God through our Lord Jesus Christ.", "topic": "peace"},

    # Strength
    {"ref": "Philippians 4:13", "text": "I can do all this through him who gives me strength.", "topic": "strength"},
    {"ref": "Isaiah 40:31", "text": "But those who hope in the Lord will renew their strength. They will soar on wings like eagles; they will run and not grow weary, they will walk and not be faint.", "topic": "strength"},
    {"ref": "Psalm 46:1", "text": "God is our refuge and strength, an ever-present help in trouble.", "topic": "strength"},

    # Creation
    {"ref": "Genesis 1:1", "text": "In the beginning God created the heavens and the earth.", "topic": "creation"},
    {"ref": "Psalm 19:1", "text": "The heavens declare the glory of God; the skies proclaim the work of his hands.", "topic": "creation"},
    {"ref": "Colossians 1:16", "text": "For in him all things were created: things in heaven and on earth, visible and invisible.", "topic": "creation"},

    # Eternal life
    {"ref": "John 17:3", "text": "Now this is eternal life: that they know you, the only true God, and Jesus Christ, whom you have sent.", "topic": "eternal life"},
    {"ref": "Romans 6:23", "text": "For the wages of sin is death, but the gift of God is eternal life in Christ Jesus our Lord.", "topic": "eternal life"},
    {"ref": "1 John 5:13", "text": "I write these things to you who believe in the name of the Son of God so that you may know that you have eternal life.", "topic": "eternal life"},
]

CANONICAL_BOOKS = [
    "Genesis","Exodus","Leviticus","Numbers","Deuteronomy","Joshua","Judges",
    "Ruth","1 Samuel","2 Samuel","1 Kings","2 Kings","1 Chronicles","2 Chronicles",
    "Ezra","Nehemiah","Esther","Job","Psalms","Proverbs","Ecclesiastes",
    "Song of Solomon","Isaiah","Jeremiah","Lamentations","Ezekiel","Daniel",
    "Hosea","Joel","Amos","Obadiah","Jonah","Micah","Nahum","Habakkuk",
    "Zephaniah","Haggai","Zechariah","Malachi",
    "Matthew","Mark","Luke","John","Acts","Romans",
    "1 Corinthians","2 Corinthians","Galatians","Ephesians","Philippians",
    "Colossians","1 Thessalonians","2 Thessalonians","1 Timothy","2 Timothy",
    "Titus","Philemon","Hebrews","James","1 Peter","2 Peter",
    "1 John","2 John","3 John","Jude","Revelation",
    # Deuterocanonical (Catholic/Orthodox)
    "Tobit","Judith","1 Maccabees","2 Maccabees","Wisdom","Sirach","Baruch",
]

CHAPTER_BOUNDS = {
    "Genesis": 50, "Exodus": 40, "Psalms": 150, "Proverbs": 31,
    "Isaiah": 66, "Jeremiah": 52, "Matthew": 28, "Mark": 16,
    "Luke": 24, "John": 21, "Acts": 28, "Romans": 16,
    "Revelation": 22, "Hebrews": 13, "James": 5,
    "1 Corinthians": 16, "2 Corinthians": 13,
    "Ephesians": 6, "Philippians": 4, "Galatians": 6, "1 John": 5,
}


def load_documents() -> list[Document]:
    """Convert verse dicts into LangChain Documents for FAISS indexing."""
    return [
        Document(
            page_content=f"{v['ref']}: {v['text']}",
            metadata={"ref": v["ref"], "topic": v["topic"], "text": v["text"]}
        )
        for v in VERSES
    ]


def is_canonical_book(book: str) -> bool:
    return any(b.lower() == book.strip().lower() for b in CANONICAL_BOOKS)


def check_chapter_bounds(book: str, chapter: int) -> bool:
    max_ch = CHAPTER_BOUNDS.get(book)
    if max_ch is None:
        return True
    return chapter <= max_ch


if __name__ == "__main__":
    print("=== Scripture Loader Test ===")

    docs = load_documents()
    print(f"✅ Total verses loaded: {len(docs)}")
    print(f"📖 First verse: {docs[0].page_content}")
    print(f"📌 Metadata: {docs[0].metadata}")

    print("\n--- Canonical Book Check ---")
    print(f"✅ John is canonical:     {is_canonical_book('John')}")
    print(f"✅ Genesis is canonical:  {is_canonical_book('Genesis')}")
    print(f"✅ Sirach is canonical:   {is_canonical_book('Sirach')}")
    print(f"❌ FakeBook is canonical: {is_canonical_book('FakeBook')}")
    print(f"❌ 4 Kings is canonical:  {is_canonical_book('4 Kings')}")

    print("\n--- Chapter Bounds Check ---")
    print(f"✅ John ch 3 valid:      {check_chapter_bounds('John', 3)}")
    print(f"❌ John ch 99 valid:     {check_chapter_bounds('John', 99)}")
    print(f"✅ Psalms ch 150 valid:  {check_chapter_bounds('Psalms', 150)}")
    print(f"❌ Psalms ch 200 valid:  {check_chapter_bounds('Psalms', 200)}")
    print(f"✅ Genesis ch 50 valid:  {check_chapter_bounds('Genesis', 50)}")
    print(f"❌ Genesis ch 51 valid:  {check_chapter_bounds('Genesis', 51)}")
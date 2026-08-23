"""
Arabic Sign Language + Quranic signing knowledge base.
Builds a vector store from static reference content.
Falls back to keyword search if sentence-transformers or
chromadb are not installed.
"""
import os
from pathlib import Path

KNOWLEDGE_DOCS = [
    {
        "id": "intro_arsl",
        "content": (
            "لغة الإشارة العربية (ArSL) هي لغة بصرية يستخدمها "
            "الصم وضعاف السمع في الدول العربية للتواصل. تعتمد "
            "على حركات اليدين وتعبيرات الوجه بدلاً من الكلام."
        ),
    },
    {
        "id": "letter_transitions",
        "content": (
            "بعض الحروف العربية في لغة الإشارة تتشابه بصرياً "
            "وقد تسبب ارتباكاً، مثل ع وغ، أو ب وت. الفرق الدقيق "
            "غالباً في اتجاه الأصابع أو موضع اليد بالنسبة للوجه."
        ),
    },
    {
        "id": "fatiha_signing",
        "content": (
            "تلاوة سورة الفاتحة بلغة الإشارة تتطلب توقيع كل حرف "
            "بوضوح مع الحفاظ على تسلسل الكلمات. يُنصح المبتدئون "
            "بالتدرب على الحروف المنفردة أولاً قبل الانتقال "
            "لتلاوة السورة كاملة."
        ),
    },
    {
        "id": "practice_tips",
        "content": (
            "لتحسين دقة التعرف على إشاراتك: تأكد من إضاءة جيدة، "
            "ضع يدك في وسط الكاميرا، تجنب الخلفيات المزدحمة، "
            "وحافظ على مسافة ثابتة بين يدك والكاميرا."
        ),
    },
    {
        "id": "front_back_hand",
        "content": (
            "بعض الحروف مثل ط وظ يمكن توقيعها بإظهار باطن اليد "
            "(الأمامي) أو ظاهرها (الخلفي). ميسورة تدعم كلا "
            "الوضعين، لكن التدريب على الوضع الأمامي حالياً "
            "أكثر دقة."
        ),
    },
    {
        "id": "about_mysora",
        "content": (
            "ميسورة هو أول نظام ذكاء اصطناعي يتعرف على لغة "
            "الإشارة العربية في الوقت الفعلي، مصمم لمساعدة "
            "الصم على تعلم القرآن الكريم باستقلالية. يستخدم "
            "MediaPipe لاستخراج نقاط اليد وResNet-50 للتصنيف."
        ),
    },
    {
        "id": "weak_letters_context",
        "content": (
            "الحروف التي تحتاج أكثر تحسين في نموذج ميسورة الحالي "
            "هي: ح، و، ق، ب، ث، ز، ط، ظ. يمكن للمستخدمين المساهمة "
            "في تحسين النموذج عبر توفير عينات إشارة عبر صفحة جمع البيانات."
        ),
    },
]

_vectorstore = None
_use_vector = True


def _keyword_search(query: str, k: int = 2) -> list:
    """Simple keyword fallback when vector store is unavailable."""
    query_lower = query.lower()
    scored = []
    for doc in KNOWLEDGE_DOCS:
        score = sum(1 for word in query_lower.split() if word in doc["content"])
        scored.append((score, doc["content"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [content for _, content in scored[:k] if _ > 0] or [KNOWLEDGE_DOCS[3]["content"]]


def get_vectorstore():
    global _vectorstore, _use_vector
    if not _use_vector:
        return None
    if _vectorstore is not None:
        return _vectorstore

    try:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings

        from langchain_community.vectorstores import Chroma

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        texts = [doc["content"] for doc in KNOWLEDGE_DOCS]
        metadatas = [{"id": doc["id"]} for doc in KNOWLEDGE_DOCS]
        persist_dir = Path(os.environ.get("DATA_DIR", "outputs")) / "chroma_kb"
        persist_dir.mkdir(parents=True, exist_ok=True)
        _vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            persist_directory=str(persist_dir),
        )
        return _vectorstore
    except Exception:
        _use_vector = False
        return None


def search_knowledge_base(query: str, k: int = 2) -> list:
    """Return top-k relevant knowledge snippets for a query."""
    try:
        vs = get_vectorstore()
        if vs is None:
            return _keyword_search(query, k)
        results = vs.similarity_search(query, k=k)
        return [r.page_content for r in results]
    except Exception:
        return _keyword_search(query, k)

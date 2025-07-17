import re, os
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

def extract_meta(text: str) -> dict:
    type_keywords = {
        "QUYẾT ĐỊNH": "Quyết định",
        "NGHỊ ĐỊNH": "Nghị định",
        "THÔNG TƯ": "Thông tư",
        "LUẬT": "Luật"
    }
    doc_type = next((v for k, v in type_keywords.items() if k in text.upper()), "Văn bản")

    number_match = re.search(r"Số\s*:\s*([\d\w/-]+)", text, re.I)
    number = number_match.group(1).strip() if number_match else ""

    date = (re.search(r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})", text) or [""])[0]

    title_match = re.search(rf"{doc_type.upper()}\s*\n(.*?)(?=Căn cứ|$)", text, flags=re.S | re.I)
    title = title_match.group(1).strip().replace("\n", " ") if title_match else ""

    return {"doc_type": doc_type, "number": number, "date": date, "title": title}

def build_chapter_map(text: str) -> list[tuple]:
    return [(m.start(), m.group(1)) for m in re.finditer(r"(Chương\s+[IVXLCDM\d]+)", text)]

def split_articles(text: str, doc_type: str) -> list[dict]:
    start = re.search(r"Điều\s+1\b", text)
    if not start:
        return []

    chapter_map = build_chapter_map(text) if doc_type in {"Luật", "Nghị định"} else []
    text_from_art1 = text[start.start():]

    matches = re.finditer(r"(Điều\s+\d+(?:\s+[a-zA-ZÀ-ỹ]*)?)\b\s*\.\s*(.*?)(?=Điều\s+\d+|$)", text_from_art1, flags=re.S)
    articles = []
    for m in matches:
        abs_pos = start.start() + m.start()
        chapter = None
        for ch_pos, ch in chapter_map:
            if ch_pos <= abs_pos:
                chapter = ch
            else:
                break
        articles.append({
            "article": m.group(1).strip(),
            "chapter": chapter,
            "content": m.group(2).strip()
        })
    return articles

def split_documents(text: str) -> list[Document]:
    meta = extract_meta(text)
    articles = split_articles(text, meta["doc_type"])

    docs = []
    for art in articles:
        header = f"{art['article']}, {meta['doc_type']} {meta['title']} số {meta['number']} {meta['date']}"

        full_content = f"{header}\n{art['content']}"
        docs.append(
            Document(
                page_content=full_content,
                metadata={
                    "article": art["article"],
                    "chapter": art["chapter"],
                    "doc_type": meta["doc_type"],
                    "number": meta["number"],
                    "date": meta["date"],
                    "title": meta["title"]
                }
            )
        )
    return docs

def load_and_split_docs_dir(dir_path: str = "data"):
    os.makedirs(dir_path, exist_ok=True)
    all_docs = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.lower().endswith(".pdf"):
                file_path = os.path.join(root, file)
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                text = "".join([doc.page_content for doc in docs])
                split_docs = split_documents(text)
                all_docs.extend(split_docs)
    return all_docs
    
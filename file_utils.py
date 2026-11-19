import io

import pdfplumber


def read_uploaded_bytes(filename, content_bytes):
    name = filename.lower()

    if name.endswith(".pdf"):
        pages = []
        with pdfplumber.open(io.BytesIO(content_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    pages.append(page_text)
        return "\\n".join(pages).strip()

    if name.endswith(".txt") or name.endswith(".md"):
        return content_bytes.decode("utf-8", errors="replace").strip()

    if name.endswith(".docx"):
        try:
            from docx import Document
        except ImportError as exc:
            raise ValueError("DOCX support requires: pip install python-docx") from exc
        try:
            document = Document(io.BytesIO(content_bytes))
        except Exception:
            return ""
        return "\\n".join(p.text for p in document.paragraphs if p.text.strip()).strip()

    return ""

"""Download/register/inspect/confirm PDF files. No credentials required."""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse
from task2_bot.config import RAW, SOURCES

MAX_BYTES = 120 * 1024 * 1024

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def sources():
    return json.loads(SOURCES.read_text())

def source_by_id(source_id):
    for source in sources():
        if source["id"] == source_id:
            return source
    raise ValueError("Неизвестный source_id")

def metadata_path(source_id):
    return RAW / f"{source_id}.metadata.json"

def pdf_path(source_id):
    return RAW / f"{source_id}.pdf"

def inspect_pdf(path):
    from pypdf import PdfReader
    with Path(path).open("rb") as f:
        if f.read(5) != b"%PDF-":
            raise ValueError("Файл не PDF")
    reader = PdfReader(path)
    if reader.is_encrypted:
        raise ValueError("Зашифрованный PDF не поддерживается")
    preview = "\n".join((p.extract_text() or "")[:2500] for p in reader.pages[:5])
    return len(reader.pages), preview

def register(source_id, path, document_url=None):
    source = source_by_id(source_id)
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("PDF превышает лимит 120 MiB")
    pages, preview = inspect_pdf(path)
    RAW.mkdir(parents=True, exist_ok=True)
    target = pdf_path(source_id)
    if path.resolve() != target.resolve():
        shutil.copyfile(path, target)
    meta = dict(source_id=source_id, title=source.get("title", source_id),
                source_url=source["url"], document_url=document_url,
                file_name=target.name, sha256=sha256(target), pages=pages,
                downloaded_at_utc=datetime.now(timezone.utc).isoformat(),
                full_report_verified=False, review_note=None)
    if meta['sha256'] == source.get('verified_sha256') and pages == source.get('verified_pages'):
        meta.update(full_report_verified=True, review_note=source.get('review_note'))
    metadata_path(source_id).write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta, preview

def resolve_pdf(source, client):
    if source["type"] == "pdf":
        return source["url"]
    if source.get("resolved_document_url"):
        return source["resolved_document_url"]
    from bs4 import BeautifulSoup
    response = client.get(source["url"])
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    candidates = []
    for a in soup.select("a[href]"):
        url = urljoin(str(response.url), a["href"])
        if urlparse(url).path.lower().endswith(".pdf"):
            candidates.append((url, a.get_text(" ", strip=True).lower()))
    preferred = {u for u, label in candidates if "download pdf" in label}
    urls = preferred or {u for u, _ in candidates}
    if len(urls) != 1:
        raise ValueError("Нет однозначной ссылки PDF. Скачайте полный отчёт вручную и используйте register.")
    return urls.pop()

def download_one(source, client):
    target = pdf_path(source["id"])
    meta_path = metadata_path(source["id"])
    if target.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if sha256(target) == meta["sha256"]:
            return "уже загружен (хеш совпадает)"
    url = resolve_pdf(source, client)
    if urlparse(url).scheme != "https":
        raise ValueError("Ожидается HTTPS URL")
    RAW.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")
    try:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            total = 0
            with tmp.open("wb") as out:
                for block in response.iter_bytes():
                    total += len(block)
                    if total > MAX_BYTES:
                        raise ValueError("PDF превышает лимит 120 MiB")
                    out.write(block)
            actual_url = str(response.url)
        meta, _ = register(source["id"], tmp, actual_url)
        return f"загружен, {meta['pages']} страниц; проверка полноты: {meta['full_report_verified']}"
    finally:
        tmp.unlink(missing_ok=True)

def confirm(source_id, note):
    path = metadata_path(source_id)
    meta = json.loads(path.read_text())
    if sha256(pdf_path(source_id)) != meta["sha256"]:
        raise ValueError("Файл изменён: зарегистрируйте его заново")
    if len(note.strip()) < 15:
        raise ValueError("Опишите проверку названия, оглавления и разделов (минимум 15 символов)")
    meta.update(full_report_verified=True, review_note=note.strip())
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

def main():
    p = argparse.ArgumentParser(description="Документы RAG: скачивание и проверка")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch")
    for cmd in ("inspect", "confirm", "register"):
        sp = sub.add_parser(cmd)
        sp.add_argument("source_id", choices=[s["id"] for s in sources()])
        if cmd == "confirm": sp.add_argument("--note", required=True)
        if cmd == "register":
            sp.add_argument("file", type=Path)
            sp.add_argument("--url")
    a = p.parse_args()
    try:
        if a.command == "fetch":
            import httpx
            failures = 0
            with httpx.Client(timeout=60, follow_redirects=True, headers={"User-Agent": "SBER-RAG-study/1.0"}) as client:
                for s in sources():
                    try: print(s["id"], download_one(s, client))
                    except Exception as exc:
                        failures += 1
                        print(s["id"], f"не загружен ({type(exc).__name__}); используйте register для ручного PDF")
            if failures: raise SystemExit(1)
        elif a.command == "register":
            meta, _ = register(a.source_id, a.file, a.url)
            print(f"Зарегистрировано {meta['pages']} страниц. Выполните inspect и confirm.")
        elif a.command == "inspect":
            pages, preview = inspect_pdf(pdf_path(a.source_id))
            print(f"Страниц PDF: {pages}\n{preview}\nОткройте сам PDF и проверьте оглавление и основные разделы.")
        else:
            confirm(a.source_id, a.note)
            print("Проверка полноты записана; это ручное подтверждение, не автоматическая экспертиза.")
    except (ValueError, OSError, KeyError) as exc:
        p.exit(1, f"Ошибка: {exc}\n")

if __name__ == "__main__": main()

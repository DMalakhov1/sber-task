import argparse
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from task2_bot.config import INDEX, MODEL
from task2_bot.rag.download import sources, pdf_path, metadata_path, sha256


def split_text(text, tokenizer=None, size=380, overlap=60):
    """Semantic mode counts actual model tokens; lexical demo counts words."""
    if not 0 <= overlap < size:
        raise ValueError("Нужно 0 <= overlap < size")
    if tokenizer:
        tokens = tokenizer.encode(text, add_special_tokens=False, verbose=False)
        decode = lambda part: tokenizer.decode(part, skip_special_tokens=True)
    else:
        tokens = text.split()
        decode = lambda part: " ".join(part)
    for start in range(0, len(tokens), size - overlap):
        result = decode(tokens[start:start + size]).strip()
        if result: yield result
        if start + size >= len(tokens): break


def extract_chunks(source, meta, tokenizer=None):
    from pypdf import PdfReader
    reader = PdfReader(pdf_path(source["id"]))
    chunks, weak_pages = [], []
    for page_number, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        if len(text) < 80:
            weak_pages.append(page_number)
        for part_number, part in enumerate(split_text(text, tokenizer)):
            if len(part) < 30: continue
            chunks.append(dict(id=f"{source['id']}:p{page_number}:c{part_number}",
                         source_id=source["id"], title=meta["title"], page=page_number,
                         url=meta.get("document_url") or meta["source_url"], text=part))
    if not chunks or len(weak_pages) / max(len(reader.pages), 1) > 0.35:
        raise ValueError(f"{source['id']}: мало извлечённого текста; требуется OCR/другой парсер")
    return chunks, weak_pages


def save_index(chunks, destination, vectors=None, model=None, metadata=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if not chunks or len({c['id'] for c in chunks}) != len(chunks):
        raise ValueError("Пустой индекс или повторяющиеся chunk_id")
    if vectors is not None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(chunks) or not np.isfinite(vectors).all():
            raise ValueError("Неверная матрица эмбеддингов")
        vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    # Versioned payloads + atomic manifest prevent mixing old/new files after interruption.
    generation = uuid.uuid4().hex[:12]
    chunk_file = f"chunks-{generation}.json"
    vector_file = f"vectors-{generation}.npy" if vectors is not None else None
    (destination / chunk_file).write_text(json.dumps(chunks, ensure_ascii=False))
    if vectors is not None: np.save(destination / vector_file, vectors, allow_pickle=False)
    manifest = dict(version=1, chunks_file=chunk_file, vectors_file=vector_file,
                    chunks_sha256=sha256(destination / chunk_file),
                    vectors_sha256=sha256(destination / vector_file) if vector_file else None,
                    model=model, count=len(chunks), metadata=metadata or {},
                    created_at_utc=datetime.now(timezone.utc).isoformat())
    temporary = destination / "manifest.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    temporary.replace(destination / "manifest.json")
    return manifest


def main():
    p = argparse.ArgumentParser(description="Построить индекс трёх проверенных полных PDF")
    p.add_argument("--backend", choices=["semantic", "lexical"], default="semantic")
    p.add_argument("--model", choices=[MODEL, "intfloat/multilingual-e5-base"], default=MODEL)
    p.add_argument("--chunks-from", type=Path, help="Manifest existing index: reuse identical chunks to compare embeddings")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--output", type=Path, default=INDEX)
    a = p.parse_args()
    # Check corpus before downloading model weights.
    corpus = []
    for source in sources():
        mp = metadata_path(source["id"])
        if not mp.exists(): p.exit(1, f"Нет документа {source['id']}; сначала download fetch/register.\n")
        meta = json.loads(mp.read_text())
        if not meta.get("full_report_verified") or sha256(pdf_path(source["id"])) != meta["sha256"]:
            p.exit(1, f"Проверьте полноту и хеш {source['id']}; выполните download confirm.\n")
        corpus.append((source, meta))
    embedder = None
    if a.backend == "semantic":
        from task2_bot.rag.embeddings import Embedder
        embedder = Embedder(model=a.model, device=a.device)
    chunks, reports = [], {}
    for source, meta in corpus:
        found, weak = extract_chunks(source, meta, embedder.tokenizer if embedder else None)
        chunks.extend(found)
        reports[source["id"]] = dict(sha256=meta["sha256"], pages=meta["pages"], weak_pages=weak)
    if a.chunks_from:
        old = json.loads(a.chunks_from.read_text())
        chunk_path = (a.chunks_from.parent / old['chunks_file']).resolve()
        if chunk_path.parent != a.chunks_from.parent.resolve() or sha256(chunk_path) != old['chunks_sha256']:
            raise ValueError('Повреждён исходный набор фрагментов')
        for source_id, report in reports.items():
            if old.get('metadata', {}).get(source_id, {}).get('sha256') != report['sha256']:
                raise ValueError('Для сравнения моделей нужен одинаковый корпус')
        chunks = json.loads(chunk_path.read_text())
    vectors = embedder.passages([c["text"] for c in chunks]) if embedder else None
    save_index(chunks, a.output, vectors, a.model if embedder else None, reports)
    print(f"Готово: {len(chunks)} фрагментов. Backend: {a.backend}. Слабые страницы:")
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    if not embedder: print("Лексический режим — диагностика, не полноценный русско-английский поиск.")

if __name__ == "__main__": main()

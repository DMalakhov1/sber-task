from pathlib import Path
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
INDEX = ROOT / "rag_index"
SOURCES = ROOT / "sources.json"
MODEL = "intfloat/multilingual-e5-small"

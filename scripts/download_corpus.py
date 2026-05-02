import os
import sys
import logging
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.config import CORPUS_SOURCES, CORPUS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_file(url: str, dest_path: Path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    block_size = 1024
    with open(dest_path, 'wb') as f:
        for data in response.iter_content(block_size):
            f.write(data)
    logger.info(f"Downloaded: {dest_path.name}")

def main():
    pdf_dir = CORPUS_DIR / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    
    for source in CORPUS_SOURCES:
        url = source["url"]
        if url.endswith(".pdf"):
            filename = f"{source['tag']}_{source['name'].replace(' ', '_')}.pdf"
            dest_path = pdf_dir / filename
            if dest_path.exists():
                logger.info(f"Already exists: {filename}")
                continue
            logger.info(f"Downloading {source['name']}...")
            try:
                download_file(url, dest_path)
            except Exception as e:
                logger.error(f"Failed to download {url}: {e}")
        else:
            logger.warning(f"Skipping non-PDF URL for {source['name']}: {url}")

if __name__ == "__main__":
    main()

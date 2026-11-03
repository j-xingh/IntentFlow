from pathlib import Path
from urllib.request import urlopen
import hashlib
import csv
import io

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = (
    "https://raw.githubusercontent.com/"
    "PolyAI-LDN/task-specific-datasets/"
    "master/banking_data/"
)

EXPECTED = {
    "train.csv": {
        "rows": 10003,
        "sha256": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    },
    "test.csv": {
        "rows": 3080,
        "sha256": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
    },
}


def download_and_verify(filename):
    print(f"\nDownloading {filename}...")

    with urlopen(BASE_URL + filename, timeout=60) as response:
        content = response.read()

    digest = hashlib.sha256(content).hexdigest()

    if digest != EXPECTED[filename]["sha256"]:
        raise ValueError(
            f"Checksum mismatch for {filename}.\n"
            f"Downloaded size: {len(content):,} bytes\n"
            f"Downloaded SHA-256: {digest}\n"
            "The file was NOT replaced."
        )

    reader = csv.DictReader(
        io.StringIO(content.decode("utf-8-sig"))
    )

    if reader.fieldnames != ["text", "category"]:
        raise ValueError(
            f"Unexpected columns in {filename}: {reader.fieldnames}"
        )

    rows = list(reader)

    if len(rows) != EXPECTED[filename]["rows"]:
        raise ValueError(
            f"Expected {EXPECTED[filename]['rows']} rows, "
            f"found {len(rows)}."
        )

    categories = {row["category"] for row in rows}

    if len(categories) != 77:
        raise ValueError(
            f"Expected 77 categories, found {len(categories)}."
        )

    if any(
        not row["text"].strip() or not row["category"].strip()
        for row in rows
    ):
        raise ValueError("Empty text or category found.")

    destination = RAW_DIR / filename
    temporary = destination.with_suffix(".csv.tmp")
    temporary.write_bytes(content)
    temporary.replace(destination)

    print(f"Verified: {filename}")
    print(f"Rows: {len(rows):,}")
    print(f"Categories: {len(categories)}")
    print(f"Size: {len(content):,} bytes")
    print("SHA-256: MATCH")


for filename in EXPECTED:
    download_and_verify(filename)

print("\nSUCCESS: Dataset verification complete.")
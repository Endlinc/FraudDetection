"""Left-join transaction and identity CSV files using TransactionID.

Run `python join_data.py` to write data/train_joined.csv. Transactions without
an identity record are retained, with empty values in the identity columns.
Only Python's standard library is required.
"""

import argparse
import csv
import os
from pathlib import Path
import tempfile


KEY = "TransactionID"
DATA_DIR = Path(__file__).resolve().parent / "data"


def read_header(reader, path):
    header = next(reader, None)
    if not header or KEY not in header:
        raise ValueError(f"{path}: missing {KEY} column")
    if len(header) != len(set(header)):
        raise ValueError(f"{path}: duplicate column names")
    return header


def read_rows(reader, header, path):
    key_index = header.index(KEY)
    seen = set()
    for row in reader:
        if len(row) != len(header):
            raise ValueError(f"{path}, line {reader.line_num}: incorrect column count")
        key = row[key_index]
        if not key.strip() or key in seen:
            raise ValueError(f"{path}, line {reader.line_num}: empty or duplicate {KEY}")
        seen.add(key)
        yield key, row


def join_data(transactions: Path, identity: Path, output: Path):
    """Write a one-to-one left join; return transaction and matched row counts."""
    if output.resolve() in {transactions.resolve(), identity.resolve()}:
        raise ValueError("Output must not overwrite an input file")

    with identity.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        identity_header = read_header(reader, identity)
        identity_indices = [i for i, name in enumerate(identity_header) if name != KEY]
        identities = {
            key: [row[i] for i in identity_indices]
            for key, row in read_rows(reader, identity_header, identity)
        }

    temporary_path = None
    total = matched = 0
    try:
        with transactions.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.reader(stream)
            transaction_header = read_header(reader, transactions)
            identity_columns = [identity_header[i] for i in identity_indices]
            overlap = set(transaction_header) & set(identity_columns)
            if overlap:
                raise ValueError(f"Non-key columns exist in both files: {sorted(overlap)}")

            output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", newline="", encoding="utf-8", dir=output.parent,
                prefix=f".{output.name}.", suffix=".tmp", delete=False,
            ) as destination:
                temporary_path = Path(destination.name)
                writer = csv.writer(destination)
                writer.writerow(transaction_header + identity_columns)
                missing_identity = [""] * len(identity_columns)
                for key, row in read_rows(reader, transaction_header, transactions):
                    values = identities.get(key)
                    matched += values is not None
                    writer.writerow(row + (values if values is not None else missing_identity))
                    total += 1
            os.replace(temporary_path, output)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return total, matched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transactions", type=Path, default=DATA_DIR / "train_transaction.csv")
    parser.add_argument("--identity", type=Path, default=DATA_DIR / "train_identity.csv")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "train_joined.csv")
    args = parser.parse_args()
    total, matched = join_data(args.transactions, args.identity, args.output)
    print(f"Saved {total:,} transactions to {args.output}")
    print(f"With identity: {matched:,}; without identity: {total - matched:,}")


if __name__ == "__main__":
    main()

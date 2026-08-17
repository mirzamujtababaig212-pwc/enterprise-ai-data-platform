from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

WAREHOUSE = PROJECT_ROOT / "spark-warehouse"

METASTORE = PROJECT_ROOT / "metastore_db"

DATA_ROOT = PROJECT_ROOT / "data"


def print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def directory_size(path: Path) -> int:

    total = 0

    if not path.exists():
        return 0

    for item in path.rglob("*"):

        if item.is_file():

            try:
                total += item.stat().st_size
            except OSError:
                pass

    return total


def format_size(size: int) -> str:

    value = float(size)

    for unit in [
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    ]:

        if value < 1024:

            return f"{value:.2f} {unit}"

        value /= 1024

    return f"{value:.2f} PB"


def inspect_directory(
    name: str,
    path: Path,
) -> None:

    print_header(name)

    print(f"Path: {path}")
    print(f"Exists: {path.exists()}")
    print(f"Directory size: " f"{format_size(directory_size(path))}")

    if not path.exists():
        return

    print("\nTop-level entries:")

    for item in sorted(path.iterdir()):

        print(f"  {item.name}")


def main() -> None:

    print_header("ENTERPRISE AI PLATFORM STORAGE VERIFICATION")

    inspect_directory(
        "HIVE METASTORE",
        METASTORE,
    )

    inspect_directory(
        "SPARK WAREHOUSE",
        WAREHOUSE,
    )

    inspect_directory(
        "DATA ROOT",
        DATA_ROOT,
    )

    print_header("DELTA TABLE DISCOVERY")

    delta_logs = []

    search_roots = [
        WAREHOUSE,
        DATA_ROOT,
    ]

    for root in search_roots:

        if not root.exists():
            continue

        for path in root.rglob("_delta_log"):

            delta_logs.append(path)

    if not delta_logs:

        print("No _delta_log directories found.")

    else:

        for delta_log in sorted(delta_logs):

            print(f"DELTA: {delta_log}")

            json_files = list(delta_log.glob("*.json"))

            print(f"  transaction files: " f"{len(json_files)}")

    print_header("STORAGE VERIFICATION COMPLETE")


if __name__ == "__main__":
    main()

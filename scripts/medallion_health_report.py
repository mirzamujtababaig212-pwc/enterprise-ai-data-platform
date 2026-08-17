from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from common.spark.spark_builder import SparkSessionBuilder

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TABLES = {
    "bronze.vehicle_events": (PROJECT_ROOT / "data" / "bronze" / "vehicle_events"),
    "silver.vehicle_events": (PROJECT_ROOT / "data" / "silver" / "vehicle_events"),
    "gold.vehicle_metrics": (PROJECT_ROOT / "data" / "gold" / "vehicle_metrics"),
}


def table_health(
    spark,
    table_name: str,
    path: Path,
) -> dict:

    result = {
        "table": table_name,
        "path": str(path),
        "exists": path.exists(),
        "format": None,
        "row_count": None,
        "status": "FAILED",
    }

    if not path.exists():
        return result

    df = spark.read.format("delta").load(str(path))

    result["format"] = "delta"
    result["row_count"] = df.count()

    if result["row_count"] > 0:
        result["status"] = "HEALTHY"
    else:
        result["status"] = "EMPTY"

    return result


def main() -> None:

    spark = SparkSessionBuilder.build("MedallionHealthReport")

    try:

        report = {
            "platform": "enterprise_ai_platform",
            "report_type": "medallion_health",
            "generated_at": datetime.now(UTC).isoformat(),
            "spark_version": spark.version,
            "tables": [],
        }

        for table_name, path in TABLES.items():

            report["tables"].append(
                table_health(
                    spark,
                    table_name,
                    path,
                )
            )

        report["overall_status"] = (
            "HEALTHY"
            if all(table["status"] == "HEALTHY" for table in report["tables"])
            else "DEGRADED"
        )

        output_dir = PROJECT_ROOT / "reports" / "pipeline"

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = output_dir / "medallion_health_report.json"

        with output_file.open(
            "w",
            encoding="utf-8",
        ) as handle:

            json.dump(
                report,
                handle,
                indent=2,
            )

        print("=" * 80)
        print("MEDALLION HEALTH REPORT")
        print("=" * 80)

        print(
            json.dumps(
                report,
                indent=2,
            )
        )

        print()
        print(f"Report written to: {output_file}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()

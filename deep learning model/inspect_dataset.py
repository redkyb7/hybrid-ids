import json
import os

import numpy as np
import pandas as pd

import config


MAX_UNIQUE_VALUES_TO_PRINT = 100
SAMPLE_ROWS = 5


def find_likely_label_columns(
    columns: list[str],
) -> list[str]:
    """
    Return columns whose names suggest a target/label field.
    """

    keywords = [
        "label",
        "attack",
        "class",
        "category",
        "target",
        "type",
        "traffic",
    ]

    candidates = []

    for column in columns:
        lower_name = str(column).lower()

        if any(
            keyword in lower_name
            for keyword in keywords
        ):
            candidates.append(
                str(column)
            )

    return candidates


def safe_value_counts(
    series: pd.Series,
) -> dict[str, int]:
    """
    Convert value counts into JSON-safe string keys.
    """

    counts = series.value_counts(
        dropna=False
    )

    return {
        str(index): int(count)
        for index, count in counts.items()
    }


def inspect_dataset():
    print("=" * 70)
    print("CIC COLLECTION PARQUET DATASET INSPECTION")
    print("=" * 70)

    data_path = config.DATA_PATH

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "Dataset file not found:\n"
            f"{data_path}"
        )

    print(
        f"\nDataset path:\n{data_path}"
    )

    print(
        "\n[1/7] Reading Parquet schema..."
    )

    # Reading the full dataframe because we need a complete,
    # reliable inspection of data types and labels.
    df = pd.read_parquet(
        data_path
    )

    row_count, column_count = df.shape

    print(
        f"      Rows    : {row_count:,}"
    )

    print(
        f"      Columns : {column_count:,}"
    )

    print(
        "\n[2/7] Listing all columns..."
    )

    for index, column in enumerate(
        df.columns,
        start=1,
    ):
        print(
            f"{index:02d}. "
            f"{column} "
            f"[{df[column].dtype}]"
        )

    column_names = [
        str(column)
        for column in df.columns
    ]

    print(
        "\n[3/7] Finding likely label columns..."
    )

    likely_label_columns = find_likely_label_columns(
        column_names
    )

    if likely_label_columns:
        for column in likely_label_columns:
            print(
                f"      Candidate: {column}"
            )
    else:
        print(
            "      No obvious label columns found "
            "from column names."
        )

    print(
        "\n[4/7] Inspecting non-numeric columns..."
    )

    non_numeric_columns = df.select_dtypes(
        exclude=[
            np.number,
            "bool",
        ]
    ).columns.tolist()

    if non_numeric_columns:
        for column in non_numeric_columns:
            unique_count = int(
                df[column].nunique(
                    dropna=False
                )
            )

            print(
                f"\n      Column: {column}"
            )

            print(
                f"      Dtype : {df[column].dtype}"
            )

            print(
                f"      Unique values: "
                f"{unique_count:,}"
            )

            if unique_count <= MAX_UNIQUE_VALUES_TO_PRINT:
                print(
                    "      Value counts:"
                )

                counts = df[
                    column
                ].value_counts(
                    dropna=False
                )

                for value, count in counts.items():
                    print(
                        f"        "
                        f"{str(value):40s} "
                        f"{count:>12,}"
                    )
            else:
                print(
                    "      Too many unique values to "
                    "print fully."
                )

                print(
                    "      Top 20 values:"
                )

                counts = df[
                    column
                ].value_counts(
                    dropna=False
                ).head(20)

                for value, count in counts.items():
                    print(
                        f"        "
                        f"{str(value):40s} "
                        f"{count:>12,}"
                    )
    else:
        print(
            "      All columns are numeric or boolean."
        )

    print(
        "\n[5/7] Inspecting likely label candidates..."
    )

    label_reports = {}

    for column in likely_label_columns:
        series = df[column]

        unique_count = int(
            series.nunique(
                dropna=False
            )
        )

        missing_count = int(
            series.isna().sum()
        )

        print(
            f"\n      Label candidate: {column}"
        )

        print(
            f"      Dtype          : "
            f"{series.dtype}"
        )

        print(
            f"      Unique values  : "
            f"{unique_count:,}"
        )

        print(
            f"      Missing values : "
            f"{missing_count:,}"
        )

        counts = series.value_counts(
            dropna=False
        )

        print(
            "      Value counts:"
        )

        limit = min(
            len(counts),
            MAX_UNIQUE_VALUES_TO_PRINT,
        )

        for value, count in counts.head(
            limit
        ).items():
            percentage = (
                count / len(series)
            ) * 100

            print(
                f"        "
                f"{str(value):40s} "
                f"{count:>12,} "
                f"({percentage:6.2f}%)"
            )

        if len(counts) > limit:
            print(
                f"        ... "
                f"{len(counts) - limit:,} "
                f"more values omitted"
            )

        label_reports[column] = {
            "dtype": str(
                series.dtype
            ),
            "unique_values": unique_count,
            "missing_values": missing_count,
            "value_counts": safe_value_counts(
                series
            ),
        }

    print(
        "\n[6/7] Checking nulls and numeric infinities..."
    )

    null_counts = df.isna().sum()

    columns_with_nulls = {
        str(column): int(count)
        for column, count in null_counts.items()
        if count > 0
    }

    if columns_with_nulls:
        print(
            "      Columns containing null values:"
        )

        for column, count in columns_with_nulls.items():
            print(
                f"        {column:40s} "
                f"{count:>12,}"
            )
    else:
        print(
            "      No null values found."
        )

    numeric_columns = df.select_dtypes(
        include=[
            np.number,
        ]
    ).columns.tolist()

    infinity_counts = {}

    for column in numeric_columns:
        values = df[
            column
        ].to_numpy(
            copy=False
        )

        count = int(
            np.isinf(values).sum()
        )

        if count > 0:
            infinity_counts[
                str(column)
            ] = count

    if infinity_counts:
        print(
            "\n      Columns containing "
            "positive/negative infinity:"
        )

        for column, count in infinity_counts.items():
            print(
                f"        {column:40s} "
                f"{count:>12,}"
            )
    else:
        print(
            "\n      No infinite numeric values found."
        )

    print(
        "\n[7/7] Printing sample rows..."
    )

    print(
        df.head(
            SAMPLE_ROWS
        ).to_string()
    )

    report = {
        "dataset_path": os.path.abspath(
            data_path
        ),
        "shape": {
            "rows": int(row_count),
            "columns": int(column_count),
        },
        "columns": [
            {
                "name": str(column),
                "dtype": str(df[column].dtype),
                "null_count": int(
                    df[column].isna().sum()
                ),
                "unique_count": int(
                    df[column].nunique(
                        dropna=False
                    )
                ),
            }
            for column in df.columns
        ],
        "likely_label_columns": likely_label_columns,
        "label_reports": label_reports,
        "non_numeric_columns": [
            str(column)
            for column in non_numeric_columns
        ],
        "columns_with_nulls": columns_with_nulls,
        "columns_with_infinities": infinity_counts,
    }

    report_path = os.path.join(
        config.BASE_DIR,
        "dataset_inspection_report.json",
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)

    print(
        f"\nJSON report saved to:\n"
        f"{report_path}"
    )

    print(
        "\nNext steps:"
    )

    print(
        "1. Find the correct target label column."
    )

    print(
        "2. Set LABEL_COLUMN in config.py to that exact name."
    )

    print(
        "3. Copy the label value counts and provide them."
    )

    print(
        "4. We can then build a correct LABEL_MAP and "
        "feature-exclusion list."
    )


if __name__ == "__main__":
    inspect_dataset()
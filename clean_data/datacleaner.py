"""
SentinelFlow - NSL-KDD Data Cleaning & Preprocessing Pipeline
============================================================
A modular, production-ready preprocessor for network telemetry datasets (NSL-KDD).
Supports leak-free train/test scaling, persistent one-hot categorical encoding,
binary flag preservation, and real-time inference serialization (joblib).
"""

import argparse
import os
from io import StringIO
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class KDDDataPipeline:
    """
    Modular Preprocessor Pipeline for NSL-KDD Telemetry Data.
    Ensures zero data leakage between training and testing / real-time deployment.
    """

    CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
    BINARY_COLS = [
        "land",
        "logged_in",
        "root_shell",
        "su_attempted",
        "is_host_login",
        "is_guest_login",
    ]
    TARGET_COL = "class"

    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.continuous_cols = []
        self.feature_names = []
        self.is_fitted = False

    @staticmethod
    def load_arff(file_path: str) -> pd.DataFrame:
        """
        Parses ARFF files into a pandas DataFrame with header and comment handling.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"ARFF file not found at: {file_path}")

        attributes = []
        data_lines = []
        is_data_section = False

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line_clean = line.strip()
                if not line_clean or line_clean.startswith("%"):
                    continue

                if line_clean.lower().startswith("@data"):
                    is_data_section = True
                    continue

                if not is_data_section:
                    if line_clean.lower().startswith("@attribute"):
                        parts = line_clean.split()
                        col_name = parts[1].strip("'\"")
                        attributes.append(col_name)
                else:
                    data_lines.append(line_clean)

        # Load into DataFrame
        df = pd.read_csv(
            StringIO("\n".join(data_lines)), names=attributes, header=None
        )

        # Sanitize whitespace / quote artifacts in string columns
        str_cols = df.select_dtypes(include=["object"]).columns
        for col in str_cols:
            df[col] = df[col].astype(str).str.strip().str.strip("'\"")

        return df

    def _determine_columns(self, df: pd.DataFrame):
        """Identifies continuous numeric columns excluding target, categorical, and binary flags."""
        exclude = set(self.CATEGORICAL_COLS + self.BINARY_COLS + [self.TARGET_COL])
        self.continuous_cols = [col for col in df.columns if col not in exclude]

    def fit(self, df: pd.DataFrame):
        """
        Learns scaling parameters on continuous features and categories on categorical features.
        MUST ONLY be called on Training Data to prevent Data Leakage.
        """
        df_clean = df.copy()
        self._determine_columns(df_clean)

        # 1. Fit StandardScaler on Continuous features
        self.scaler.fit(df_clean[self.continuous_cols])

        # 2. Fit OneHotEncoder on Categorical features
        self.encoder.fit(df_clean[self.CATEGORICAL_COLS])
        encoded_cat_names = list(self.encoder.get_feature_names_out(self.CATEGORICAL_COLS))

        # 3. Store the deterministic feature column layout
        self.feature_names = self.continuous_cols + self.BINARY_COLS + encoded_cat_names
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame):
        """
        Transforms test data or real-time packets using the learned parameters.
        Returns:
            X (pd.DataFrame): Normalized, one-hot encoded feature matrix.
            y (pd.Series or None): Target class vector (0=normal, 1=anomaly) if present.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before transforming data. Call .fit() or load a pre-fitted pipeline.")

        df_input = df.copy()

        # Sanitize string columns
        str_cols = df_input.select_dtypes(include=["object"]).columns
        for col in str_cols:
            df_input[col] = df_input[col].astype(str).str.strip().str.strip("'\"")

        # 1. Transform Continuous Features
        scaled_continuous = self.scaler.transform(df_input[self.continuous_cols])
        df_cont = pd.DataFrame(
            scaled_continuous,
            columns=self.continuous_cols,
            index=df_input.index
        )

        # 2. Extract Binary Features (force 0/1 integer)
        df_bin = pd.DataFrame(index=df_input.index)
        for col in self.BINARY_COLS:
            if col in df_input.columns:
                df_bin[col] = pd.to_numeric(df_input[col], errors="coerce").fillna(0).astype(int)
            else:
                df_bin[col] = 0

        # 3. Transform Categorical Features
        encoded_cat = self.encoder.transform(df_input[self.CATEGORICAL_COLS])
        encoded_cat_names = list(self.encoder.get_feature_names_out(self.CATEGORICAL_COLS))
        df_cat = pd.DataFrame(
            encoded_cat.astype(int),
            columns=encoded_cat_names,
            index=df_input.index
        )

        # 4. Concatenate into Final Feature Matrix
        X = pd.concat([df_cont, df_bin, df_cat], axis=1)

        # Ensure exact column ordering
        X = X[self.feature_names]

        # 5. Process Target Label if present
        y = None
        if self.TARGET_COL in df_input.columns:
            # Map 'normal' -> 0, all attacks/anomalies -> 1
            y = df_input[self.TARGET_COL].apply(
                lambda val: 0 if str(val).lower() == "normal" else 1
            )

        return X, y

    def fit_transform(self, df: pd.DataFrame):
        """Fit on training data and transform in one step."""
        return self.fit(df).transform(df)

    def save(self, filepath: str):
        """Serializes the fitted pipeline to a .joblib file for model inference."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted pipeline.")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"[+] Pipeline artifact saved successfully: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "KDDDataPipeline":
        """Loads a pre-fitted pipeline from a .joblib file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Pipeline artifact not found: {filepath}")
        pipeline = joblib.load(filepath)
        print(f"[+] Loaded pre-fitted pipeline with {len(pipeline.feature_names)} features.")
        return pipeline


# ==============================================================================
# CLI EXECUTION & DEMO
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Clean and preprocess NSL-KDD datasets.")
    parser.add_argument("--input", type=str, default="KDDTest+.arff", help="Path to input .arff dataset file")
    parser.add_argument("--output", type=str, default="KDDTest_cleaned.csv", help="Path to save cleaned CSV")
    parser.add_argument("--pipeline-out", type=str, default="kdd_pipeline.joblib", help="Path to save fitted pipeline")
    parser.add_argument("--fit", action="store_true", default=True, help="Fit pipeline on the input dataset")
    args = parser.parse_args()

    print("=" * 60)
    print("SENTINELFLOW IDS - DATA CLEANING PIPELINE")
    print("=" * 60)

    # 1. Load raw dataset
    print(f"[*] Loading raw ARFF dataset: {args.input}")
    raw_df = KDDDataPipeline.load_arff(args.input)
    print(f"[+] Raw data shape: {raw_df.shape}")

    # 2. Deduplication check
    initial_len = len(raw_df)
    raw_df = raw_df.drop_duplicates()
    removed = initial_len - len(raw_df)
    if removed > 0:
        print(f"[+] Removed {removed} duplicate rows.")

    # 3. Fit and transform pipeline
    pipeline = KDDDataPipeline()
    print("[*] Fitting preprocessor (StandardScaler + OneHotEncoder)...")
    X, y = pipeline.fit_transform(raw_df)

    # 4. Save cleaned CSV dataset
    cleaned_df = X.copy()
    if y is not None:
        cleaned_df["class"] = y

    cleaned_df.to_csv(args.output, index=False)
    print(f"[+] Cleaned dataset exported: {args.output}")

    # 5. Save reusable pipeline artifact for backend inference
    pipeline.save(args.pipeline_out)

    # 6. Display Dataset Summary Metrics
    print("\n" + "=" * 60)
    print("DATASET & FEATURE SUMMARY")
    print("=" * 60)
    print(f"Total Processed Samples : {len(cleaned_df):,}")
    print(f"Feature Dimension (X)   : {X.shape[1]} features")
    print(f"  |-- Continuous Features: {len(pipeline.continuous_cols)}")
    print(f"  |-- Binary Flag Features: {len(pipeline.BINARY_COLS)}")
    print(f"  \\-- One-Hot Encoded    : {X.shape[1] - len(pipeline.continuous_cols) - len(pipeline.BINARY_COLS)}")
    print(f"Missing (NaN) Values    : {cleaned_df.isnull().sum().sum()}")

    if y is not None:
        normal_cnt = (y == 0).sum()
        anomaly_cnt = (y == 1).sum()
        print("\nClass Distribution:")
        print(f"  |-- Normal Traffic (0) : {normal_cnt:,} ({normal_cnt/len(y)*100:.2f}%)")
        print(f"  \\-- Anomaly/Attack (1) : {anomaly_cnt:,} ({anomaly_cnt/len(y)*100:.2f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
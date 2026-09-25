# Stage 1 ML Colab run record

Source: console output pasted by the user after the Colab run. This record preserves the reported values. The original `models.zip` contained the selected model and feature list but no evaluation file; the archive was removed after extraction. Metrics were printed to four decimal places, so more precise values cannot be recovered from this record.

- Dataset: `/content/hybrid-ids/clean_data/cic-collection.parquet`, full 9,167,581 rows and 59 columns.
- Raw binary class shares: Benign 0.7838697034692139; Attack 0.21613029653078603.
- Correlation pruning: 12 features dropped; 20 retained.
- Deduplication before splitting: 7,686,543 rows removed; 1,481,038 retained.
- Split: train 1,036,726; validation 222,156; test 222,156.
- Test support: Benign 192,600; Attack 29,556.

| Reported test measure | Random Forest | XGBoost |
| --- | ---: | ---: |
| Accuracy | 0.9770 | 0.9810 |
| Attack precision | 0.9371 | 0.9966 |
| Attack recall | 0.8870 | 0.8604 |
| Attack F1 | 0.9113 | 0.9235 |
| Benign precision | 0.9828 | 0.9790 |
| Benign recall | 0.9909 | 0.9996 |
| Benign F1 | 0.9868 | 0.9892 |
| Macro precision | 0.9599 | 0.9878 |
| Macro recall | 0.9389 | 0.9300 |
| Macro F1 | 0.9491 | 0.9564 |
| Weighted F1 | 0.9768 | 0.9804 |
| Reported prediction time per flow | 0.00187 ms | 0.00052 ms |

The script reported `SELECTED STAGE 1 MODEL: XGBoost (F1: 0.9564)` and saved `backend/models/stage1_binary_filter.joblib` and `backend/models/stage1_feature_list.joblib` in Colab. Its selection used the test set. The output did not include metrics at the live IDS's 0.10 attack threshold or per-attack-category recall.

Those are the paths printed in the Colab run. The deployed XGBoost files in
this repository are the later uploaded export under `updated_models/five_attack/ml`.

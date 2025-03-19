# config.py


config = {
    # Specify the backtest evaluation directory
    "backtest": {
        "dir": ('/workspace/projects/fx_classify_demo/training/iterations/04_03_2025_14_49_49'),
        "model_weights": "epoch10_trainLoss_0.9859_trainAcc_0.4527_valLoss_1.0054_valAcc_0.4684.h5",
        "create_model_module_src": "fx_classify_custom_train_funcs.py",
        "model_module": "create_model",
        "model_config_src": "config.py",
        "config_dict_name": "config",
        "model_args": "model_args",
        "feature_categories": "feature_categories",
        "save_pred_dir": ('/workspace/projects/fx_classify_demo/evaluation/predictions'),
        "primary_keys": ("date", "pair")
        },
    "explain": {
        "dir": ('/workspace/projects/fx_classify_demo/training/iterations/04_03_2025_14_49_49'),
        "model_weights": "epoch10_trainLoss_0.9859_trainAcc_0.4527_valLoss_1.0054_valAcc_0.4684.h5",
        "create_model_module_src": "fx_classify_custom_train_funcs.py",
        "model_module": "create_model",
        "model_config_src": "config.py",
        "config_dict_name": "config",
        "model_args": "model_args",
        "file_num": 4, # Specify which group in prepped_data to load
        "class_names": [0,1,2],
        "contains_categorical_features": True,
        "categorical_feature_id": "pair_",
        "sample_num": 16, # Specify which entry index in the file to explain,
        "id_cols": ["date", "pair"],
        "prediction_dir": ('/workspace/projects/fx_classify_demo/evaluation/predictions/04_03_2025_14_49_49_epoch10.db')
        },
    "metrics": {
        "db": ('/workspace/projects/fx_classify_demo/evaluation/predictions/04_03_2025_14_49_49_epoch10.db'),
        "query": """
        SELECT * FROM predictions
        """,
        "metric_categories": {
            "split_col": "split",
            "asset_col": 'pair',
            "target_col": 'predicted_category',
            },
        "y_true": "target",
        "y_pred": "predicted_category",
        "y_conf": "confidence",
        "metrics": ['accuracy', 'precision', 'recall', 'f1_score', 'log_loss', 'roc_auc'],
        },
    "calibration": {
        "db": ('/workspace/projects/fx_classify_demo/evaluation/predictions/04_03_2025_14_49_49_epoch10.db'),
        "query": """
        SELECT * FROM predictions
        """,
        "y_true": "target",
        "y_pred": "predicted_category",
        "y_conf": "confidence",
        "cal_categories": {
            "split_col": "split",
            "asset_col": 'pair',
            "target_col": 'predicted_category',
            }
        },
    "candidates": {
        "db": ('/workspace/projects/fx_classify_demo/evaluation/predictions/04_03_2025_14_49_49_epoch10.db'),
        # ***GENERALLY ENSURE THE DATASET IS FILTERED BY VAL AND TEST SPLITS***
        "query": """
        SELECT * FROM predictions
        WHERE split IN ('val', 'test') 
        """,
        "y_true": "target",
        "y_pred": "predicted_category",
        "y_conf": "confidence",
        "class_filter": {
            "column_name": "predicted_category",
            "classes": [1,2]
            },
        "asset_col": "pair",
        # The custom function must use df as an arguement
        "custom_func": True,
        "custom_func_path": ('/workspace/projects/fx_classify_demo/evaluation/fx_classify_custom_eval_funcs.py'),
        "custom_func_name": "calculate_running_profit"
        }
    }
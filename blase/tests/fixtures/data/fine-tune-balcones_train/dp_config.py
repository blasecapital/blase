# config.py


config = {
    # Arguements for load_data
    "load_mode": "full", # Options: "full", "batch", or "pair"
    "collect_pairs": "query", # Options: query or list
    "pair_query": {"BASE_DATABASE": """
                   SELECT DISTINCT(pair) FROM test_hourly_exchange_rate 
                   """}, # Key: path to test_raw_data database, Value: query
    "pair_list": ["AUDUSD"], # Explicitly list all the pairs to ETL
    "df_batch": 4, # Only if load_mode is batch
    
    "group_by": "pair", # Options: "pair" or None
    
    "source_mode": "base", # Options: base, feature, or target
    
    "base_source": "BASE_DATABASE",
    "base_query": """
        SELECT * from test_hourly_exchange_rate
        WHERE pair IN ({placeholders})
        """,
    "feature_source": "FEATURE_DATABASE",
    "feature_query": """
        SELECT * FROM test_hourly_exchange_rate
        WHERE pair IN ({placeholders})
        """,
    "target_source": "TARGET_DATABASE",
    "target_query": """
        SELECT * FROM test_hourly_exchange_rate
        WHERE pair IN ({placeholders})
        """,

    # Primary key for feature and target tables
    "primary_key": ['date', 'pair'],  
  
    # Arguements for create_features
    "feature_save_mode": "single", # Options: single, batch, full
    "feature_batch": 4,
    "feature_modules_path": ('/workspace/projects/fx_classify_demo/data_preparation/fx_classify_feature_engineering.py'),
    "main_feature_module": 'features',
    "feature_storage_map": 'storage_map',

    # Arguements for create_targets
    "target_save_mode": "single", # Options: single, batch, full
    "target_batch": 4,
    "target_modules_path": ('/workspace/projects/fx_classify_demo/data_preparation/fx_classify_target_engineering.py'),
    "main_target_module": 'targets',
    "target_storage_map": 'storage_map'
    }
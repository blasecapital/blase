# config.py


config = {
    # Arguements for load_training_data
    # Name the data key (name of feature set or target set), write the .env
    # reference name for the source db_path and the associated query
    "source_query": {
        "hourly_features" : ("FEATURE_DATABASE", 
         """
         SELECT * FROM test_hourly_feature_data
         ORDER BY pair, date
         """),
         "tech_features": ("FEATURE_DATABASE",
         """ 
         SELECT * FROM test_technical_feature_data
         """
             ),
         "targets" : ("TARGET_DATABASE",
         """
         SELECT * FROM targets
         """)},
         
    "project_directory": ('/workspace/projects/fx_classify_demo/training'),
    
    # Use accross modules
    "primary_key": ['date', 'pair'],
    
    # Args for raw data processing
    # Path to low level functionality to pass to ProcessRawData
    "data_processing_modules_path": ('/workspace/projects/fx_classify_demo/training/fx_classify_process_raw_data.py'),
    # Dictionary for collecting bad data primary keys, the dict key is a 
    # metadata name and its item is a set containing the env key to the 
    # database, a query for specifying the table and data to filter, and 
    # the name of the filter function
    "clean_functions": {
        "hourly_features": ("FEATURE_DATABASE", 
         """
         SELECT * FROM test_hourly_feature_data
         """,
         'filter_hourly'),
        
        "targets": ("TARGET_DATABASE",
        """
        SELECT * FROM targets
        """,
        'filter_targets')
        },
    "bad_keys_path": ('/workspace/projects/fx_classify_demo/training/bad_keys'),
    
    # Dictionary for cleaning and saving
    "align": {
        "hourly_features": ("FEATURE_DATABASE",
            """
            SELECT * FROM test_hourly_feature_data
            """
            ),
        "targets": ("TARGET_DATABASE",
        """
        SELECT * FROM targets
        """)
        },
    
    
    # Dicts for PrepData engineer function
    # Keys: meta
    #   Items:
    #       1) .env database path reference
    #       2) Query
    #       3) function name
    #       4) save table name
    "feature_engineering": {
        "hourly_features": (
            "CLEAN_FEATURE_DATABASE",
            """
            SELECT date, pair FROM test_technical_feature_data
            """,
            'feature_engineering',
            'test_engineered_feature_data'
            )
        },
    # Only use if creating new targets in a new table
    # Not suited for replacing existing targets
    "target_engineering": {
        "target_engineer": (
            "CLEAN_TARGET_DATABASE",
            """
            SELECT * FROM targets
            """,
            'target_engineering',
            'targets'
            )
        },
    
    "scaler_save_dir": ('/workspace/projects/fx_classify_demo/training/scalers'),
    
    # Specify the db_path to scale features for
    # Keys: meta
    #   Items:
    #       1) .env database path reference
    #       2) List of table names to scale
    "feature_scaling": {
        "test_features" : (
            "CLEAN_FEATURE_DATABASE",
            ['test_technical_feature_data']
            )
        },
    
    "prepped_data_dir": ('/workspace/projects/fx_classify_demo/training/prepped_data'),
    
    # Specify the db_path to scale features for
    # Keys: meta
    #   Items:
    #       1) .env database path reference
    #       2) Dict of table names to specify if they need to be scaled or reshaped
    #       3) Path to save scale files
    "prep_and_save": {
        "test_features": (
            "CLEAN_FEATURE_DATABASE",
            {
                'test_hourly_feature_data': {
                    'scaler': True, 
                    'keep_primary_key': False
                    },
                'test_technical_feature_data': {
                    'scaler': True,
                    'keep_primary_key': False
                    },
                'test_engineered_feature_data': {
                    'scaler': False,
                    'keep_primary_key': False                    
                    }
                }
            ),
        "test_targets": (
            "CLEAN_TARGET_DATABASE",
            {
                'targets': {
                    'scaler': False,
                    'weights_dict': 'target',
                    'keep_primary_key': True
                    }
                }
            )
        },
    "weight_dict_save_path": ('/workspace/projects/fx_classify_demo/training/weights_dict/target_weights_dict.json'), 
            
    # Args for training
    # Reshape (number of windows (48 hours of ohlc=48), size of windows (ohlc=4))
    "feature_categories": 
        {
            'test_hourly_feature_data': {
                'reshape': True,
                'shape': (12, 4)
                },
            'test_technical_feature_data': {
                'reshape': False,
                'shape': None
                },
            'test_engineered_feature_data': {
                'reshape': False,
                'shape': None
                }
        },
    "model_modules_path": ('/workspace/projects/fx_classify_demo/training/fx_classify_custom_train_funcs.py'),
    "model_function": 'create_model',
    "callback_function": 'AggregateCallbacks',
    "model_args": {
        'n_hours_back_hourly': 48 // 4, # len(features) // len(window)
        'n_ohlc_features': 4, # window
        'l2_strength': 0.00, 
        'dropout_rate': 0.0,
        'n_tech_features': 3,
        'n_eng_features': 7, 
        'activation': 'relu', 
        'n_targets': 3, 
        'output_activation': 'softmax',
        'initial_bias': True
        },
    
    # Use for both initial_bias and class_weights
    "weight_dict_path": ('/workspace/projects/fx_classify_demo/training/weights_dict/target_weights_dict.json'), 
    "data_dir": ('/workspace/projects/fx_classify_demo/training/prepped_data'),    
    "custom_loss": {"custom_loss_path": ('/workspace/projects/fx_classify_demo/training/fx_classify_custom_train_funcs.py'),
                    "module_name": "custom_loss"},
    #"custom_loss": {},
    "use_weight_dict": True,
    # The layer is the key and the loss is the item
    "loss": {"output_layer": "sparse_categorical_crossentropy"},
    "metrics": ["accuracy"],
    "optimizer": {"type": "adam", "learning_rate": 0.001, "clipvalue": 1.0},
    "epochs": 100,
    "batch_size": 64,
    "iteration_dir": ('/workspace/projects/fx_classify_demo/training/iterations'),
    # Specify the file paths of the iteration config.py and model architecture file
    "requirements_paths": {
        "config": ('/workspace/projects/fx_classify_demo/training/config.py'),
        "model": ('/workspace/projects/fx_classify_demo/training/fx_classify_custom_train_funcs.py')
        }
    }
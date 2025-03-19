# fx_classify_training.py


from training.components import CleanRawData, PrepData, Train


def training(clean=False, prep=False, train=True):
    """
    Orchestrate the entire process of loading and preprocessing the training data,
    defining model architectures and custom functions, and saving results
    and supporting files for model evaluation.
    
    Cleaning Process:
        1) Inspect the features and targets to inform filter functions
            a. Begin with the raw feature and target database(s)
        2) Write the filter functions according to inspection observations
        3) Run the clean.clean() module to collect bad primary keys
        4) Run clean.align() module to save filtered data to clean database
        5) Revise config's source_query's database reference to point to new,
           clean database
        6) Loop starting at step 1 if necessary
            a. For step 1, change the database reference to the new, clean 
               database to observe the previous filter process's impact
            b. Ensure filter functions are holistic each time, the process
               is designed to filter the entire, raw database
               
        Key args:
            - Source: /<main path>/projects/<iteration folder>/training/config.py
            - Keys used: source_query, primary_key, data_processing_modules_path,
              clean_functions, bad_keys_path, align
            - Configure inspect, filter_keys, align flags according to your needs
              each run
    
    Prep Process:
        1) Create or modify features or using prep.engineer()
            - write this in the script specified in config's 'data_processing_modules_path'
              and fill out the 'feature_engineering' and/or 'target_engineering' key in config
        2) Fit and save scalers to your desired feature sets using prep.scale()
            - specify the scaler save path in config's 'scaler_save_dir' and
              the table data to be scaled in 'feature_scaling'
        3) Split and save features and targets as .tfrecord files using prep.save_batches()
            - save your files to the config's 'prepped_data_dir' and include
              the location, table names, and relevant functions to apply to
              the data during preppring with 'prep_and_save'
        4) Inspect the .tfrecords with prep.validate_record()
    
        Key args:
            - Source: /<main path>/projects/<iteration folder>/training/config.py
            - Keys used: data_processing_modules_path, feature_engineering, 
              target_engineering, scaler_save_dir, feature_scaling, prepped_data_dir,
              prep_and_save
            - Configure engineer, scale, prep_and_save, and validate_data flags 
              according to your needs each run
            
    Train Process:
        1) Train your neural network using train.train_models()
            - Create a model_modules script to include custom_loss, callbacks,
              and model architecture,
            - Specify the model_modules path in config's 'model_modules_path'
            - Reference your model architecture funtion's name with 'model_function'
            - Reference your callbacks class name with 'callback_function'
            - Specify your model archtitecture funtion's arguements with 'model_args'
            - Specify the path to stored class weights with 'weight_dict_path'
            - Specify where the prepped data is stored with 'data_dir'
            - If using a custom loss, fill in 'custom_loss', if not leave empty {}
            - If using weights dict set 'use_weight_dict' to True, otherwise False
            - Specify training arguements with 'loss', 'metrics', 'optimizer',
              'epochs', 'batch_size'
            - Specify where to store the models weights and associated training
              data with 'iteration_dir'
            - Specify the paths to the test config and model modules scripts using
              'requirements_paths' to store training dependencies for replication
              
        Key args:
            - Source: /<main path>/projects/<iteration folder>/training/config.py
            - Keys used: model_modules_path, model_function, callback_function,
              model_args, weight_dict_path, data_dir, custom_loss, use_weight_dict,
              loss, metrics, optimizer, epochs, batch_size, iteration_dir,
              requirements_paths
        
    Take care to update the config.py file every time you run the 
    cleaning modules.    
    """
    if clean:
        # Initialize the preprocess object
        clean = CleanRawData()
        
        # Which clean processes do you want to run
        inspect = False
        filter_keys = False
        align = True
                
        if inspect:        
            clean.inspect_data(
                data_keys=['tech_features'], # refers to the config's source_query keys
                describe_features=True,
                describe_targets=False,
                target_type='cat',
                plot_features=True, 
                # using 'rate' is the only time the query should use ORDER BY pair
                plot_mode='stat', 
                plot_skip=1
                )
            
        if filter_keys:
            clean.clean(data_keys=[])
            
        if align:
            clean.align()
        
    if prep:
        # Initialize the data preparation object
        prep = PrepData()
        
        # Which prep processes do you want to run
        engineer=False
        scale=False
        prep_and_save=False
        validate_data=True
        
        if engineer:
            prep.engineer(mode='all')
        if scale:
            prep.scale()
        if prep_and_save:
            prep.save_batches()
        if validate_data:
            prep.validate_record()
        
    if train:
        train = Train()
        train.train_models()
        
    
if __name__ == "__main__":
    training(clean=False, prep=False, train=True)
    
# fx_classify_data_preparation.py


from data_preparation.components import CreateFeatures, CreateTargets


def data_preparation(features=True, targets=True):
    """
    Orchestrate the entire process of loading raw data, engineering features, 
    and storing datasets for model training.
    
    Prerequisites:
        - load_data requires data be stored in a database
        
    Returns:
        None: Data is stored in a database
        
    Quick start:
        1) Update /config.env which stores database paths
        2) Update /tests/data_preparation/config.py which stores arguements           
    """
    
    if features:
        # Load base feature data, create features, and store them
        cf = CreateFeatures()
        cf.calculate_and_store_features()
    
    if targets:
        # Load base target data, create targets, and store them
        ct = CreateTargets()
        ct.calculate_and_store_targets()
    

if __name__ == "__main__":
    data_preparation()
    
# fx_classify_evaluation.py


from evaluation.components import Eval


def evaluation():
    """
    predict_and_store: 
        Load model weights, run predictions on specified training split(s), 
        and store results. This is a prerequisite to all subsequent functions.
        
    explain:
        Run LIME to display the most important features pertinent to the choosen
        observation's prediction.
    
    report_metrics:
        Prints accuracy, precision, recall, f1 score, log-loss, and roc-auc score 
        for the entire dataset, each dataset split (train, val, test), and further
        by asset and target category.
    
    report_calibration:
        Bins prediction accuracy by confidence and plots the calibration in 
        reference to a benchmark (y=x where y is accuracy and x is confidence).
        Plots are grouped just like report_metrics.
    
    report_candidates:
        Filters dataset by relevant predicted categories (in this case, 'buy'
        and 'sell'), sorts filtered predictions by confidence, calculates a
        rolling accuracy, and uses the accuracy_threshold and volume arguements
        to print assets which have the desired accuracy and volume. Useful
        for informing custom business-impact functions, in this case a running
        profit model.
    
    Arg's are stored in /training/config.py. 
    """
    e = Eval()
    
    pred=False
    explain=False
    metrics=False
    cal=False
    candidates=True
    
    if pred:
        e.predict_and_store(mode="full")
        
    if explain:
        e.explain()
    
    if metrics:
        e.report_metrics(display_mode='convert')
        
    if cal:    
        e.report_calibration(mode='convert')
        
    if candidates:
        e.report_candidates(
            mode='convert',
            accuracy_threshold=.35,
            volume=10)
    
if __name__ == "__main__":
    evaluation()
    
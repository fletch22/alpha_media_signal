MLP between 
    
    Small Survey: The sample size is too small. Create model for each day of the date span.
    Exlude random days from each model. Use those random days as hold out. Test. 
    
    predict:
        that open price is higher than previous trading day close price 
    purchase: if signal is positive, buy at close 
    sell:
        when target_frac_roi = .006, sell otherwise sell at close.
        results:  
            default: roi 0.00274
            where stocks in first 100 alphabetical: roi: 0.005 
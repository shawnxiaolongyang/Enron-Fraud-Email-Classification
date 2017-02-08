#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []
    

    ### your code goes here
    residual = predictions - net_worths
    
    
    cleaned_data["age"] = ages
    cleaned_data["net_worth"] = net_worths
    cleaned_data["error"] = residual
                
            
        
            
    
    return cleaned_data


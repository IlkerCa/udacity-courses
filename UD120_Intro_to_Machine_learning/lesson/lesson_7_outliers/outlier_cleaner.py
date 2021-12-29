#!/usr/bin/python


from numpy.core.fromnumeric import sort


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    error = (net_worths - predictions)**2
    zipped = zip(ages, net_worths, error)
    cleaned = sorted(zipped, key = lambda x: x[2])
    cleaned_data = cleaned[:int(len(cleaned)*0.9)]

    ### your code goes here

    
    return cleaned_data


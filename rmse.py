from math import sqrt
import numpy as np
def rmse(x, y):
    '''
    It calculates de root mean square error (RMSE).

    Input:
        * x -> list or array of measured/actual data. For example,
        in terms of et0, it would be FAO56-PM et0.

        * y -> list or array of predicted values. For example, in
        terms of et0, it would be the predicted et0 by a neural
        network model, etc.

    Output:
        * rmse -> a float with the root mean squared error value
    '''
    # check the lengths are the same 
    assert len(x)==len(y)
    # convert the inputs in numpy array
    x = np.array(x)
    y = np.array(y)
    # rmse calculus
    delta = (y - x)**2
    rmse = sqrt(sum(delta)*1.0/len(x))
    return rmse
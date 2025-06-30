import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler


def log_scale_transform(X, Y, site_number, dates, satellites):
    shift_value_X = np.abs(np.min(X)) + 1  # Shift X to ensure positive values
    shift_value_Y = np.abs(np.min(Y)) + 1  # Shift Y similarly

    X_shifted = X + shift_value_X
    Y_shifted = Y + shift_value_Y

    X_log = np.log(X_shifted)
    Y_log = np.log(Y_shifted)

    s1 = np.arange(X_log.shape[0])
    np.random.shuffle(s1)
    X_log = X_log[s1, :]
    Y_log = Y_log[s1]

    site_number_new = [site_number[i] for i in s1]
    dates_new = [dates[i] for i in s1]
    satellites_new = [satellites[i] for i in s1]

    transformerX = RobustScaler().fit(X_log)
    X_trans = transformerX.transform(X_log)
    min_max_scalerX = MinMaxScaler().fit(X_trans)
    X_trans2 = min_max_scalerX.transform(X_trans)

    transformerY = RobustScaler().fit(np.reshape(Y_log, (-1, 1)))
    Y_trans = transformerY.transform(np.reshape(Y_log, (-1, 1)))
    min_max_scalerY = MinMaxScaler().fit(Y_trans)
    Y_trans2 = min_max_scalerY.transform(Y_trans)

    return (X_trans2, Y_trans2, transformerX, transformerY,
            min_max_scalerX, min_max_scalerY, shift_value_X,
            shift_value_Y, site_number_new,
            dates_new, satellites_new, s1)

def ytest_to_initial_scale(ytest, min_max_scalerY, transformerY, shift_value_Y):
    ytest_inv1 = min_max_scalerY.inverse_transform(
        np.reshape(ytest, (-1, 1)))
    ytest_inv2 = transformerY.inverse_transform(ytest_inv1)
    ytest_inv2 = np.exp(ytest_inv2)
    ytest_rescaled = ytest_inv2 - shift_value_Y - 1
    return ytest_rescaled
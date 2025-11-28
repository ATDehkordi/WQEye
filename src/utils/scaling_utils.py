# src/utils/scaling_utils.py
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler,RobustScaler
import numpy as np


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




def apply_scaling(x, y, scaler_name, site_number, dates, satellite):
    # if scaler_name == "StandardScaler":
    #     x_scaler = StandardScaler()
    #     y_scaler = StandardScaler()
    #     x_rescaled = x_scaler.fit_transform(x)
    #     y_rescaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    #     scalers = {
    #         'x_scaler': x_scaler,
    #         'y_scaler': y_scaler
    #     }
    #     extras = {}

    if scaler_name == "MinMaxScaler":

        s1 = np.arange(x.shape[0])
        np.random.shuffle(s1)
        X = x[s1, :]
        Y = y[s1]
        
        site_number_new = [site_number[i] for i in s1]
        dates_new = [dates[i] for i in s1]
        satellite_new = [satellite[i] for i in s1]
        
        min_max_scalerX = MinMaxScaler().fit(X)
        x_rescaled = min_max_scalerX.transform(X)

        min_max_scalerY = MinMaxScaler().fit(np.reshape(Y, (-1, 1)))
        y_rescaled = min_max_scalerY.transform(np.reshape(Y, (-1, 1)))
    
        scalers = {
            'method': 'MinMaxScaler',
            'min_max_scalerX': min_max_scalerX,
            'min_max_scalerY': min_max_scalerY
            
        }
        extras = {
            'site_number': site_number_new,
            'dates': dates_new,
            'satellite': satellite_new,
            's1': s1
        }
        return x_rescaled, y_rescaled, scalers, extras

    else:
        (x_rescaled, y_rescaled, transformerX, transformerY,
         min_max_scalerX, min_max_scalerY, shift_value_X,
         shift_value_Y, site_number_new, dates_new, satellite_new, s1) = log_scale_transform(x, y, site_number, dates, satellite)

        scalers = {
            'method': 'LogScale',
            'transformerX': transformerX,
            'transformerY': transformerY,
            'min_max_scalerX': min_max_scalerX,
            'min_max_scalerY': min_max_scalerY,
            'shift_value_X': shift_value_X,
            'shift_value_Y': shift_value_Y
            
        }
        extras = {
            'site_number': site_number_new,
            'dates': dates_new,
            'satellite': satellite_new,
            's1': s1
        }

    return x_rescaled, y_rescaled, scalers, extras


def inverse_transform_y(y_scaled, scalers):
    y_scaled = np.asarray(y_scaled)
    method = scalers.get('method', 'LogScale')

    if method == 'LogScale':
        return ytest_to_initial_scale(
            y_scaled,
            scalers['min_max_scalerY'],
            scalers['transformerY'],
            scalers['shift_value_Y'],
        )

    elif method == 'MinMaxScaler':
        return scalers['min_max_scalerY'].inverse_transform(
            np.reshape(y_scaled, (-1, 1))
        )

    else:
        raise ValueError(f"Unknown scaling method: {method}")





def ytest_to_initial_scale(ytest, min_max_scalerY, transformerY, shift_value_Y):
    ytest_inv1 = min_max_scalerY.inverse_transform(
        np.reshape(ytest, (-1, 1)))
    ytest_inv2 = transformerY.inverse_transform(ytest_inv1)
    ytest_inv2 = np.exp(ytest_inv2)
    ytest_rescaled = ytest_inv2 - shift_value_Y - 1
    return ytest_rescaled



def r_squared(y, y_hat):
    ''' Logarithmic R^2 '''
    slope_, intercept_, r_value, p_value, std_err = stats.linregress(y, y_hat)
    return r_value**2 * 100


def mape(y, y_hat):
    ''' Mean Absolute Percentage Error '''
    return 100 * np.mean(np.abs((y - y_hat) / y))
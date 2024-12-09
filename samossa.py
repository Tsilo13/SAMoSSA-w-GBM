import numpy as np
import pandas as pd
from scipy.linalg import svd
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
import yfinance as yf

def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def construct_trajectory_matrix(data, window_length):
    N = len(data)
    K = N - window_length + 1
    trajectory_matrix = np.zeros((window_length, K))
    for i in range(K):
        trajectory_matrix[:, i] = data[i:i + window_length]
    return trajectory_matrix

def apply_svd(trajectory_matrix):
    U, Sigma, VT = svd(trajectory_matrix, full_matrices=False)
    return U, Sigma, VT

def variance_explained(sigma, threshold=0.95):
    normalized_sigma = sigma / np.sum(sigma)
    cumulative_variance = np.cumsum(normalized_sigma)
    k = np.argmax(cumulative_variance >= threshold) + 1
    return k, cumulative_variance

def reconstruct_components(U, Sigma, VT, k, window_length):
    reconstructed_matrix = np.dot(U[:, :k], np.dot(np.diag(Sigma[:k]), VT[:k, :]))
    reconstructed_series = np.zeros(reconstructed_matrix.shape[1] + window_length - 1)
    for i in range(reconstructed_matrix.shape[1]):
        reconstructed_series[i:i + window_length] += reconstructed_matrix[:, i]
    count = np.zeros_like(reconstructed_series)
    for i in range(reconstructed_matrix.shape[1]):
        count[i:i + window_length] += 1
    reconstructed_series /= count
    return reconstructed_series

def calculate_residuals(time_series, reconstructed_series):
    N = len(time_series)
    residuals = time_series - reconstructed_series[:N]
    return residuals

def fit_ar_model(residuals, max_ar=10):
    best_aic = np.inf
    best_order = 0
    best_model = None

    for lag in range(1, max_ar + 1):
        try:
            model = AutoReg(residuals, lags=lag)
            model_fitted = model.fit()
            aic = model_fitted.aic
            if aic < best_aic:
                best_aic = aic
                best_order = lag
                best_model = model_fitted
        except (ValueError, np.linalg.LinAlgError) as e:
            continue

    if best_model is None:
        raise ValueError("No valid AR model found.")
        
    return best_model, best_order

def forecast_residuals_ar(ar_model_fitted, forecast_steps):
    forecast = ar_model_fitted.predict(start=len(ar_model_fitted.model.endog), end=len(ar_model_fitted.model.endog) + forecast_steps - 1, dynamic=False)
    return forecast

def fit_gbm_model(residuals, forecast_steps):
    dt = 1
    mu = np.mean(residuals)
    sigma = np.std(residuals)

    forecast = np.zeros(forecast_steps)
    forecast[0] = residuals[-1]

    for t in range(1, forecast_steps):
        forecast[t] = forecast[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal())
    
    return forecast

def calculate_mae(actual, forecast):
    return np.mean(np.abs(actual - forecast))

def calculate_rmse(actual, forecast):
    return np.sqrt(np.mean((actual - forecast) ** 2))

def plot_forecast(time_series, deterministic_component, combined_forecast_ar, combined_forecast_gbm, forecast_dates):
    plt.figure(figsize=(14, 7))
    plt.plot(time_series.index, time_series.values, label='Original Time Series', color='blue')
    plt.plot(time_series.index, deterministic_component[:len(time_series)], label='Deterministic Component', color='orange')
    plt.plot(forecast_dates, combined_forecast_ar, label='Combined Forecast (AR)', color='red')
    plt.plot(forecast_dates, combined_forecast_gbm, label='Combined Forecast (GBM)', color='green')
    plt.title('Combined Forecast (AR and GBM)')
    plt.xlabel('Time')
    plt.ylabel('Forecasted Value')
    plt.legend()
    plt.show()

def samossa(data, window_length=50, variance_threshold=0.95, forecast_steps=10):
    time_series = data['Close']  # Assuming 'Close' is the column with the time series data

    # Step 1: Construct the Trajectory Matrix
    trajectory_matrix = construct_trajectory_matrix(time_series.values, window_length)

    # Step 2: Apply SVD
    U, Sigma, VT = apply_svd(trajectory_matrix)

    # Step 3: Determine Number of Significant Singular Values
    k, cumulative_variance = variance_explained(Sigma, threshold=variance_threshold)
    print(f"Number of significant singular values (k): {k}")

    # Step 4: Reconstruct the Deterministic Components
    deterministic_component = reconstruct_components(U, Sigma, VT, k, window_length)

    # Step 5: Calculate Residuals
    residuals = calculate_residuals(time_series.values, deterministic_component)

    # Step 6a: Fit an AR Model to the Residuals
    ar_model_fitted, best_lag = fit_ar_model(residuals)
    forecasted_residuals_ar = forecast_residuals_ar(ar_model_fitted, forecast_steps)

    # Step 6b: Fit a GBM Model to the Residuals
    forecasted_residuals_gbm = fit_gbm_model(residuals, forecast_steps)

    # Combine deterministic component with forecasted residuals
    deterministic_last_value = deterministic_component[-1]
    combined_forecast_ar = deterministic_last_value + forecasted_residuals_ar
    combined_forecast_gbm = deterministic_last_value + forecasted_residuals_gbm

    # Generate forecast dates
    last_date = data.index[-1]
    forecast_dates = pd.date_range(start=last_date, periods=forecast_steps + 1, freq='B')[1:]

    # Calculate error metrics
    mae_ar = calculate_mae(time_series[-forecast_steps:].values, combined_forecast_ar)
    rmse_ar = calculate_rmse(time_series[-forecast_steps:].values, combined_forecast_ar)
    mae_gbm = calculate_mae(time_series[-forecast_steps:].values, combined_forecast_gbm)
    rmse_gbm = calculate_rmse(time_series[-forecast_steps:].values, combined_forecast_gbm)

    # Plot the forecasts
    plot_forecast(time_series, deterministic_component, combined_forecast_ar, combined_forecast_gbm, forecast_dates)

    # Return forecasts, dates, and error metrics for comparison
    return combined_forecast_ar, combined_forecast_gbm, forecast_dates, mae_ar, rmse_ar, mae_gbm, rmse_gbm

def main():
    ticker = 'INTC'
    start_date = '2008-01-01'
    end_date = '2024-07-05'
    forecast_steps = 10
    window_length = 50
    variance_threshold = 0.95

    data = load_data(ticker, start_date, end_date)
    combined_forecast_ar, combined_forecast_gbm, forecast_dates, mae_ar, rmse_ar, mae_gbm, rmse_gbm = samossa(data, window_length, variance_threshold, forecast_steps)

    print("AR Model Forecasts:")
    for date, value in zip(forecast_dates, combined_forecast_ar):
        print(f"{date.date()}: {value}")
    
    print(f"AR Model MAE: {mae_ar}")
    print(f"AR Model RMSE: {rmse_ar}")

    print("\nGBM Model Forecasts:")
    for date, value in zip(forecast_dates, combined_forecast_gbm):
        print(f"{date.date()}: {value}")
    
    print(f"GBM Model MAE: {mae_gbm}")
    print(f"GBM Model RMSE: {rmse_gbm}")

if __name__ == "__main__":
    main()

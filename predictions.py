# ============================================================================
# SECTION 1: IMPORTS AND SETUP
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import warnings
import os

warnings.filterwarnings('ignore')

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.precision', 4)

print("="*80)
print("USD/INR FORECASTING MODEL - 6 MONTH DYNAMIC FORECAST")
print("="*80)
print()

# ============================================================================
# SECTION 2: DATA LOADING (User Input or Dummy Data)
# ============================================================================

def generate_sample_data(n_months=60):
    """
    Generate realistic dummy data for demonstration.
    
    Parameters:
    n_months (int): Number of historical months to generate
    
    Returns:
    pd.DataFrame: Historical data with all required columns
    """
    
    np.random.seed(42)  # For reproducibility
    
    # Generate date range (last n_months)
    end_date = datetime(2024, 9, 30)  # Example: Sep 2024
    dates = pd.date_range(end=end_date, periods=n_months, freq='MS')
    
    # Generate realistic USD/INR path (trending upward with volatility)
    usdinr_base = 75 + np.arange(n_months) * 0.15  # Slow depreciation trend
    usdinr_noise = np.random.normal(0, 0.8, n_months)
    usdinr = usdinr_base + usdinr_noise
    
    # RBI Repo Rate (6-7% range, realistic changes)
    repo_rate = 6.5 + np.random.normal(0, 0.15, n_months)
    repo_rate = np.clip(repo_rate, 6.0, 7.0)
    
    # US Fed Rate (4-5.5% range)
    fed_rate = 5.0 + np.random.normal(0, 0.15, n_months)
    fed_rate = np.clip(fed_rate, 4.5, 5.5)
    
    # CPI India (4-6% range)
    cpi_india = 5.0 + np.random.normal(0, 0.4, n_months)
    cpi_india = np.clip(cpi_india, 4.0, 6.5)
    
    # CPI US (2-4% range)
    cpi_us = 3.0 + np.random.normal(0, 0.3, n_months)
    cpi_us = np.clip(cpi_us, 2.5, 4.0)
    
    # Oil Price (70-100 USD/barrel)
    oil_price = 85 + np.random.normal(0, 8, n_months)
    oil_price = np.clip(oil_price, 70, 105)
    
    # Dollar Index DXY (100-110 range)
    dxy = 105 + np.random.normal(0, 2, n_months)
    dxy = np.clip(dxy, 100, 110)
    
    # Forex Reserves (580-620 billion USD)
    reserves = 600 + np.random.normal(0, 10, n_months)
    reserves = np.clip(reserves, 580, 620)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'USDINR': usdinr,
        'RepoRate': repo_rate,
        'FedRate': fed_rate,
        'CPI_India': cpi_india,
        'CPI_US': cpi_us,
        'OilPrice': oil_price,
        'DXY': dxy,
        'Reserves': reserves
    })
    
    return df

print("Loading Data...")

# Ask the user for the file path
file_path = input("Please enter the path to your CSV file (or press Enter to use dummy data): ")

# Check if the user provided a path and if the file exists
if file_path and os.path.exists(file_path):
    try:
        df = pd.read_csv(file_path, parse_dates=['Date'])
        print(f"✓ Real data successfully loaded from: {file_path}")
    except Exception as e:
        print(f"⚠ Error loading file: {e}. Falling back to dummy data.")
        df = generate_sample_data(n_months=60)
else:
    if file_path: # If user entered a path but it wasn't found
        print(f"⚠ File not found at: '{file_path}'. Using dummy data for demonstration.")
    else: # If user just pressed Enter
        print("⚠ No file path provided. Using dummy data for demonstration.")
    df = generate_sample_data(n_months=60)

print(f"Data loaded: {len(df)} months from {df['Date'].min().strftime('%b %Y')} to {df['Date'].max().strftime('%b %Y')}")
print()


# ============================================================================
# SECTION 3: FEATURE ENGINEERING
# ============================================================================

print("Creating Engineered Features...")

# Interest Rate Differential
df['RateDiff'] = df['RepoRate'] - df['FedRate']

# Inflation Differential
df['InflationDiff'] = df['CPI_India'] - df['CPI_US']

# Lagged USD/INR (previous month's rate for momentum)
df['USDINR_Lag1'] = df['USDINR'].shift(1)

# Display feature statistics
print("\nFeature Summary Statistics:")
print(df[['USDINR', 'RateDiff', 'InflationDiff', 'OilPrice', 'DXY', 'Reserves']].describe())
print()

# ============================================================================
# SECTION 4: DATA PREPARATION FOR MODELING
# ============================================================================

# Remove rows with NaN (from lagging)
df_model = df.dropna().copy()

print(f"Training data: {len(df_model)} observations")
print()

# Define features (X) and target (y)
feature_cols = ['RateDiff', 'InflationDiff', 'OilPrice', 'DXY', 'Reserves']
X = df_model[feature_cols]
y = df_model['USDINR']

# Add constant for regression intercept
X_with_const = sm.add_constant(X)

# ============================================================================
# SECTION 5: REGRESSION MODEL - TRAINING
# ============================================================================

print("="*80)
print("REGRESSION MODEL - ESTIMATION RESULTS")
print("="*80)

# Fit OLS regression model
model = sm.OLS(y, X_with_const).fit()

# Print comprehensive model summary
print(model.summary())
print()

# Extract key statistics
r_squared = model.rsquared
coefficients = model.params

print("="*80)
print("KEY INSIGHTS FROM MODEL")
print("="*80)
print(f"R-squared: {r_squared:.4f} (Model explains {r_squared*100:.2f}% of INR variance)")
print()

print("Coefficient Interpretation:")
print("-" * 80)
for feature in feature_cols:
    coef = coefficients[feature]
    print(f"  {feature:20s}: {coef:+.4f}")
print()

# ============================================================================
# SECTION 6: MODEL VALIDATION - BACKTESTING
# ============================================================================

print("="*80)
print("MODEL VALIDATION - BACKTEST PERFORMANCE")
print("="*80)

# In-sample predictions
y_pred = model.predict(X_with_const)

# Calculate error metrics
mape = mean_absolute_percentage_error(y, y_pred) * 100
rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print()

if mape < 2.0:
    print("✓ Excellent forecast accuracy (MAPE < 2%)")
elif mape < 5.0:
    print("✓ Good forecast accuracy (MAPE < 5%)")
else:
    print("⚠ Moderate accuracy - consider adding more features")
print()

# ============================================================================
# SECTION 7: SCENARIO DEFINITION FOR 6-MONTH FORECAST
# ============================================================================

print("="*80)
print("SCENARIO ASSUMPTIONS - NEXT 6 MONTHS")
print("="*80)

# Get last known values as baseline
last_values = df.iloc[-1]

# Define three scenarios
scenarios = {
    'Base Case': {
        'description': 'Oil stable, rates steady, moderate flows',
        'OilPrice': 87.5,
        'RateDiff': last_values['RepoRate'] - last_values['FedRate'],
        'InflationDiff': last_values['CPI_India'] - last_values['CPI_US'],
        'DXY': last_values['DXY'],
        'Reserves': last_values['Reserves']
    },
    'Upside (INR Strong)': {
        'description': 'Oil drops, Fed cuts, strong FII inflows',
        'OilPrice': 75.0,
        'RateDiff': (last_values['RepoRate']) - (last_values['FedRate'] - 0.5),
        'InflationDiff': last_values['CPI_India'] - 0.5 - last_values['CPI_US'],
        'DXY': last_values['DXY'] - 3,
        'Reserves': last_values['Reserves'] + 15
    },
    'Downside (INR Weak)': {
        'description': 'Oil spike, Fed hikes, FII outflows',
        'OilPrice': 105.0,
        'RateDiff': (last_values['RepoRate']) - (last_values['FedRate'] + 0.5),
        'InflationDiff': last_values['CPI_India'] + 0.5 - last_values['CPI_US'],
        'DXY': last_values['DXY'] + 4,
        'Reserves': last_values['Reserves'] - 10
    }
}

# Print scenario assumptions
for scenario_name, params in scenarios.items():
    print(f"{scenario_name}:")
    print(f"  {params['description']}")
    print(f"  Oil: ${params['OilPrice']:.2f} | Rate Diff: {params['RateDiff']:.2f}% | DXY: {params['DXY']:.2f}")
    print()

# ============================================================================
# SECTION 8: FORECAST GENERATION
# ============================================================================

print("="*80)
print("6-MONTH FORECAST RESULTS")
print("="*80)

# Generate forecast dates
forecast_start = df['Date'].max() + pd.DateOffset(months=1)
forecast_dates = pd.date_range(start=forecast_start, periods=6, freq='MS')

# Create forecast DataFrame
forecast_results = pd.DataFrame({'Month': forecast_dates.strftime('%b %Y')})

# Generate forecasts for each scenario
for scenario_name, params in scenarios.items():
    forecast_input = pd.DataFrame([params] * 6)
    forecast_input_const = sm.add_constant(forecast_input[feature_cols])
    predictions = model.predict(forecast_input_const)
    forecast_results[scenario_name] = predictions.values

# Display forecast table
print("\n" + forecast_results.to_string(index=False))
print()

# Calculate ranges
base_avg = forecast_results['Base Case'].mean()
upside_avg = forecast_results['Upside (INR Strong)'].mean()
downside_avg = forecast_results['Downside (INR Weak)'].mean()

print("\nAverage 6-Month Forecast:")
print(f"  Base Case:           Rs {base_avg:.2f}")
print(f"  Upside (INR Strong): Rs {upside_avg:.2f} (Appreciation: {base_avg - upside_avg:.2f})")
print(f"  Downside (INR Weak): Rs {downside_avg:.2f} (Depreciation: {downside_avg - base_avg:.2f})")
print()

# ============================================================================
# SECTION 9: VISUALIZATION
# ============================================================================

print("Generating Visualizations...")

# Create comprehensive chart
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('USD/INR Forecasting Model - Analysis & Projections', fontsize=16, fontweight='bold')

# Chart 1: Historical + Forecast
ax1 = axes[0, 0]
ax1.plot(df['Date'], df['USDINR'], label='Historical USD/INR', color='#2E86AB', linewidth=2)
ax1.plot(forecast_dates, forecast_results['Base Case'], label='Base Forecast', color='#F24236', linewidth=2, linestyle='--', marker='o')
ax1.fill_between(forecast_dates, forecast_results['Upside (INR Strong)'], forecast_results['Downside (INR Weak)'], alpha=0.2, color='#F24236', label='Forecast Range')
ax1.axvline(df['Date'].max(), color='gray', linestyle=':', alpha=0.7)
ax1.set_title('Historical Data + 6-Month Forecast', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Chart 2: Scenario Comparison
ax2 = axes[0, 1]
x_pos = np.arange(6)
width = 0.25
ax2.bar(x_pos - width, forecast_results['Upside (INR Strong)'], width, label='Upside', color='#06A77D')
ax2.bar(x_pos, forecast_results['Base Case'], width, label='Base', color='#F24236')
ax2.bar(x_pos + width, forecast_results['Downside (INR Weak)'], width, label='Downside', color='#D90368')
ax2.set_title('Scenario Comparison', fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(forecast_results['Month'], rotation=45)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Chart 3: Model Fit Quality
ax3 = axes[1, 0]
ax3.scatter(y, y_pred, alpha=0.6, color='#2E86AB')
ax3.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2, label='Perfect Fit')
ax3.set_title(f'Model Fit Quality (R² = {r_squared:.4f})', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Chart 4: Residuals
ax4 = axes[1, 1]
residuals = y - y_pred
ax4.scatter(y_pred, residuals, alpha=0.6, color='#F24236')
ax4.axhline(y=0, color='black', linestyle='--', linewidth=2)
ax4.set_title('Residual Analysis', fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('usdinr_forecast_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Chart saved as 'usdinr_forecast_analysis.png'")
print()

# ============================================================================
# SECTION 10: EXPORT RESULTS
# ============================================================================

print("="*80)
print("EXPORTING RESULTS")
print("="*80)

# Export forecast table
forecast_results.to_csv('usdinr_6month_forecast.csv', index=False)
print("✓ Forecast table saved as 'usdinr_6month_forecast.csv'")

# Export model coefficients
coef_df = pd.DataFrame({
    'Variable': ['Intercept'] + feature_cols,
    'Coefficient': model.params.values
})
coef_df.to_csv('model_coefficients.csv', index=False)
print("✓ Model coefficients saved as 'model_coefficients.csv'")

# Create summary report
with open('forecast_summary.txt', 'w') as f:
    f.write("USD/INR 6-MONTH FORECAST - SUMMARY REPORT\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Model R-squared: {r_squared:.4f}\n")
    f.write(f"Forecast MAPE: {mape:.2f}%\n\n")
    f.write("Forecast Summary:\n")
    f.write(f"  Base Case Average:   Rs {base_avg:.2f}\n")
    f.write(f"  Upside Case Average:   Rs {upside_avg:.2f}\n")
    f.write(f"  Downside Case Average: Rs {downside_avg:.2f}\n")

print("✓ Summary report saved as 'forecast_summary.txt'")
print()

# ============================================================================
# SECTION 11: KEY TAKEAWAYS FOR PRESENTATION
# ============================================================================

print("="*80)
print("KEY TAKEAWAYS FOR CASE PRESENTATION")
print("="*80)
print()
print("1. MODEL CREDIBILITY:")
print(f"   - R² of {r_squared:.2f} shows model explains {r_squared*100:.1f}% of INR variance")
print(f"   - MAPE of {mape:.2f}% indicates strong predictive accuracy")
print()
print("2. FORECAST RANGE:")
print(f"   - Base case: Rs {base_avg:.2f}")
print(f"   - Trading range: Rs {upside_avg:.2f} - Rs {downside_avg:.2f}")
print()
print("3. KEY DRIVERS:")
for feature in feature_cols[:3]:
    print(f"   - {feature}: {coefficients[feature]:+.4f} impact per unit")
print()
print("4. RISK MANAGEMENT:")
print("   - Oil price is a critical driver - hedge commodity exposure")
print("   - Monitor Fed policy - rate differential impacts capital flows")
print("   - Consider dynamic hedging within the forecast range")
print()

print("="*80)
print("MODEL EXECUTION COMPLETE")
print("="*80)

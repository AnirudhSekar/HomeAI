import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from sklearn.feature_selection import RFECV
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold
from roi_metrics import calculate_roi_metrics
from plot_data import plot_analysis
import pickle
import os
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from house_statistics import print_house_statistics

# Set global plotting parameters
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.autolayout'] = True
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
# Set Seaborn style
sns.set_theme(style="whitegrid")
sns.set_context("notebook", font_scale=1.2)

# Set Matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')


# Load and preprocess data
print("Loading data...")
df = pd.read_csv('real_estate_texas_500_2024.csv')
print("Initial shape:", df.shape)

# Check initial missing values
print("\nInitial missing value counts:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Drop non-predictive columns

# Get the location from the url column using regex
df['location'] = df['url'].str.extract(r'(\w+)$')
df['location'] = df['location'].str.replace('_', ' ').str.title()
# Make location numerical
df['location'] = pd.factorize(df['location'])[0]

df = df.drop(['url', 'id', 'text'], axis=1, errors='ignore')
print("\nShape after dropping non-predictive columns:", df.shape)

# Handle missing values
print("\nHandling missing values...")


# Clean price data
if df['listPrice'].dtype == 'object':
    df['listPrice'] = df['listPrice'].str.replace('$', '').str.replace(',', '')
df['listPrice'] = pd.to_numeric(df['listPrice'], errors='coerce')
df['listPrice_log'] = np.log1p(df['listPrice'])

# Create price brackets for stratification
df['price_bracket'] = pd.qcut(df['listPrice'], q=5, labels=False)

# Handle missing values in listPrice first
if df['listPrice'].isnull().sum() > 0:
    print(f"\nMissing values in listPrice: {df['listPrice'].isnull().sum()}")
    df = df.dropna(subset=['listPrice'])
    print(f"Rows remaining after dropping missing listPrice: {len(df)}")

# Handle numeric features
numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
numeric_missing = df[numeric_features].isnull().sum()
print("\nMissing values in numeric features:")
print(numeric_missing[numeric_missing > 0])

# Use more robust imputation for numeric columns
for col in numeric_features:
    if col not in ['listPrice', 'listPrice_log']:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            # Use median for features with low skewness, mean otherwise
            skewness = df[col].skew()
            if abs(skewness) < 1:
                fill_value = df[col].mean()
                method = "mean"
            else:
                fill_value = df[col].median()
                method = "median"
            df[col] = df[col].fillna(fill_value)
            print(f"Imputed {missing_count} values in {col} using {method}")

# Handle categorical features
categorical_features = df.select_dtypes(include=['object']).columns
categorical_missing = df[categorical_features].isnull().sum()
print("\nMissing values in categorical features:")
print(categorical_missing[categorical_missing > 0])

# Use mode imputation for categorical variables
for col in categorical_features:
    missing_count = df[col].isnull().sum()
    if missing_count > 0:
        df[col] = df[col].fillna(df[col].mode()[0])
        print(f"Imputed {missing_count} values in {col} using mode")

# Validate no missing values remain
remaining_missing = df.isnull().sum()
if remaining_missing.sum() > 0:
    print("\nWarning: Missing values still present:")
    print(remaining_missing[remaining_missing > 0])
    raise ValueError("Dataset still contains missing values after imputation")
else:
    print("\nAll missing values have been handled successfully")

# Now proceed with outlier detection
print("\nPerforming outlier detection...")
iso_forest = IsolationForest(contamination=0.1, random_state=42)
outlier_labels = iso_forest.fit_predict(df[['listPrice', 'sqft']])
df = df[outlier_labels == 1]  # Keep only inliers
print(f"Shape after outlier removal: {df.shape}")

# Verify no NaN values remain
remaining_nans = df.isnull().sum()
if remaining_nans.sum() > 0:
    print("\nWarning: NaN values still present in dataset:")
    print(remaining_nans[remaining_nans > 0])
    raise ValueError("Dataset contains NaN values after cleaning")
else:
    print("\nAll NaN values have been handled.")

# Feature engineering
print("Shape after outlier removal:", df.shape)

# Add seasonality features if listing date is available
if 'listDate' in df.columns:
    df['listDate'] = pd.to_datetime(df['listDate'])
    df['list_month'] = df['listDate'].dt.month
    df['list_quarter'] = df['listDate'].dt.quarter
    df['list_year'] = df['listDate'].dt.year
    df['list_day_of_week'] = df['listDate'].dt.dayofweek
    df['list_is_weekend'] = df['list_day_of_week'].isin([5, 6]).astype(int)
    
    # Create cyclical features for month and day of week
    df['month_sin'] = np.sin(2 * np.pi * df['list_month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['list_month']/12)
    df['day_sin'] = np.sin(2 * np.pi * df['list_day_of_week']/7)
    df['day_cos'] = np.cos(2 * np.pi * df['list_day_of_week']/7)

# Safe division function to handle zeros
def safe_division(numerator, denominator, fill_value=None):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = numerator / denominator
        if fill_value is not None:
            result[denominator == 0] = fill_value
        return result

if 'sqft' in df.columns:
    # Avoid division by zero for price_per_sqft
    df['price_per_sqft'] = safe_division(df['listPrice'], df['sqft'])
    df['sqft_log'] = np.log1p(df['sqft'])
    
if 'beds' in df.columns and 'baths' in df.columns:
    # Avoid division by zero for bed_bath_ratio
    df['bed_bath_ratio'] = safe_division(df['beds'], df['baths'], fill_value=np.nan)
    df['total_rooms'] = df['beds'] + df['baths']

# Replace infinite values with NaN
df = df.replace([np.inf, -np.inf], np.nan)

# Handle any new NaN values from inf replacement
inf_replaced_missing = df.isnull().sum()
if inf_replaced_missing.sum() > 0:
    print("\nMissing values after replacing infinities:")
    print(inf_replaced_missing[inf_replaced_missing > 0])
    
    # Impute new NaN values
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_features:
        if col not in ['listPrice', 'listPrice_log']:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                df[col] = df[col].fillna(df[col].median())
                print(f"Imputed {missing_count} new NaN values in {col}")

# Verify no infinite values remain
inf_check = np.isinf(df.select_dtypes(include=np.number)).any()
inf_cols = inf_check[inf_check].index.tolist()
if inf_cols:
    raise ValueError(f"Infinite values found in columns: {inf_cols}")

# Handle categorical features
# Handle categorical features
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
df = pd.get_dummies(df, columns=categorical_columns)

# Create polynomial features for sqft and price_per_sqft
if 'sqft' in df.columns:
    poly = PolynomialFeatures(degree=2, include_bias=False)
    sqft_poly = poly.fit_transform(df[['sqft']])[:, 1:]  # Exclude the original feature
    df['sqft_squared'] = sqft_poly[:, 0]

if 'price_per_sqft' in df.columns:
    price_sqft_poly = poly.fit_transform(df[['price_per_sqft']])[:, 1:]
    df['price_per_sqft_squared'] = price_sqft_poly[:, 0]

# Add price per room feature
if 'listPrice' in df.columns and 'total_rooms' in df.columns:
    df['price_per_room'] = safe_division(df['listPrice'], df['total_rooms'])
    # Replace any remaining infinite values with median of non-infinite values
    non_inf_mask = ~np.isinf(df['price_per_room'])
    if non_inf_mask.any():
        median_value = df.loc[non_inf_mask, 'price_per_room'].median()
        df.loc[~non_inf_mask, 'price_per_room'] = median_value
# Create interaction terms between numerical features
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
numeric_cols = [col for col in numeric_cols if col not in ['listPrice', 'listPrice_log']]

# Select key features for interactions to avoid exponential feature growth
key_features = ['sqft', 'beds', 'baths', 'total_rooms']
key_features = [f for f in key_features if f in numeric_cols]

for i in range(len(key_features)):
    for j in range(i+1, len(key_features)):
        feat1, feat2 = key_features[i], key_features[j]
        if feat1 in df.columns and feat2 in df.columns:
            df[f'{feat1}_{feat2}_interaction'] = df[feat1] * df[feat2]

print("Shape after preprocessing:", df.shape)

# Split features and target
X = df.drop(['listPrice', 'listPrice_log'], axis=1)
y = df['listPrice_log']  # Use log-transformed target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

 # Create validation set for early stopping
scaler = RobustScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train_scaled, y_train, test_size=0.2, random_state=42
        )



# Custom weighted objective function
def weighted_objective(y_true, y_pred, sample_weight=None):
    grad = y_pred - y_true
    hess = np.ones_like(y_pred)
    
    # Calculate weights based on absolute error
    errors = np.abs(grad)
    weights = 1 + np.log1p(errors)  # Higher weight for larger errors
    
    # Combine with sample_weight if provided
    if sample_weight is not None:
        weights *= sample_weight
        
    # Apply weights to gradients and hessians
    grad *= weights
    hess *= weights
    return grad, hess


# Create custom callback for learning rate scheduling
class LearningRateScheduler(xgb.callback.TrainingCallback):
    def __init__(self, base_lr, decay_factor, decay_every_n_rounds):
        self.base_lr = base_lr
        self.current_lr = base_lr
        self.decay_factor = decay_factor
        self.decay_every_n_rounds = decay_every_n_rounds
        
    def after_iteration(self, model, epoch, evals_log):
        if (epoch + 1) % self.decay_every_n_rounds == 0:
            self.current_lr *= self.decay_factor
            model.set_param('learning_rate', str(self.current_lr))
        return False


def model_exists(filepath):
    return os.path.exists(filepath)

# Add error handling for model fitting
try:
    model_path = "xgb_model.pkl"
    
    if model_exists(model_path):
        print("\nLoading existing XGBoost model...")
        xgb_model = pickle.load(open(model_path, "rb"))
    else:
        # Debug infinite values
        print("\nChecking for infinite values in X_train:")
        inf_mask = np.any(np.isinf(X_train), axis=0)
        print("Columns with infinite values:", X_train.columns[inf_mask].tolist())

        print("\nChecking for extremely large values in X_train:")
        max_values = X_train.max()
        large_value_cols = max_values[max_values > 1e10].index.tolist()
        print("Columns with extremely large values:", large_value_cols)
        if large_value_cols:
            for col in large_value_cols:
                print(f"\nMax value in {col}:", X_train[col].max())

        

        # Feature selection using RFECV
        print("\nPerforming feature selection with RFECV...")
        base_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        rfecv = RFECV(
            estimator=base_model,
            step=1,
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
            scoring='neg_mean_squared_error',
            n_jobs=8,
            min_features_to_select=5
        )

        # Fit RFECV
        X_train_selected = rfecv.fit_transform(X_train_scaled, y_train)
        X_test_selected = rfecv.transform(X_test_scaled)

        print(f"Optimal number of features: {rfecv.n_features_}")
        selected_features = X.columns[rfecv.support_]
        print("Selected features:", selected_features.tolist())

        xgb_params = {
            'n_estimators': [3000, 4000, 5000],
            'learning_rate': [0.001, 0.003, 0.005, 0.008],
            'max_depth': [4, 6, 8],
            'min_child_weight': [3, 5, 7],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'colsample_bylevel': [0.7, 0.8, 0.9],
            'gamma': [0.1, 0.2, 0.3],
            'reg_alpha': [0.01, 0.1, 1.0, 10.0],  # L1 regularization
            'reg_lambda': [1.0, 10.0, 100.0],    # L2 regularization
            'max_delta_step': [1, 3, 5],
            'scale_pos_weight': [1],
            'rate_drop': [0.1, 0.2],             # Learning rate decay
            'skip_drop': [0.7, 0.8]              # Skip dropout rate
        }
        # Create evaluation callback
        eval_log = xgb.callback.EvaluationMonitor()
        # Train XGBoost with stratified k-fold CV
        print("\nTraining XGBoost...")

        # Create stratified k-fold CV
        skf = KFold(n_splits=5, shuffle=True, random_state=42)

        # Create and fit feature selector
        print("\nPerforming feature selection...")
        feature_selector = RFECV(
            estimator=xgb.XGBRegressor(objective='reg:squarederror'),
            step=1,
            cv=3,
            min_features_to_select=5,
            scoring='neg_mean_squared_error'
        )
        feature_selector.fit(X_train_scaled, y_train)

        # Get selected features and their rankings
        selected_mask = feature_selector.support_
        feature_rankings = feature_selector.ranking_
        selected_features = X_train.columns[selected_mask]
        print(f"\nSelected {len(selected_features)} features")

        # Calculate feature weights
        feature_weights = 1.0/np.array(feature_rankings)


        xgb_model = RandomizedSearchCV(
            estimator=xgb.XGBRegressor(
                objective=weighted_objective,
                random_state=42,
                enable_categorical=True,
                tree_method='hist',
                early_stopping_rounds=100,
                eval_metric=['rmse', 'mae'],
                callbacks=[
                    xgb.callback.EarlyStopping(
                        rounds=100,
                        save_best=True,
                        maximize=False
                    ),
                    LearningRateScheduler(
                        base_lr=0.01,
                        decay_factor=0.95,
                        decay_every_n_rounds=100
                    )
                ]
            ),
            param_distributions=xgb_params,
            n_iter=50,  # Increased number of iterations
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
            scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
            refit='neg_mean_squared_error',
            n_jobs=8,
            verbose=2,
            return_train_score=True,
            error_score='raise'
        )

       

        # Set up evaluation sets for early stopping
        eval_set = [
            (X_train_final, y_train_final),  # Training set
            (X_val, y_val),                  # Validation set
            (X_test_scaled, y_test)          # Test set
        ]
        xgb_model.fit(
            X_train_final,
            y_train_final,
            eval_set=[(X_train_final, y_train_final), (X_val, y_val), (X_test_scaled, y_test)],
            verbose=True,
            feature_weights=feature_weights,
            sample_weight=1 / (1 + np.abs(y_train_final - y_train_final.mean()))
        )
        
        # Save the model
        print("\nSaving model to disk...")
        pickle.dump(xgb_model, open(model_path, "wb"))

except Exception as e:
    print(f"Error during model operations: {str(e)}")
    print("Model configuration details:")
    raise

xgb_test_predictions = xgb_model.predict(X_test_scaled)

# Transform predictions back to original scale
y_test_orig = np.expm1(y_test)
xgb_test_pred_orig = np.expm1(xgb_test_predictions)

# Print metrics
print("\nTest Set Metrics:")
print(f"XGBoost R2: {r2_score(y_test_orig, xgb_test_pred_orig):.4f}")
print(f"XGBoost RMSE: ${np.sqrt(mean_squared_error(y_test_orig, xgb_test_pred_orig)):,.2f}")
print(f"XGBoost MAE: ${mean_absolute_error(y_test_orig, xgb_test_pred_orig):,.2f}")
# Calculate and print prediction statistics
print("\nPrediction Statistics:")
print(f"Median Price: ${np.median(xgb_test_pred_orig):,.2f}")
print(f"Standard Deviation: ${np.std(xgb_test_pred_orig):,.2f}")
print(f"Mean Price: ${np.mean(xgb_test_pred_orig):,.2f}")

print("\nActual Statistics:")
print(f"Median Price: ${np.median(y_test_orig):,.2f}")
print(f"Standard Deviation: ${np.std(y_test_orig):,.2f}")
print(f"Mean Price: ${np.mean(y_test_orig):,.2f}")
# Define currency formatter for plots
def currency_formatter(x, p):
    return f"${x:,.0f}"
formatter = FuncFormatter(currency_formatter)


# Calculate ROI metrics using actual predictions
roi_metrics = calculate_roi_metrics(xgb_test_pred_orig)

print_house_statistics(df)
plot_analysis(df, xgb_model, X, X_test_scaled, y_test_orig, xgb_test_pred_orig)

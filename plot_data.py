import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from roi_metrics import calculate_roi_metrics
import regex as re
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
def plot_analysis(df, xgb_model, X, X_test_scaled, y_test_orig, xgb_test_pred_orig):
    
    plt.rcParams['font.size'] = 8  # Reduce font size
    try:
        # Global styling
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.autolayout'] = True
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12

        if df is None or df.empty:
            raise ValueError("No data available for plotting")
        
        # ===== Plot 1: Learning Curves =====
        fig1 = plt.figure(figsize=(10, 8))
        try:
            eval_log = xgb_model.best_estimator_.evals_result()
            val_rmse = eval_log['validation_1']['rmse']
            val_mae = eval_log['validation_1']['mae']
            iterations = range(len(val_rmse))
            
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            
            # Plot RMSE
            ax1.plot(iterations, val_rmse, label='Val RMSE', color='#e74c3c', linewidth=2, linestyle='-')
            
            # Plot MAE
            ax2.plot(iterations, val_mae, label='Val MAE', color='#e74c3c', linewidth=2, linestyle='--')
            
            # Ensure both axes share the same scale
            combined_min = min(min(val_rmse), min(val_rmse), min(val_mae), min(val_mae))
            combined_max = max(max(val_rmse), max(val_rmse), max(val_mae), max(val_mae))
            ax1.set_ylim(combined_min, combined_max)
            ax2.set_ylim(combined_min, combined_max)
            
            ax1.set_xlabel('Boosting Iterations', fontsize=14)
            ax1.set_ylabel('RMSE / MAE', fontsize=14)
            ax2.set_ylabel('RMSE / MAE', fontsize=14)
            plt.title('XGBoost Learning Curves', pad=20, fontsize=16)
            
            # Combine legends for both axes
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize=12)
        except Exception as e:
            warnings.warn(f"Could not plot learning curves: {str(e)}")
        plt.show()

        # ===== Plot 2: Predictions vs Actual =====
        fig2 = plt.figure(figsize=(10, 8))
        try:
            plt.scatter(y_test_orig, xgb_test_pred_orig, alpha=0.5, color='#2ecc71', s=100, label='Predictions')
            plt.plot([y_test_orig.min(), y_test_orig.max()], 
                     [y_test_orig.min(), y_test_orig.max()], 
                     'r--', lw=2, label='Perfect Prediction')
            plt.title('XGBoost: Actual vs Predicted Values', pad=20)
            plt.xlabel('Actual Price ($)')
            plt.ylabel('Predicted Price ($)')
            plt.legend()
        except Exception as e:
            warnings.warn(f"Could not plot predictions vs actual: {str(e)}")
        plt.show()
        
        # ===== Plot 3: Residuals =====
        fig3 = plt.figure(figsize=(10, 8))
        try:
            residuals_xgb = y_test_orig - xgb_test_pred_orig
            plt.scatter(xgb_test_pred_orig, residuals_xgb, alpha=0.5, color='#2ecc71', s=100)
            plt.axhline(y=0, color='r', linestyle='--', label='Zero Residual')
            plt.title('XGBoost: Residual Analysis', pad=20)
            plt.xlabel('Predicted Price ($)')
            plt.ylabel('Residual ($)')
            plt.legend()
        except Exception as e:
            warnings.warn(f"Could not plot residuals: {str(e)}")
        plt.show()
        
        # ===== Plot 4: Feature Importance =====
        fig4 = plt.figure(figsize=(10, 8))
        try:
            explainer = shap.TreeExplainer(xgb_model.best_estimator_)
            shap_values = explainer.shap_values(X_test_scaled)
            shap_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': np.abs(shap_values).mean(0)
            }).sort_values('importance', ascending=True).tail(10)
            
            sns.barplot(data=shap_importance, x='importance', y='feature', palette=['#2ecc71'])
            plt.title('SHAP: Top 10 Important Features', pad=20)
            plt.xlabel('Mean |SHAP Value|')
            plt.ylabel('Features')
        except Exception as e:
            warnings.warn(f"Could not plot feature importance: {str(e)}")
        plt.show()

        # ===== Plot 5: ROI Analysis =====

        fig5 = plt.figure(figsize=(12, 8))
        try:
            # Calculate ROI metrics using all predictions
            roi_metrics = calculate_roi_metrics(xgb_test_pred_orig)
            
            # Plot future value and ROI on the same graph
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            
            # Future Value (Green Line)
            
            ax1.plot(
                roi_metrics["year"], 
                roi_metrics["mean_future_value"], 
                color="#27ae60", 
                label="Future Value", 
                linewidth=2
            )
            ax1.set_ylabel("Future Value (in millions of $)", fontsize=14)
            
            # ROI (Red Dashed Line)
            ax2.plot(
                roi_metrics["year"], 
                roi_metrics["mean_roi"], 
                color="#e74c3c", 
                linestyle="--", 
                label="ROI (%)", 
                linewidth=2
            )
            ax2.set_ylabel("ROI (%)", fontsize=14)
            
            # Formatting
            ax1.set_xlabel("Years", fontsize=14)
            plt.title("10-Year ROI Projection (5.25% Annual Appreciation)", pad=20, fontsize=16)
            
            # Combine legends
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            plt.legend(lines + lines2, labels + labels2, fontsize=12)
            
            plt.grid(True, alpha=0.3)
        except Exception as e:
            warnings.warn(f"Could not plot ROI projection: {str(e)}")
        plt.show()

        # ===== Plot 6: 5-Year Future Price Distribution =====
        def format_price(x, p):
            if x >= 1000000:
                return f'${x/1000000:.1f}M'
            return f'${x/1000:.0f}K'

        # ===== Plot 6: 5-Year Future Price Distribution =====
        fig6 = plt.figure(figsize=(12, 8))
        try:
            # Calculate current and future values with non-negative constraint
            appreciation_rate = 0.0525
            years = 5
            current_values = np.clip(xgb_test_pred_orig, 0, None)  # Clip at minimum 0
            future_values = np.clip(current_values * (1 + appreciation_rate) ** years, 0, None)
            
            # Calculate statistics
            current_median = np.median(current_values)
            future_median = np.median(future_values)
            roi_percentage = ((future_median - current_median) / current_median) * 100
            
            # Create KDE plots
            sns.kdeplot(data=current_values, color='blue', linewidth=2, 
                    label='Current Prices')
            sns.kdeplot(data=future_values, color='green', linewidth=2, 
                    label='5-Year Projected Prices')
            plt.xlim(left=0)

            # Add reference lines with formatted prices
            plt.axvline(current_median, color='blue', linestyle='--', 
                       label=f'Current Median: {format_price(current_median, None)}')
            plt.axvline(future_median, color='green', linestyle='--', 
                       label=f'Future Median: {format_price(future_median, None)}\n(ROI: {roi_percentage:.1f}%)')

            plt.title('Current vs Projected Home Prices (5-Year Horizon)', 
                     fontsize=16, pad=20)
            plt.xlabel('Price', fontsize=12)
            plt.ylabel('Density', fontsize=12)
            
            # Format axis with custom formatter
            plt.gca().xaxis.set_major_formatter(FuncFormatter(format_price))
            
            # Adjust legend
            plt.legend(title='Price Distribution', 
                      title_fontsize=12,
                      fontsize=10,
                      bbox_to_anchor=(1.05, 1),
                      loc='upper left')
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
        except Exception as e:
            warnings.warn(f"Could not plot price distribution: {str(e)}")
        plt.show()

        try:
            # Clean numeric data for additional plots
            df['listPrice'] = pd.to_numeric(df['listPrice'], errors='coerce')
            df['sqft'] = pd.to_numeric(df['sqft'], errors='coerce')
            df['beds'] = pd.to_numeric(df['beds'], errors='coerce')
            df['baths'] = pd.to_numeric(df['baths'], errors='coerce')
            df['year_built'] = pd.to_numeric(df['year_built'], errors='coerce')

            # Plot 7: Price vs Sqft Scatter
            fig7 = plt.figure(figsize=(12, 8))
            plt.scatter(df['sqft'], df['listPrice'], alpha=0.5)
            z = np.polyfit(df['sqft'].dropna(), df['listPrice'].dropna(), 1)
            p = np.poly1d(z)
            plt.plot(df['sqft'].dropna(), p(df['sqft'].dropna()), "r--", alpha=0.8)
            plt.title('Price vs Square Footage', fontsize=16)
            plt.xlabel('Square Footage', fontsize=12)
            plt.ylabel('List Price ($)', fontsize=12)
            plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            warnings.warn(f"Error generating plots: {str(e)}")
            plt.close('all')
        try:
        # Plot 8: Price Distribution by Property Type
            fig8 = plt.figure(figsize=(12, 8))
            if 'sub_type' in df.columns:
                type_col = 'sub_type'
            elif 'type' in df.columns:
                type_col = 'type'
            else:
                raise ValueError("No property type column found")
                
            sns.boxplot(x=type_col, y='listPrice', data=df.dropna(subset=[type_col, 'listPrice']))
            plt.xticks(rotation=45, ha='right')
            plt.title('Price Distribution by Property Type', fontsize=16)
            plt.xlabel('Property Type', fontsize=12)
            plt.ylabel('List Price ($)', fontsize=12)
            plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            warnings.warn(f"Error generating plots: {str(e)}")
            plt.close('all')
        try:
            # Check if location needs to be extracted
            if 'location' not in df.columns:
                # Extract city from URL with error handling
                df['location'] = df['url'].apply(lambda x: re.search(r'_([^_]+)_TX_', str(x)).group(1).replace('-', ' '))
            
            # Encode location values
            location_encoder = LabelEncoder()
            df['location_encoded'] = location_encoder.fit_transform(df['location'])
            
            # Plot 9: Correlation Matrix with Location
            fig9 = plt.figure(figsize=(12, 8))
            numeric_cols = ['listPrice', 'sqft', 'beds', 'baths', 'year_built', 'location_encoded']
            
            # Create and display correlation matrix
            correlation_matrix = df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, 
                        annot=True, 
                        cmap='coolwarm', 
                        center=0,
                        fmt='.2f')
            
            # Update labels
            labels = correlation_matrix.columns.tolist()
            labels[labels.index('location_encoded')] = 'location'
            plt.xticks(range(len(labels)), labels, rotation=45)
            plt.yticks(range(len(labels)), labels, rotation=0)
            
            plt.title('Feature Correlation Matrix', fontsize=16)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error details: {str(e)}")
            print(f"Available columns: {df.columns.tolist()}")
            warnings.warn(f"Error generating correlation matrix: {str(e)}")
            plt.close('all')
        """ try:
            # Plot 10: Price vs Bedrooms
            fig10 = plt.figure(figsize=(12, 8))
            plt.scatter(df['beds'], df['listPrice'], alpha=0.5)
            z = np.polyfit(df['beds'].dropna(), df['listPrice'].dropna(), 1)
            p = np.poly1d(z)
            r2 = np.corrcoef(df['beds'].dropna(), df['listPrice'].dropna())[0,1]**2
            plt.plot(df['beds'].dropna(), p(df['beds'].dropna()), "r--", alpha=0.8)
            plt.title(f'Price vs Bedrooms', fontsize=16)
            plt.xlabel('Number of Bedrooms', fontsize=12)
            plt.ylabel('List Price ($)', fontsize=12)
            plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            warnings.warn(f"Error generating plots: {str(e)}")
            plt.close('all')
        try:
            # Plot 11: Price vs Bathrooms
            fig11 = plt.figure(figsize=(12, 8))
            bath_data = df.dropna(subset=['baths', 'listPrice'])
            plt.scatter(bath_data['baths'], bath_data['listPrice'], alpha=0.5)
            z = np.polyfit(bath_data['baths'], bath_data['listPrice'], 1)
            p = np.poly1d(z)
            r2 = np.corrcoef(bath_data['baths'], bath_data['listPrice'])[0,1]**2
            plt.plot(bath_data['baths'], p(bath_data['baths']), "r--", alpha=0.8)
            plt.title(f'Price vs Bathrooms', fontsize=16)
            plt.xlabel('Number of Bathrooms', fontsize=12)
            plt.ylabel('List Price ($)', fontsize=12)
            plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            warnings.warn(f"Error generating plots: {str(e)}")
            plt.close('all')   
        try:
            # Plot 12: Price vs Year Built
            fig12 = plt.figure(figsize=(12, 8))
            year_data = df.dropna(subset=['year_built', 'listPrice'])
            plt.scatter(year_data['year_built'], year_data['listPrice'], alpha=0.5)
            z = np.polyfit(year_data['year_built'], year_data['listPrice'], 1)
            p = np.poly1d(z)
            r2 = np.corrcoef(year_data['year_built'], year_data['listPrice'])[0,1]**2
            plt.plot(year_data['year_built'], p(year_data['year_built']), "r--", alpha=0.8)
            plt.title(f'Price vs Year Built', fontsize=16)
            plt.xlabel('Year Built', fontsize=12)
            plt.ylabel('List Price ($)', fontsize=12)
            plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            warnings.warn(f"Error generating plots: {str(e)}")
            plt.close('all')
        # Plot 13: Property Type Distribution"""
        # Plot 13: Property Type Distribution
        try:
            # Get property type columns
            type_columns = [col for col in df.columns if col.startswith('type_')]
            
            # Create property type counts
            type_counts = {}
            for col in type_columns:
                count = df[col].sum()
                if count > 0:
                    type_name = col.replace('type_', '').replace('_', ' ').title()
                    type_counts[type_name] = count
            
            # Create bar plot
            fig13 = plt.figure(figsize=(12, 8))
            plt.bar(type_counts.keys(), type_counts.values(), color='skyblue')
            
            # Add value labels
            for i, (type_name, count) in enumerate(type_counts.items()):
                plt.text(i, count, str(int(count)), 
                        ha='center', va='bottom',
                        fontsize=10)
            
            # Customize plot
            plt.title('Distribution of Property Types in Texas', fontsize=16, pad=20)
            plt.xlabel('Property Type', fontsize=12)
            plt.ylabel('Number of Properties', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')
            
            # Print statistics
            total = sum(type_counts.values())
            print("\nProperty Type Distribution:")
            for prop_type, count in type_counts.items():
                print(f"{prop_type}: {count} ({(count/total)*100:.1f}%)")
            
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"\nError in property type plot:")
            print(f"Available columns: {df.columns.tolist()}")
            print(f"Error message: {str(e)}")
# Remove the plot_additional_analysis function as it's now integrated
   
    except Exception as e:
        warnings.warn(f"Error generating plots: {str(e)}")
        plt.close('all')  # Clean up any partial plots

# Generate plots
print("\nGenerating analysis plots...")

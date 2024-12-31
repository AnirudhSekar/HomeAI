import pandas as pd
import numpy as np
from typing import Union, Dict
from numpy.typing import ArrayLike
def calculate_roi_metrics(initial_investment, appreciation_rate=0.0525, years=range(5,21)):
    """Calculate ROI metrics and statistics"""

    initial_investment = np.asarray(initial_investment).flatten()
    roi_data = []

    # Go through each initial_investment and years and use the appreciation rate to calculate that year's ROI for that prediction
    for year in years:
        future_values = initial_investment * (1 + appreciation_rate) ** year
        roi_percentages = ((future_values - initial_investment) / initial_investment) * 100

        roi_data.append({
            'year': year,
            'future_values': future_values.tolist(),
            'roi_percentages': roi_percentages.tolist(),
            'mean_future_value': float(np.mean(future_values)),
            'median_future_value': float(np.median(future_values)),
            'std_future_value': float(np.std(future_values)),
            'mean_roi': float(np.mean(roi_percentages)),
            'median_roi': float(np.median(roi_percentages)),
        })


    return pd.DataFrame(roi_data)
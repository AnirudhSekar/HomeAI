import pandas as pd

def print_house_statistics(df):
    """Calculate and print key statistics for house features"""
    # Convert columns to numeric, handling errors
    df['listPrice'] = pd.to_numeric(df['listPrice'], errors='coerce')
    df['sqft'] = pd.to_numeric(df['sqft'], errors='coerce')
    df['beds'] = pd.to_numeric(df['beds'], errors='coerce')
    df['baths'] = pd.to_numeric(df['baths'], errors='coerce')
    df['year_built'] = pd.to_numeric(df['year_built'], errors='coerce')

    # Calculate statistics for each column
    stats = {
        'List Price': {
            'mean': df['listPrice'].mean(),
            'median': df['listPrice'].median(),
            'std': df['listPrice'].std()
        },
        'Square Footage': {
            'mean': df['sqft'].mean(),
            'median': df['sqft'].median(),
            'std': df['sqft'].std()
        },
        'Bedrooms': {
            'mean': df['beds'].mean(),
            'median': df['beds'].median(),
            'std': df['beds'].std()
        },
        'Bathrooms': {
            'mean': df['baths'].mean(),
            'median': df['baths'].median(),
            'std': df['baths'].std()
        },
        'Year Built': {
            'mean': df['year_built'].mean(),
            'median': df['year_built'].median(),
            'std': df['year_built'].std()
        }
    }

    # Print formatted statistics
    print("\nHouse Statistics Summary:")
    print("-" * 60)
    print(f"{'Feature':<15} {'Mean':>12} {'Median':>12} {'Std Dev':>12}")
    print("-" * 60)

    for feature, values in stats.items():
        if feature == 'List Price':
            mean = f"${values['mean']:,.0f}"
            median = f"${values['median']:,.0f}"
            std = f"${values['std']:,.0f}"
        else:
            mean = f"{values['mean']:.2f}"
            median = f"{values['median']:.2f}"
            std = f"{values['std']:.2f}"
        
        print(f"{feature:<15} {mean:>12} {median:>12} {std:>12}")
    print("-" * 60)

# Call function

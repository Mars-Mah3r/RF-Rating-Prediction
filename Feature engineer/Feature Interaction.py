import pandas as pd

# File paths
input_file = '/Users/mars/Documents/01-Applications/14-Moodys Rating/3rd Round/Data/6_Top_95_features.csv'
output_file = '/Users/mars/Documents/01-Applications/14-Moodys Rating/3rd Round/Data/7_Final_Dataset.csv'


data = pd.read_csv(input_file)

# Create interaction features for Profitability
data['ROCE_NetProfitMargin'] = data['returnOnCapitalEmployed'] * data['netProfitMargin']
data['NetProfitMargin_OpCashFlow'] = data['netProfitMargin'] * data['operatingCashFlowPerShare']

# Create interaction features for Liquidity
data['CashRatio_OpCashFlow'] = data['cashRatio'] * data['operatingCashFlowPerShare']
data['CashRatio_NetProfitMargin'] = data['cashRatio'] * data['netProfitMargin']

# Create interaction features for Leverage
data['DebtRatio_ROCE'] = data['debtRatio'] * data['returnOnCapitalEmployed']
data['DebtRatio_OpCashFlow'] = data['debtRatio'] * data['operatingCashFlowPerShare']

# Save the updated dataset
data.to_csv(output_file, index=False)

print(f"Updated dataset with new interaction features saved to {output_file}")

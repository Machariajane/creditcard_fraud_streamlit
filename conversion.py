import pandas as pd
df = pd.read_csv('creditcard.csv')
df.to_parquet('creditcard.parquet')

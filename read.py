import numpy as np
import pandas as pd
from IPython import get_ipython
from IPython.display import display

# Replace with the actual path to your CSV file
file_path = 'C:/Users/HP/Desktop/new test/new test/Fake.csv'  

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# Now you can work with the data in the DataFrame 'df'
# For example, you can display the first few rows:
print(df.head())

# Or, you can perform other operations like data analysis, cleaning, etc.
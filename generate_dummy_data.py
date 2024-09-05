import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

def generate_dummy_data():
    # Generate random mass functions
    focal_elements = ['A', 'B', 'C', 'AB', 'AC', 'BC', 'ABC']
    num_mass_functions = 5
    
    # Create a dictionary to store the data
    data = {'Focal Element': focal_elements}
    
    for i in range(num_mass_functions):
        masses = np.random.dirichlet(np.ones(len(focal_elements)), size=1)[0]
        data[f'm{i+1}'] = masses
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure sum of masses equals 1 for each mass function
    for col in df.columns[1:]:
        df[col] = df[col] / df[col].sum()
    
    # Round values to 4 decimal places
    df = df.round(4)
    
    return df

def save_to_excel(df, filename='input_data.xlsx'):
    wb = Workbook()
    ws = wb.active
    ws.title = "Mass Functions"
    
    for r in dataframe_to_rows(df, index=False, header=True):
        ws.append(r)
    
    wb.save(filename)
    print(f"Data saved to {filename}")

# Generate and save dummy data
df = generate_dummy_data()
save_to_excel(df)

print("Dummy data for belief function analysis:")
print(df)
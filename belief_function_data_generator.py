import numpy as np
import pandas as pd
import json
import argparse

def generate_example_data(num_sources=3, num_focal_elements=5):
    focal_elements = ['A', 'B', 'C', 'D', 'E', 'AB', 'AC', 'BC', 'ABC', 'ABCD', 'ABCDE'][:num_focal_elements]
    data = {}
    for i in range(num_sources):
        masses = np.random.dirichlet(np.ones(num_focal_elements), size=1)[0]
        data[f'Source_{i+1}'] = dict(zip(focal_elements, masses))
    return data

def save_example_data(data, filename):
    if filename.endswith('.csv'):
        df = pd.DataFrame(data)
        df.to_csv(filename)
    elif filename.endswith('.xlsx'):
        df = pd.DataFrame(data)
        df.to_excel(filename)
    elif filename.endswith('.json'):
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    else:
        raise ValueError("Unsupported file format. Use .csv, .xlsx, or .json")

def main():
    parser = argparse.ArgumentParser(description="Generate example data for Belief Function Analysis")
    parser.add_argument("--sources", type=int, default=3, help="Number of sources")
    parser.add_argument("--elements", type=int, default=5, help="Number of focal elements")
    parser.add_argument("--output", required=True, help="Output file path (CSV, Excel, or JSON)")
    args = parser.parse_args()

    data = generate_example_data(args.sources, args.elements)
    save_example_data(data, args.output)
    print(f"Example data saved to {args.output}")

if __name__ == "__main__":
    main()
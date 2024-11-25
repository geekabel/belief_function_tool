import tkinter as tk
from tkinter import filedialog, ttk, messagebox, simpledialog
import pandas as pd
import numpy as np
import json
import argparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from functools import reduce
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import itertools

class BeliefFunctionTool:
    def __init__(self):
        self.rules = {
            'Dempster': self.dempster_rule,
            'Yager': self.yager_rule,
            'Dubois_Prade': self.dubois_prade_rule,
            'PCR5': self.pcr5_rule,
            'PCR6': self.pcr6_rule
        }
    
    def dempster_rule(self, *mass_functions):
        def combine_two(m1, m2):
            result = {}
            k = 0
            for A in m1:
                for B in m2:
                    if set(A) & set(B):
                        intersection = ''.join(sorted(set(A) & set(B)))
                        result[intersection] = result.get(intersection, 0) + m1[A] * m2[B]
                    else:
                        k += m1[A] * m2[B]
            
            if k == 1:
                raise ValueError("Complete conflict in evidence")
            
            for A in result:
                result[A] /= (1 - k)
            
            return result
        
        return reduce(combine_two, mass_functions)

    def yager_rule(self, *mass_functions):
        def combine_two(m1, m2):
            result = {}
            for A in m1:
                for B in m2:
                    intersection = ''.join(sorted(set(A) & set(B)))
                    if intersection:
                        result[intersection] = result.get(intersection, 0) + m1[A] * m2[B]
                    else:
                        result['Ω'] = result.get('Ω', 0) + m1[A] * m2[B]
            return result
        
        return reduce(combine_two, mass_functions)

    def dubois_prade_rule(self, *mass_functions):
        def combine_two(m1, m2):
            result = {}
            for A in m1:
                for B in m2:
                    intersection = ''.join(sorted(set(A) & set(B)))
                    union = ''.join(sorted(set(A) | set(B)))
                    if intersection:
                        result[intersection] = result.get(intersection, 0) + m1[A] * m2[B]
                    else:
                        result[union] = result.get(union, 0) + m1[A] * m2[B]
            return result
        
        return reduce(combine_two, mass_functions)

    def pcr5_rule(self, *mass_functions):
        def combine_two(m1, m2):
            result = {}
            for A in m1:
                for B in m2:
                    if set(A) & set(B):
                        intersection = ''.join(sorted(set(A) & set(B)))
                        result[intersection] = result.get(intersection, 0) + m1[A] * m2[B]
                    else:
                        factor = m1[A] + m2[B]
                        if factor != 0:
                            result[A] = result.get(A, 0) + (m1[A]**2 * m2[B]) / factor
                            result[B] = result.get(B, 0) + (m2[B]**2 * m1[A]) / factor
            return result
        
        return reduce(combine_two, mass_functions)

    def pcr6_rule(self, *mass_functions):
        def combine_two(m1, m2):
            result = {}
            for A in m1:
                for B in m2:
                    if set(A) & set(B):
                        intersection = ''.join(sorted(set(A) & set(B)))
                        result[intersection] = result.get(intersection, 0) + m1[A] * m2[B]
                    else:
                        total = m1[A] * m2[B]
                        result[A] = result.get(A, 0) + total * m1[A] / (m1[A] + m2[B])
                        result[B] = result.get(B, 0) + total * m2[B] / (m1[A] + m2[B])
            return result
        
        return reduce(combine_two, mass_functions)

    def apply_rule(self, rule, *mass_functions):
        if rule not in self.rules:
            raise ValueError(f"Unknown rule: {rule}")
        return self.rules[rule](*mass_functions)

    def conditioning(self, m, condition):
        conditioned_m = {}
        normalization_factor = sum(m[A] for A in m if set(A).issubset(set(condition)))
        
        if normalization_factor == 0:
            raise ValueError(f"Conditioning on {condition} is not possible as it has zero mass")
        
        for A in m:
            if set(A).issubset(set(condition)):
                conditioned_m[A] = m[A] / normalization_factor
        
        return conditioned_m

    def marginalization(self, m, keep_variables):
        marginalized_m = {}
        for A in m:
            new_A = ''.join(sorted(set(A) & keep_variables))
            marginalized_m[new_A] = marginalized_m.get(new_A, 0) + m[A]
        return marginalized_m

    def calculate_belief(self, m):
        belief = {}
        for A in m:
            belief[A] = sum(m[B] for B in m if set(B).issubset(set(A)))
        return belief

    def calculate_plausibility(self, m):
        plausibility = {}
        for A in m:
            plausibility[A] = sum(m[B] for B in m if set(A) & set(B))
        return plausibility

    def load_data_from_excel(self, filename):
        df = pd.read_excel(filename, index_col=0)
        return {col: df[col].to_dict() for col in df.columns}

    def load_data_from_json(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
    
    def load_data_from_csv(self, filename):
        df = pd.read_csv(filename, index_col=0)
        return {col: df[col].to_dict() for col in df.columns}

    def visualize_mass_functions(self, mass_functions):
        fig, ax = plt.subplots(figsize=(10, 6))
        x = list(set().union(*[m.keys() for m in mass_functions]))
        width = 0.8 / len(mass_functions)
        
        for i, m in enumerate(mass_functions):
            values = [m.get(key, 0) for key in x]
            ax.bar([i + width * i for i in range(len(x))], values, width, label=f'Source {i+1}')
        
        ax.set_xlabel('Focal Elements')
        ax.set_ylabel('Mass')
        ax.set_title('Comparison of Mass Functions from Different Sources')
        ax.set_xticks([i + width * (len(mass_functions) - 1) / 2 for i in range(len(x))])
        ax.set_xticklabels(x, rotation=45, ha='right')
        ax.legend()
        plt.tight_layout()
        return fig

    def get_calculation_details(self, rule, *mass_functions):
        details = f"Calculation details for {rule} rule:\n\n"
        
        if rule == 'Dempster':
            details += self._get_dempster_details(*mass_functions)
        elif rule == 'Yager':
            details += self._get_yager_details(*mass_functions)
        elif rule == 'Dubois_Prade':
            details += self._get_dubois_prade_details(*mass_functions)
        elif rule == 'PCR5':
            details += self._get_pcr5_details(*mass_functions)
        elif rule == 'PCR6':
            details += self._get_pcr6_details(*mass_functions)
        
        return details

    def _get_dempster_details(self, *mass_functions):
        details = ""
        result = {}
        k = 0
        
        for combination in self._generate_combinations(*mass_functions):
            intersection = ''.join(sorted(set.intersection(*map(set, combination))))
            product = np.prod([m[f] for m, f in zip(mass_functions, combination)])
            
            if intersection:
                result[intersection] = result.get(intersection, 0) + product
                details += f"{' ∩ '.join(combination)} = {intersection}, mass = {product:.4f}\n"
            else:
                k += product
                details += f"{' ∩ '.join(combination)} = ∅, conflict mass = {product:.4f}\n"
        
        details += f"\nTotal conflict (k): {k:.4f}\n"
        details += "Normalization factor: 1 / (1 - k) = {:.4f}\n\n".format(1 / (1 - k))
        
        for A in result:
            result[A] /= (1 - k)
            details += f"Normalized mass for {A}: {result[A]:.4f}\n"
        
        return details

    def _get_yager_details(self, *mass_functions):
        details = ""
        result = {}
        
        for combination in self._generate_combinations(*mass_functions):
            intersection = ''.join(sorted(set.intersection(*map(set, combination))))
            product = np.prod([m[f] for m, f in zip(mass_functions, combination)])
            
            if intersection:
                result[intersection] = result.get(intersection, 0) + product
                details += f"{' ∩ '.join(combination)} = {intersection}, mass = {product:.4f}\n"
            else:
                result['Ω'] = result.get('Ω', 0) + product
                details += f"{' ∩ '.join(combination)} = ∅, mass added to Ω = {product:.4f}\n"
        
        details += "\nFinal mass assignments:\n"
        for A in result:
            details += f"m({A}) = {result[A]:.4f}\n"
        
        return details

    def _get_dubois_prade_details(self, *mass_functions):
        details = ""
        result = {}
        
        for combination in self._generate_combinations(*mass_functions):
            intersection = ''.join(sorted(set.intersection(*map(set, combination))))
            union = ''.join(sorted(set.union(*map(set, combination))))
            product = np.prod([m[f] for m, f in zip(mass_functions, combination)])
            
            if intersection:
                result[intersection] = result.get(intersection, 0) + product
                details += f"{' ∩ '.join(combination)} = {intersection}, mass = {product:.4f}\n"
            else:
                result[union] = result.get(union, 0) + product
                details += f"{' ∩ '.join(combination)} = ∅, mass added to {union} = {product:.4f}\n"
        
        details += "\nFinal mass assignments:\n"
        for A in result:
            details += f"m({A}) = {result[A]:.4f}\n"
        
        return details

    def _get_pcr5_details(self, *mass_functions):
        details = ""
        result = {}
        
        for combination in self._generate_combinations(*mass_functions):
            intersection = ''.join(sorted(set.intersection(*map(set, combination))))
            product = np.prod([m[f] for m, f in zip(mass_functions, combination)])
            
            if intersection:
                result[intersection] = result.get(intersection, 0) + product
                details += f"{' ∩ '.join(combination)} = {intersection}, mass = {product:.4f}\n"
            else:
                details += f"{' ∩ '.join(combination)} = ∅, redistributing mass:\n"
                factor = sum(m[f] for m, f in zip(mass_functions, combination))
                for m, f in zip(mass_functions, combination):
                    proportional_mass = (m[f]**2 * product) / factor
                    result[f] = result.get(f, 0) + proportional_mass
                    details += f"  m({f}) += {proportional_mass:.4f}\n"
        
        details += "\nFinal mass assignments:\n"
        for A in result:
            details += f"m({A}) = {result[A]:.4f}\n"
        
        return details

    def _get_pcr6_details(self, *mass_functions):
        details = ""
        result = {}
        
        for combination in self._generate_combinations(*mass_functions):
            intersection = ''.join(sorted(set.intersection(*map(set, combination))))
            product = np.prod([m[f] for m, f in zip(mass_functions, combination)])
            
            if intersection:
                result[intersection] = result.get(intersection, 0) + product
                details += f"{' ∩ '.join(combination)} = {intersection}, mass = {product:.4f}\n"
            else:
                details += f"{' ∩ '.join(combination)} = ∅, redistributing mass:\n"
                total = sum(m[f] for m, f in zip(mass_functions, combination))
                for m, f in zip(mass_functions, combination):
                    proportional_mass = product * m[f] / total
                    result[f] = result.get(f, 0) + proportional_mass
                    details += f"  m({f}) += {proportional_mass:.4f}\n"
        
        details += "\nFinal mass assignments:\n"
        for A in result:
            details += f"m({A}) = {result[A]:.4f}\n"
        
        return details

    def _generate_combinations(self, *mass_functions):
        return itertools.product(*[m.keys() for m in mass_functions])

class BeliefFunctionGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Multi-Source Belief Function Analysis Tool")
        self.tool = BeliefFunctionTool()
        
        self.mass_functions = []
        self.data_types = []
        
        self.create_widgets()
    
    def create_widgets(self):
        self.frame = ttk.Frame(self.master, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        ttk.Button(self.frame, text="Load Mass Function", command=self.load_mass_function).grid(column=0, row=0, pady=5)
        
        self.rule_var = tk.StringVar(value="Dempster")
        ttk.Label(self.frame, text="Select Rule:").grid(column=0, row=1, pady=5)
        for i, rule in enumerate(self.tool.rules):
            ttk.Radiobutton(self.frame, text=rule, variable=self.rule_var, value=rule).grid(column=0, row=i+2, sticky=tk.W)
        
        ttk.Button(self.frame, text="Analyze", command=self.analyze).grid(column=0, row=7, pady=5)
        ttk.Button(self.frame, text="Conditioning", command=self.perform_conditioning).grid(column=0, row=8, pady=5)
        ttk.Button(self.frame, text="Marginalization", command=self.perform_marginalization).grid(column=0, row=9, pady=5)
        ttk.Button(self.frame, text="Calculate Plausibility", command=self.calculate_plausibility).grid(column=0, row=10, pady=5)
        # ttk.Button(self.frame, text="Calculate Belief",command=self.calculate_belief)
        
        self.result_text = tk.Text(self.frame, height=10, width=50)
        self.result_text.grid(column=1, row=0, rowspan=6, padx=10)
        
        self.calculation_text = tk.Text(self.frame, height=20, width=50)
        self.calculation_text.grid(column=1, row=6, rowspan=5, padx=10, pady=10)
        
        self.canvas_frame = ttk.Frame(self.master)
        self.canvas_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Data type selection
        self.data_type_var = tk.StringVar(value="numeric")
        ttk.Label(self.frame, text="Data Type:").grid(column=0, row=11, pady=5)
        ttk.Radiobutton(self.frame, text="Numeric", variable=self.data_type_var, value="numeric").grid(column=0, row=12, sticky=tk.W)
        ttk.Radiobutton(self.frame, text="Linguistic", variable=self.data_type_var, value="linguistic").grid(column=0, row=13, sticky=tk.W)
        ttk.Radiobutton(self.frame, text="Interval", variable=self.data_type_var, value="interval").grid(column=0, row=14, sticky=tk.W)
    
    def load_mass_function(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("JSON files", "*.json")])
        if filename:
            try:
                if filename.endswith('.csv'):
                    new_mass_functions = self.tool.load_data_from_csv(filename)
                elif filename.endswith('.xlsx'):
                    new_mass_functions = self.tool.load_data_from_excel(filename)
                elif filename.endswith('.json'):
                    new_mass_functions = self.tool.load_data_from_json(filename)
                else:
                    raise ValueError("Unsupported file format")
                
                for name, mass_function in new_mass_functions.items():
                    self.mass_functions.append(mass_function)
                    self.data_types.append(self.data_type_var.get())
                messagebox.showinfo("File Loaded", f"Mass functions from {filename} loaded successfully")
                self.update_visualization()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")

    
    def analyze(self):
        if len(self.mass_functions) < 2:
            messagebox.showerror("Error", "Please load at least two mass functions before analyzing")
            return
        
        try:
            result = self.tool.apply_rule(self.rule_var.get(), *self.mass_functions)
            
            self.result_text.delete('1.0', tk.END)
            self.result_text.insert(tk.END, f"Result using {self.rule_var.get()} rule:\n{result}\n")
            
            calculation_details = self.tool.get_calculation_details(self.rule_var.get(), *self.mass_functions)
            self.calculation_text.delete('1.0', tk.END)
            self.calculation_text.insert(tk.END, calculation_details)
            
            self.update_visualization(result)
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
    
    def perform_conditioning(self):
        if not self.mass_functions:
            messagebox.showerror("Error", "Please load at least one mass function first")
            return
        
        condition = simpledialog.askstring("Conditioning", "Enter the conditioning event (e.g., 'A' or 'AB'):")
        if condition:
            try:
                conditioned = self.tool.conditioning(self.mass_functions[0], condition)
                self.result_text.delete('1.0', tk.END)
                self.result_text.insert(tk.END, f"Conditioned mass function on {condition}:\n{conditioned}\n")
                self.update_visualization(conditioned)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to perform conditioning: {str(e)}")
    
    def perform_marginalization(self):
        if not self.mass_functions:
            messagebox.showerror("Error", "Please load at least one mass function first")
            return
        
        keep_vars = simpledialog.askstring("Marginalization", "Enter variables to keep (e.g., 'A,B'):")
        if keep_vars:
            try:
                keep_set = set(keep_vars.split(','))
                marginalized = self.tool.marginalization(self.mass_functions[0], keep_set)
                self.result_text.delete('1.0', tk.END)
                self.result_text.insert(tk.END, f"Marginalized mass function keeping {keep_vars}:\n{marginalized}\n")
                self.update_visualization(marginalized)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to perform marginalization: {str(e)}")
    
    def calculate_plausibility(self):
        if not self.mass_functions:
            messagebox.showerror("Error", "Please load at least one mass function first")
            return
        
        plausibility = self.tool.calculate_plausibility(self.mass_functions[0])
        self.result_text.delete('1.0', tk.END)
        self.result_text.insert(tk.END, f"Plausibility function:\n{plausibility}\n")
        self.update_visualization(plausibility)
    
    def update_visualization(self, additional_data=None):
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        
        data_to_visualize = self.mass_functions[:]
        if additional_data:
            data_to_visualize.append(additional_data)
        
        fig = self.tool.visualize_mass_functions(data_to_visualize)
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

def main():
    parser = argparse.ArgumentParser(description="Belief Function Analysis Tool")
    parser.add_argument("--file", help="Input file path (CSV, Excel, or JSON)")
    parser.add_argument("--rule", choices=["Dempster", "Yager", "Dubois_Prade", "PCR5", "PCR6"], default="Dempster", help="Combination rule to use")
    args = parser.parse_args()

    if args.file:
        tool = BeliefFunctionTool()
        try:
            if args.file.endswith('.csv'):
                data = tool.load_data_from_csv(args.file)
            elif args.file.endswith('.xlsx'):
                data = tool.load_data_from_excel(args.file)
            elif args.file.endswith('.json'):
                data = tool.load_data_from_json(args.file)
            else:
                raise ValueError("Unsupported file format")
            
            result = tool.apply_rule(args.rule, *data.values())
            print(f"Result using {args.rule} rule:")
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Error: {str(e)}")
    else:
        root = tk.Tk()
        gui = BeliefFunctionGUI(root)
        root.mainloop()
        
if __name__ == '__main__':
    main()
Belief Function Analysis Tool
=============================

This tool provides a comprehensive platform for analyzing and comparing different belief function rules.
It supports various combination rules, multiple input formats, and offers both console and GUI interfaces.

Key Features
-------------

1. Multiple belief function rules: Dempster, Yager, Dubois-Prade, PCR5, PCR6
2. Input from CSV, JSON, and Excel files
3. Calculation of belief and plausibility functions
4. Visualization using Matplotlib and Plotly (bar charts and heatmaps)
5. Command-line interface and graphical user interface
6. Parallel processing for faster computation with large datasets
7. Unit tests for ensuring correctness

Usage
------

1. Command-line interface:
   python belief_function_tool.py --file input_data.csv --rule Dempster

2. Graphical user interface:
   python belief_function_tool.py

3. As a module in your own Python script:
   from belief_function_tool import BeliefFunctionTool

   tool = BeliefFunctionTool()
   m1 = {'A': 0.4, 'B': 0.3, 'AB': 0.3}
   m2 = {'A': 0.5, 'C': 0.3, 'AC': 0.2}
   result = tool.apply_rule('Dempster', m1, m2)
   print(result)

Example
--------

```py
>>> tool = BeliefFunctionTool()
>>> m1 = {'A': 0.4, 'B': 0.3, 'AB': 0.3}
>>> m2 = {'A': 0.5, 'C': 0.3, 'AC': 0.2}
>>> result = tool.apply_rule('Dempster', m1, m2)
>>> print(result)
{'A': 0.7222222222222222, 'C': 0.1666666666666667, 'AC': 0.1111111111111111}

>>> belief = tool.calculate_belief(m1)
>>> print(belief)
{'A': 0.4, 'B': 0.3, 'AB': 1.0}

>>> plausibility = tool.calculate_plausibility(m1)
>>> print(plausibility)
{'A': 0.7, 'B': 0.6, 'AB': 1.0}
```

# Conditioning
```py
>>>
>>> conditioned = tool.conditioning(m1, 'A')
>>> print(conditioned)
{'A': 1.0}
```
# Marginalization
```py
>>> marginalized = tool.marginalization(m1, {'A'})
>>> print(marginalized)
{'A': 0.7, '': 0.3}
```
# Continuous belief function
```py
>>>
>>> belief = tool.continuous_belief_function(1.0, 0, 1)
>>> print(belief)
0.8413447460685429
```
# Sensitivity analysis
```py
>>>
>>> sensitivity = tool.sensitivity_analysis('Dempster', m1, m2)
>>> print(sensitivity)
{'A': [...], 'C': [...], 'AC': [...]}
```
For more information on belief functions and their applications, refer to:

- Shafer, G. (1976). A Mathematical Theory of Evidence. Princeton University Press.
- Smets, P. (1990). The combination of evidence in the transferable belief model. IEEE Transactions on Pattern Analysis and Machine Intelligence, 12(5), 447-458.

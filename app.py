from flask import Flask, render_template, request, jsonify
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    # tool = BeliefFunctionTool()
    
    m1 = data['m1']
    m2 = data['m2']
    rule = data['rule']
    result =""
    sensitivity =""
    # result = tool.apply_rule(rule, m1, m2)
    # sensitivity = tool.parallel_sensitivity_analysis(rule, m1, m2)
    
    return jsonify({
        'result': result,
        'sensitivity': sensitivity
    })


if __name__ == '__main__':
    app.run(debug=True)
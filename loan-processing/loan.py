from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/evaluate_loan', methods=['POST'])
def evaluate_loan():
    data = request.json
    credit_score = data.get('credit_score')
    credit_history = data.get('credit_history')
    yearly_salary = data.get('yearly_salary')  # Not used in the evaluation
    #net_dollars = data.get('net_dollars')
    loan_amount = data.get('loan_amount')
    
    if credit_score > 600 and credit_history.lower() == 'good' and loan_amount < yearly_salary:
        return jsonify({"approval": "yes"})
    else:
        return jsonify({"approval": "no"})

if __name__ == '__main__':
    app.run(debug=True, port=5200)

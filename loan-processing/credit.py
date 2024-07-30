from flask import Flask, request, jsonify
import random

app = Flask(__name__)

def generate_credit_score():
    score = random.gauss(700, 50)  # Normal distribution with mean 700 and std deviation 50
    score = max(300, min(850, int(score)))  # Clamp the value between 300 and 850
    return score

def determine_credit_history(score):
    if score < 650:
        return "bad"
    else:
        return "good"

@app.route('/credit_check', methods=['POST'])
def credit_check():
    data = request.get_json()
    name = data.get("name")
    ssn = data.get("ssn")
    address = data.get("address")
    dob = data.get("dob")
    
    credit_score = generate_credit_score()
    credit_history = determine_credit_history(credit_score)
    
    response = {
        "credit_score": credit_score,
        "credit_history": credit_history
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5100)


from flask import Flask, render_template, request, jsonify
from qa_model import get_answer

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    question = request.json["question"]
    answer = get_answer(question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
  
    app.run(host='0.0.0.0',port=8080)

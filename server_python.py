from flask import Flask, request, jsonify
from datetime import datetime
from transformers import RobertaTokenizer, RobertaForSequenceClassification

#

# Load the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained(
    "facebook/roberta-hate-speech-dynabench-r4-target")
model = RobertaForSequenceClassification.from_pretrained(
    "facebook/roberta-hate-speech-dynabench-r4-target")


# Function to analyze text for hate speech
def analyze_text(text):
    inputs = tokenizer(text,
                       return_tensors="pt",
                       truncation=True,
                       max_length=512)
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax().item()
    labels = ["neither", "hate", "offensive"]
    if predicted_class in [0, 2]:
        return "no-hate"
    return labels[predicted_class]


app = Flask(__name__)

good_words = ["mohamed"]
bad_words = ["israel","usa"]


def check_manual(text):
    for word in good_words:
        if word in text:
            return 1
    for word in bad_words:

        if word in text:
            return 0
    return None


@app.route("/message", methods=["POST", "GET"])
def check_message():
    try:
        data = request.get_json()
        print("here is the data:", data)
        if not data or "message" not in data:
            return jsonify(
                {"error": "Invalid JSON or missing 'message' field"}), 400

        message_content = data["message"]
        if not isinstance(message_content, str) or not message_content.strip():
            return jsonify({"error":
                            "Message must be a non-empty string"}), 400
        if (check_manual(message_content) is not None
                and check_manual(message_content) == 1):
            response = {"message": message_content, "result": "no-hate"}
            return jsonify(response)
        
        if (check_manual(message_content) is not None
                and check_manual(message_content) == 0):
            print("containsisrael")
            return jsonify({
                "message": "The message contains hate speech",
                "result": "hate"
            })
        result = analyze_text(message_content)
        print("kemelna")
        if analyze_text(message_content) == "hate":
            response = {
                "message": "The message contains hate speech",
                "result": result
            }
        else:
            response = {"message": message_content, "result": result}
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)

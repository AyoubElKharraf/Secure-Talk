from flask import Flask, request, jsonify
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# Initialisation Flask
app = Flask(__name__)

# Chargeur de modèle (version optimisée)
tokenizer = RobertaTokenizer.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target")
model = RobertaForSequenceClassification.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target")
model.eval()  # Mode évaluation pour meilleures performances

# Liste de mots avec système de priorité
WORD_LISTS = {
    "allow": ["mohamed", "islam"],  # Mots toujours autorisés
    "block": ["israel", "usa", "kill","hate"]  # Mots bloquants
}

def check_keywords(text):
    """
    Vérifie les mots-clés avec logique de priorité
    Retourne: "allow", "block" ou None
    """
    text_lower = text.lower()
    for word in WORD_LISTS["allow"]:
        if word in text_lower:
            return "allow"
    for word in WORD_LISTS["block"]:
        if word in text_lower:
            return "block"
    return None

def analyze_with_ai(text):
    """Analyse sémantique avec RoBERTa"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    
    with torch.no_grad():  # Désactive le calcul de gradient pour l'inférence
        outputs = model(**inputs)
    
    predicted_class = outputs.logits.argmax().item()
    return ["no-hate", "hate", "offensive"][predicted_class]

@app.route("/message", methods=["POST"])
def check_message():
    try:
        data = request.get_json()
        
        # Validation des données
        if not data or "message" not in data:
            return jsonify({"error": "Invalid request format"}), 400
            
        message = data["message"].strip()
        if not message:
            return jsonify({"error": "Empty message"}), 400

        # 1. Vérification rapide par mots-clés
        keyword_result = check_keywords(message)
        if keyword_result == "allow":
            return jsonify({
                "message": message,
                "result": "no-hate",
                "method": "keyword_allow"
            })
        elif keyword_result == "block":
            return jsonify({
                "message": "Content blocked",
                "result": "hate",
                "method": "keyword_block"
            }), 403

        # 2. Analyse AI si nécessaire
        ai_result = analyze_with_ai(message)
        if ai_result == "hate":
            return jsonify({
                "message": "This content violates our policy",
                "result": ai_result,
                "method": "ai_detection"
            }), 403
        else:
            return jsonify({
                "message": message,
                "result": ai_result,
                "method": "ai_verified"
            })

    except Exception as e:
        app.logger.error(f"Error processing message: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=False)  # debug=False en production
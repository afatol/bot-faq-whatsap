# app.py - VERSÃO DE TESTE DE CONECTIVIDADE (ECHO BOT)
from flask import Flask, request
import requests
import os

app = Flask(__name__)

# Variáveis de Ambiente
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "token_anderson_faq")

def enviar_mensagem(to, text):
    url = f"https://graph.facebook.com/v22.0/{WHATSAPP_PHONE_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": f"TESTE OK: {text}"},
    }
    try:
        r = requests.post(url, headers=headers, json=payload)
        print(f"Enviado: {r.status_code} - {r.text}")
    except Exception as e:
        print(f"Erro envio: {e}")

@app.route("/webhook", methods=["GET"])
def verify():
    if request.args.get("hub.mode") == "subscribe" and request.args.get("hub.verify_token") == VERIFY_TOKEN:
        return request.args.get("hub.challenge"), 200
    return "Erro Token", 403

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    print("PAYLOAD RECEBIDO:", data) # Debug no Log
    
    if data and "entry" in data:
        for entry in data["entry"]:
            for change in entry.get("changes", []):
                value = change.get("value", {})
                if "messages" in value:
                    for msg in value["messages"]:
                        telefone = msg.get("from")
                        texto = msg.get("text", {}).get("body", "")
                        # Responde imediatamente
                        if telefone and texto:
                            enviar_mensagem(telefone, texto)
    
    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

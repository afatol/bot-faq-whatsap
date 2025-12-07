# app.py
"""
Bot de FAQ para WhatsApp - Versão Otimizada (Async + Low Memory)
- WhatsApp Cloud API
- sentence-transformers (Modelo Leve)
- Flask + Threading (Para evitar Timeouts)
"""

from flask import Flask, request
import requests
import os
from threading import Thread  # <--- IMPORTANTE: Permite rodar a IA em segundo plano
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np

# ======================================================
# 1. FAQ: DADOS
# ======================================================

faq_data = [
    # ---- SEUS DADOS AQUI ----
    {
        "pergunta_faq": "O que é semaglutida?",
        "resposta_faq": "A semaglutida é um agonista do receptor GLP-1 usada para tratamento de diabetes tipo 2 e controle de peso. Ela reduz o apetite, retarda o esvaziamento gástrico e melhora a sensibilidade à insulina.",
    },
    {
        "pergunta_faq": "Quais são os efeitos colaterais mais comuns da semaglutida?",
        "resposta_faq": "Os efeitos adversos mais comuns são náuseas, vômitos, diarreia, constipação, refluxo e sensação de estômago cheio.",
    },
    {
        "pergunta_faq": "O que é tirzepatida?",
        "resposta_faq": "A tirzepatida é um agonista duplo dos receptores GIP e GLP-1, com efeito potente na perda de peso e controle do diabetes.",
    },
    {
        "pergunta_faq": "Quais são os efeitos colaterais da tirzepatida?",
        "resposta_faq": "Incluem náuseas, diarreia, constipação, redução do apetite e fadiga, similares aos do GLP-1, mas podem ser intensos no início.",
    },
    {
        "pergunta_faq": "O que é retatrutida?",
        "resposta_faq": "A retatrutida é um agonista triplo (GLP-1, GIP e glucagon) ainda em estudos, mostrando resultados promissores de perda de peso superior a 20%.",
    },
    {
        "pergunta_faq": "Quem não deve usar semaglutida ou tirzepatida?",
        "resposta_faq": "Contraindicado para quem tem histórico de carcinoma medular de tireoide, síndrome MEN2 ou alergia aos componentes.",
    },
    {
        "pergunta_faq": "Gestantes podem usar semaglutida ou tirzepatida?",
        "resposta_faq": "Não. O uso não é recomendado durante a gestação por falta de dados de segurança.",
    },
    {
        "pergunta_faq": "Como devo armazenar a caneta de semaglutida?",
        "resposta_faq": "Antes de aberta: geladeira (2°C a 8°C). Após aberta: temperatura ambiente (conforme a bula) ou geladeira.",
    },
    {
        "pergunta_faq": "Posso beber álcool usando semaglutida ou tirzepatida?",
        "resposta_faq": "Com moderação. O álcool pode piorar náuseas e hipoglicemia, além de ser calórico.",
    },
    {
        "pergunta_faq": "O que acontece se eu esquecer uma dose?",
        "resposta_faq": "Geralmente, se faltam mais de 2 dias para a próxima, tome assim que lembrar. Se estiver perto, pule e siga o fluxo normal. Consulte a bula.",
    }
    # ... Adicione o resto das suas perguntas aqui ...
]

# ======================================================
# 2. IA: Preparação do Modelo (Leve)
# ======================================================

# Preparar DataFrame
faq_df = pd.DataFrame(faq_data)
perguntas_faq = faq_df["pergunta_faq"].tolist()

# Modelo Leve (80MB) para não travar o servidor gratuito
model_name = "all-MiniLM-L6-v2"
print(f"Carregando modelo de IA: {model_name}...")
model = SentenceTransformer(model_name)
embeddings_faq = model.encode(perguntas_faq, convert_to_tensor=True)
print("Modelo carregado com sucesso!")

def buscar_resposta_faq(pergunta_usuario: str, limite_similaridade: float = 0.55) -> dict:
    """Busca a resposta mais similar no banco de dados."""
    pergunta_usuario = (pergunta_usuario or "").strip()
    if not pergunta_usuario:
        return {"confiavel": False, "resposta_encontrada": "Não entendi."}

    embedding_usuario = model.encode(pergunta_usuario, convert_to_tensor=True)
    similaridades = util.cos_sim(embedding_usuario, embeddings_faq)[0]

    indice_max = int(np.argmax(similaridades))
    similaridade_maxima = float(similaridades[indice_max])

    confiavel = similaridade_maxima >= limite_similaridade

    return {
        "resposta_encontrada": faq_df.iloc[indice_max]["resposta_faq"],
        "confiavel": confiavel
    }

# ======================================================
# 3. Servidor Flask + WhatsApp
# ======================================================

app = Flask(__name__)

# Variáveis de Ambiente
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID")
# Tenta pegar do ambiente, se não achar, usa o seu fixo
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "token_anderson_faq")

def enviar_mensagem_whatsapp(to: str, message: str):
    """Envia a mensagem final para o usuário via API."""
    if not WHATSAPP_TOKEN or not WHATSAPP_PHONE_ID:
        print("ERRO: Credenciais do WhatsApp não configuradas.")
        return

    url = f"https://graph.facebook.com/v22.0/{WHATSAPP_PHONE_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": message},
    }

    try:
        resp = requests.post(url, headers=headers, json=payload)
        print(f"Mensagem enviada para {to}. Status: {resp.status_code}")
    except Exception as e:
        print(f"Erro ao enviar mensagem: {e}")

def tarefa_assincrona(text, phone):
    """
    Esta função roda em SEGUNDO PLANO (Thread).
    Ela processa a IA pesada e envia a resposta depois.
    """
    try:
        print(f"Processando IA para: {phone}...")
        
        # 1. Busca na IA
        resultado = buscar_resposta_faq(text)
        
        if resultado["confiavel"]:
            resposta = resultado["resposta_encontrada"]
        else:
            resposta = "Desculpe, não encontrei uma resposta exata no meu banco de dados sobre isso. Tente reformular a pergunta."

        # 2. Envia de volta pro WhatsApp
        enviar_mensagem_whatsapp(phone, resposta)
        
    except Exception as e:
        print(f"Erro no processamento assíncrono: {e}")

# --- ROTAS ---

@app.route("/webhook", methods=["GET"])
def verify():
    """Validação do Webhook (Meta)"""
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        return challenge, 200
    return "Verification failed", 403

@app.route("/webhook", methods=["POST"])
def webhook():
    """Recebe a mensagem e libera o servidor imediatamente"""
    data = request.get_json()
    
    # Validação básica
    if data and "entry" in data:
        for entry in data["entry"]:
            if "changes" in entry:
                for change in entry["changes"]:
                    value = change.get("value", {})
                    messages = value.get("messages", [])

                    if not messages:
                        continue # Ignora status de leitura/entrega

                    for message in messages:
                        phone = message.get("from")
                        text = message.get("text", {}).get("body", "").strip()

                        if phone and text:
                            # --- O PULO DO GATO ---
                            # Lança uma Thread para processar sem travar o Flask
                            thread = Thread(target=tarefa_assincrona, args=(text, phone))
                            thread.start()
                            
    # Responde 200 OK em menos de 1 segundo para a Meta não dar Timeout
    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

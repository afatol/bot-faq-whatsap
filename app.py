# app.py
"""
Bot de FAQ para WhatsApp usando:
- WhatsApp Cloud API
- sentence-transformers (modelo BERT-like)
- Flask como servidor

Fluxo:
- WhatsApp -> Webhook /webhook (POST)
- processar_pergunta() -> buscar_resposta_faq()
- Resposta é enviada de volta pela API do WhatsApp
"""

from flask import Flask, request
import requests
import os

from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np

# ======================================================
# 1. FAQ: COLE AQUI O SEU faq_data COMPLETO
# ======================================================

faq_data = [
    # ---- EXEMPLOS DE ENTRADAS ----
    # Substitua ou complemente este bloco com TODAS as perguntas
    # que você já criou no notebook (seu faq_data grande).

    {
        "pergunta_faq": "O que é semaglutida?",
        "resposta_faq": (
            "A semaglutida é um agonista do receptor GLP-1 usada para tratamento de "
            "diabetes tipo 2 e controle de peso. Ela reduz o apetite, retarda o "
            "esvaziamento gástrico e melhora a sensibilidade à insulina, com eficácia "
            "comprovada em ensaios clínicos."
        ),
    },
    {
        "pergunta_faq": "Quais são os efeitos colaterais mais comuns da semaglutida?",
        "resposta_faq": (
            "Os efeitos adversos mais comuns são náuseas, vômitos, diarreia, constipação, "
            "refluxo e sensação de estômago cheio. Esses efeitos são descritos em bula e "
            "observados com frequência em estudos clínicos."
        ),
    },
    {
        "pergunta_faq": "O que é tirzepatida?",
        "resposta_faq": (
            "A tirzepatida é um agonista duplo dos receptores GIP e GLP-1. Foi desenvolvida "
            "para o tratamento do diabetes tipo 2 e demonstrou um efeito muito potente em "
            "perda de peso em estudos clínicos, geralmente maior que o observado com "
            "semaglutida em doses equivalentes."
        ),
    },
    {
        "pergunta_faq": "Quais são os efeitos colaterais da tirzepatida?",
        "resposta_faq": (
            "Os efeitos colaterais mais comuns incluem náuseas, diarreia, constipação, "
            "redução do apetite e fadiga. Esses efeitos tendem a ser mais intensos ao "
            "aumentar a dose e podem diminuir com o tempo."
        ),
    },
    {
        "pergunta_faq": "O que é retatrutida?",
        "resposta_faq": (
            "A retatrutida é um agonista triplo que atua nos receptores GLP-1, GIP e "
            "glucagon. Ainda está em fase de estudos clínicos, mas resultados preliminares "
            "mostram perdas de peso superiores a 20% do peso corporal em alguns protocolos."
        ),
    },
    {
        "pergunta_faq": "Quem não deve usar semaglutida ou tirzepatida?",
        "resposta_faq": (
            "As bulas contraindicam o uso em pessoas com histórico pessoal ou familiar de "
            "carcinoma medular de tireoide, síndrome MEN2 e alergia conhecida aos componentes. "
            "Também é necessária cautela em pacientes com histórico de pancreatite."
        ),
    },
    {
        "pergunta_faq": "Gestantes podem usar semaglutida ou tirzepatida?",
        "resposta_faq": (
            "Não. As bulas não recomendam o uso durante a gestação devido à falta de dados "
            "de segurança adequados e riscos observados em estudos com animais."
        ),
    },
    {
        "pergunta_faq": "Como devo armazenar a caneta de semaglutida?",
        "resposta_faq": (
            "Antes de aberta, a caneta deve ser mantida sob refrigeração entre 2°C e 8°C. "
            "Após aberta, muitas apresentações permitem armazenamento em temperatura ambiente "
            "controlada por algumas semanas, conforme especificado em bula."
        ),
    },
    {
        "pergunta_faq": "Posso beber álcool usando semaglutida ou tirzepatida?",
        "resposta_faq": (
            "Pequenas quantidades de álcool costumam ser permitidas, mas o uso excessivo "
            "pode piorar náuseas, sobrecarregar o fígado e atrapalhar o processo de "
            "emagrecimento. Clinicamente recomenda-se moderação."
        ),
    },
    {
        "pergunta_faq": "O que acontece se eu esquecer uma dose de semaglutida?",
        "resposta_faq": (
            "Se ainda faltarem mais de dois dias para a próxima dose, em geral pode-se "
            "aplicar assim que lembrar. Se estiver muito próximo da próxima aplicação, "
            "a orientação usual é pular a dose esquecida e seguir o calendário normal, "
            "conforme bula e orientação médica."
        ),
    },

    # --- AQUI você deve continuar colando todas as perguntas/respostas
    #     do seu faq_data completo que já está no notebook.
]

# ======================================================
# 2. Preparar DataFrame, modelo e embeddings
# ======================================================

faq_df = pd.DataFrame(faq_data)
perguntas_faq = faq_df["pergunta_faq"].tolist()

model_name "paraphrase-multilingual-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
model = SentenceTransformer(model_name)

embeddings_faq = model.encode(perguntas_faq, convert_to_tensor=True)


def buscar_resposta_faq(pergunta_usuario: str, limite_similaridade: float = 0.55) -> dict:
    """
    Calcula a similaridade da pergunta do usuário com o FAQ e devolve a melhor resposta.
    """
    pergunta_usuario = (pergunta_usuario or "").strip()
    if not pergunta_usuario:
        return {
            "pergunta_usuario": "",
            "pergunta_encontrada": "",
            "resposta_encontrada": "Não entendi a pergunta.",
            "similaridade_maxima": 0.0,
            "confiavel": False,
        }

    embedding_usuario = model.encode(pergunta_usuario, convert_to_tensor=True)
    similaridades = util.cos_sim(embedding_usuario, embeddings_faq)[0]

    indice_max = int(np.argmax(similaridades))
    similaridade_maxima = float(similaridades[indice_max])

    pergunta_encontrada = faq_df.iloc[indice_max]["pergunta_faq"]
    resposta_encontrada = faq_df.iloc[indice_max]["resposta_faq"]

    confiavel = similaridade_maxima >= limite_similaridade

    return {
        "pergunta_usuario": pergunta_usuario,
        "pergunta_encontrada": pergunta_encontrada,
        "resposta_encontrada": resposta_encontrada,
        "similaridade_maxima": similaridade_maxima,
        "confiavel": confiavel,
    }


# ======================================================
# 3. Flask + WhatsApp Cloud API
# ======================================================

app = Flask(__name__)

# Pegamos token e phone_id das variáveis de ambiente
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")          # NÃO coloque o EAAT... direto no código
WHATSAPP_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID")    # ex: 923030724225166
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "token_anderson_faq")


@app.route("/webhook", methods=["GET"])
def verify():
    """
    Endpoint de verificação usado pelo Meta quando você cadastra o Webhook.
    """
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        return challenge, 200
    return "Verification failed", 403


@app.route("/webhook", methods=["POST"])
def webhook():
    """
    Endpoint que recebe mensagens do WhatsApp.
    """
    data = request.get_json()

    # print("Webhook recebido:", data)  # debug opcional

    if data and "entry" in data:
        for entry in data["entry"]:
            if "changes" in entry:
                for change in entry["changes"]:
                    value = change.get("value", {})
                    messages = value.get("messages", [])

                    for message in messages:
                        phone = message.get("from")
                        text = message.get("text", {}).get("body", "").strip()

                        if not phone or not text:
                            continue

                        resposta = processar_pergunta(text)
                        enviar_mensagem_whatsapp(phone, resposta)

    return "OK", 200


def processar_pergunta(texto: str) -> str:
    """
    Encapsula a chamada ao modelo de FAQ. Aqui você pode personalizar respostas.
    """
    resultado = buscar_resposta_faq(texto, limite_similaridade=0.55)

    if resultado["confiavel"]:
        resposta = resultado["resposta_encontrada"]
    else:
        resposta = (
            "Não tenho alta confiança para responder exatamente essa pergunta.\n"
            "Tente reformular ou consulte um profissional de saúde."
        )

    return resposta


def enviar_mensagem_whatsapp(to: str, message: str):
    """
    Envia resposta de texto via WhatsApp Cloud API.
    """
    if not WHATSAPP_TOKEN or not WHATSAPP_PHONE_ID:
        print("ERRO: configure WHATSAPP_TOKEN e WHATSAPP_PHONE_ID nas variáveis de ambiente.")
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

    resp = requests.post(url, headers=headers, json=payload)
    print("Status WhatsApp:", resp.status_code, resp.text)


if __name__ == "__main__":
    # Para testes locais
    app.run(host="0.0.0.0", port=8000, debug=True)

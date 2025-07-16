from flask import Flask, render_template, request, jsonify
import joblib
from groq import Groq
import os
import requests

# ✅ Environment variables
os.environ['GROQ_API_KEY'] = os.getenv("groq")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")  # set this in Render
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/main", methods=["GET", "POST"])
def main():
    q = request.form.get("q")
    return render_template("main.html")

# =========================
# LLAMA Routes
# =========================
@app.route("/llama", methods=["GET", "POST"])
def llama():
    return render_template("llama.html")

@app.route("/llama_reply", methods=["GET", "POST"])
def llama_reply():
    q = request.form.get("q")
    client = Groq()
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": q}]
    )
    return render_template("llama_reply.html", r=completion.choices[0].message.content)

# =========================
# DeepSeek Routes
# =========================
@app.route("/deepseek", methods=["GET", "POST"])
def deepseek():
    return render_template("deepseek.html")

@app.route("/deepseek_reply", methods=["GET", "POST"])
def deepseek_reply():
    user_prompt = request.form.get("prompt")
    client = Groq()
    completion_ds = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[{"role": "user", "content": user_prompt}]
    )
    return render_template("deepseek_reply.html", result=completion_ds.choices[0].message.content)

# =========================
# DBS Prediction Routes
# =========================
@app.route("/dbs", methods=["GET", "POST"])
def dbs():
    return render_template("dbs.html")

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    q = float(request.form.get("q"))
    model = joblib.load("dbs.jl")
    pred = model.predict([[q]])
    return render_template("prediction.html", r=pred)

# =========================
# Telegram Route (info page)
# =========================
@app.route("/telegram", methods=["GET", "POST"])
def telegram():
    bot_link = "https://t.me/marissaybot"
    return render_template("telegram.html", bot_link=bot_link)

# =========================
# Telegram Webhook Endpoint → Reply with DeepSeek
# =========================
@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()

    if "message" in data:
        chat_id = data["message"]["chat"]["id"]
        user_text = data["message"].get("text", "")

        # Call DeepSeek reasoning model
        client = Groq()
        completion_ds = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[{"role": "user", "content": user_text}]
        )
        deepseek_reply = completion_ds.choices[0].message.content

        # Send DeepSeek reply to Telegram
        requests.post(f"{TELEGRAM_API_URL}/sendMessage", json={
            "chat_id": chat_id,
            "text": deepseek_reply
        })

    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)



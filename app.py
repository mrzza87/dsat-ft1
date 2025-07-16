from flask import Flask, render_template, request, jsonify
import joblib
from groq import Groq
import os
import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Load API keys from environment
os.environ['GROQ_API_KEY'] = os.getenv("groq")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # You'll need to set this
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

app = Flask(__name__)

def send_telegram_message(chat_id, message):
    """Send a message back to Telegram user"""
    url = f"{TELEGRAM_API_URL}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }
    
    try:
        response = requests.post(url, json=data)
        return response.json()
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        return None

def get_ai_response(user_message, model_type="llama"):
    """Get AI response using Groq"""
    try:
        client = Groq()
        
        if model_type == "deepseek":
            model = "deepseek-r1-distill-llama-70b"
        else:
            model = "llama-3.1-8b-instant"
            
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_message}]
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"AI response error: {e}")
        return "Sorry, I'm having trouble thinking right now. Try again!"

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
# Telegram Routes
# =========================
@app.route("/telegram", methods=["GET", "POST"])
def telegram():
    bot_link = "https://t.me/marissaybot"
    return render_template("telegram.html", bot_link=bot_link)

@app.route("/webhook/telegram", methods=["POST"])
def telegram_webhook():
    """Handle incoming Telegram messages"""
    try:
        data = request.get_json()
        logger.info(f"Telegram webhook received: {json.dumps(data, indent=2)}")
        
        # Check if this is a message
        if 'message' in data:
            message = data['message']
            chat_id = message['chat']['id']
            user_text = message.get('text', '')
            user_name = message.get('from', {}).get('first_name', 'Friend')
            
            logger.info(f"Message from {user_name} ({chat_id}): {user_text}")
            
            # Handle different commands
            if user_text.startswith('/start'):
                response = f"Hey {user_name}! 👋\n\nI'm your AI assistant powered by Groq!\n\n🤖 Just send me any message and I'll respond using:\n• Default: Llama 3.1\n• Type '/deepseek [message]' for DeepSeek model\n• Type '/predict [number]' for DBS prediction\n\nTry me out!"
                
            elif user_text.startswith('/help'):
                response = "📋 <b>Available Commands:</b>\n\n💬 Send any message - I'll respond with Llama 3.1\n🧠 /deepseek [message] - Use DeepSeek model\n📊 /predict [number] - Get DBS prediction\n🆘 /help - Show this help\n\nJust start typing! 🚀"
                
            elif user_text.startswith('/deepseek'):
                # Extract message after /deepseek
                prompt = user_text.replace('/deepseek', '').strip()
                if prompt:
                    response = f"🧠 <b>DeepSeek says:</b>\n\n{get_ai_response(prompt, 'deepseek')}"
                else:
                    response = "Please provide a message after /deepseek\nExample: /deepseek What is AI?"
                    
            elif user_text.startswith('/predict'):
                # Extract number after /predict
                try:
                    number_str = user_text.replace('/predict', '').strip()
                    number = float(number_str)
                    model = joblib.load("dbs.jl")
                    prediction = model.predict([[number]])
                    response = f"📊 <b>DBS Prediction:</b>\n\nInput: {number}\nPrediction: {prediction[0]}"
                except (ValueError, FileNotFoundError) as e:
                    response = "❌ Please provide a valid number after /predict\nExample: /predict 25.5"
                    
            elif user_text.strip() == '':
                response = "I didn't receive any text. Try sending me a message! 😊"
                
            else:
                # Regular message - use Llama
                response = f"🦙 <b>Llama says:</b>\n\n{get_ai_response(user_text, 'llama')}"
            
            # Send response back to user
            send_telegram_message(chat_id, response)
            
        return jsonify({"status": "ok"}), 200
        
    except Exception as e:
        logger.error(f"Telegram webhook error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# =========================
# Webhook Setup Helper
# =========================
@app.route("/set_webhook", methods=["GET"])
def set_webhook():
    """Helper endpoint to set up your Telegram webhook"""
    if not TELEGRAM_BOT_TOKEN:
        return jsonify({"error": "TELEGRAM_BOT_TOKEN not set"}), 400
    
    # Your app's URL (you'll need to replace this with your actual domain)
    webhook_url = request.url_root + "webhook/telegram"
    
    url = f"{TELEGRAM_API_URL}/setWebhook"
    data = {"url": webhook_url}
    
    try:
        response = requests.post(url, json=data)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/webhook_info", methods=["GET"])
def webhook_info():
    """Check current webhook status"""
    if not TELEGRAM_BOT_TOKEN:
        return jsonify({"error": "TELEGRAM_BOT_TOKEN not set"}), 400
    
    url = f"{TELEGRAM_API_URL}/getWebhookInfo"
    
    try:
        response = requests.get(url)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)


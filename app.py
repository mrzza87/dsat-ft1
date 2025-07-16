from flask import Flask, render_template, request, jsonify
import joblib
from groq import Groq
import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Load Groq API key from environment
os.environ['GROQ_API_KEY'] = os.getenv("groq")

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
# Telegram Route
# =========================
@app.route("/telegram", methods=["GET", "POST"])
def telegram():
    # Replace with your actual bot username
    bot_link = "https://t.me/marissaybot"
    return render_template("telegram.html", bot_link=bot_link)

# =========================
# Webhook Routes
# =========================
@app.route("/webhook", methods=["POST"])
def webhook():
    """
    Generic webhook endpoint that accepts POST requests
    """
    try:
        # Get the raw data
        data = request.get_json()
        
        # Log the incoming webhook data
        logger.info(f"Webhook received: {json.dumps(data, indent=2)}")
        
        # Process the webhook data here
        # You can add your custom logic based on the webhook source
        
        # Example: Handle different webhook types
        if data and 'type' in data:
            webhook_type = data['type']
            
            if webhook_type == 'telegram':
                return handle_telegram_webhook(data)
            elif webhook_type == 'github':
                return handle_github_webhook(data)
            elif webhook_type == 'stripe':
                return handle_stripe_webhook(data)
            else:
                return handle_generic_webhook(data)
        
        # Return success response
        return jsonify({
            "status": "success",
            "message": "Webhook received successfully",
            "data": data
        }), 200
        
    except Exception as e:
        logger.error(f"Webhook error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/webhook/telegram", methods=["POST"])
def telegram_webhook():
    """
    Specific webhook endpoint for Telegram bot
    """
    try:
        data = request.get_json()
        logger.info(f"Telegram webhook: {json.dumps(data, indent=2)}")
        
        # Process Telegram webhook
        if 'message' in data:
            message = data['message']
            chat_id = message['chat']['id']
            text = message.get('text', '')
            
            # You can add your Telegram bot logic here
            # For example, respond to specific commands
            
            logger.info(f"Received message from {chat_id}: {text}")
        
        return jsonify({"status": "ok"}), 200
        
    except Exception as e:
        logger.error(f"Telegram webhook error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/webhook/github", methods=["POST"])
def github_webhook():
    """
    Specific webhook endpoint for GitHub
    """
    try:
        data = request.get_json()
        event_type = request.headers.get('X-GitHub-Event')
        
        logger.info(f"GitHub webhook - Event: {event_type}")
        logger.info(f"GitHub webhook - Data: {json.dumps(data, indent=2)}")
        
        # Process GitHub webhook based on event type
        if event_type == 'push':
            return handle_github_push(data)
        elif event_type == 'pull_request':
            return handle_github_pr(data)
        
        return jsonify({"status": "ok"}), 200
        
    except Exception as e:
        logger.error(f"GitHub webhook error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# =========================
# Webhook Handler Functions
# =========================
def handle_telegram_webhook(data):
    """Handle Telegram-specific webhook logic"""
    logger.info("Processing Telegram webhook")
    # Add your Telegram-specific logic here
    return jsonify({"status": "telegram_processed"}), 200

def handle_github_webhook(data):
    """Handle GitHub-specific webhook logic"""
    logger.info("Processing GitHub webhook")
    # Add your GitHub-specific logic here
    return jsonify({"status": "github_processed"}), 200

def handle_github_push(data):
    """Handle GitHub push events"""
    repo_name = data.get('repository', {}).get('name', 'unknown')
    logger.info(f"Push event for repository: {repo_name}")
    # Add your push handling logic here
    return jsonify({"status": "push_processed"}), 200

def handle_github_pr(data):
    """Handle GitHub pull request events"""
    action = data.get('action', 'unknown')
    logger.info(f"Pull request action: {action}")
    # Add your PR handling logic here
    return jsonify({"status": "pr_processed"}), 200

def handle_stripe_webhook(data):
    """Handle Stripe-specific webhook logic"""
    logger.info("Processing Stripe webhook")
    # Add your Stripe-specific logic here
    return jsonify({"status": "stripe_processed"}), 200

def handle_generic_webhook(data):
    """Handle generic webhook logic"""
    logger.info("Processing generic webhook")
    # Add your generic webhook logic here
    return jsonify({"status": "generic_processed"}), 200

# =========================
# Webhook Testing Route
# =========================
@app.route("/webhook/test", methods=["GET", "POST"])
def test_webhook():
    """
    Test endpoint to simulate webhook calls
    """
    if request.method == "GET":
        return jsonify({
            "message": "Webhook test endpoint",
            "endpoints": {
                "generic": "/webhook",
                "telegram": "/webhook/telegram",
                "github": "/webhook/github"
            }
        })
    
    # POST request for testing
    test_data = {
        "type": "test",
        "message": "This is a test webhook",
        "timestamp": "2025-01-01T00:00:00Z"
    }
    
    return jsonify({
        "status": "test_success",
        "data": test_data
    }), 200

if __name__ == "__main__":
    # Keep host open for Render, fallback port 5000 for local
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

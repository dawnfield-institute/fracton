"""
Web-based Agentic Chatbot with KronosMemory

FastAPI server providing a web interface for the chatbot.
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from chatbot import AgenticChatbot


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Fracton Agentic Chatbot",
    description="Conversational AI with PAC/SEC/MED memory foundations",
    version="1.0.0",
)

# Global chatbot instance
chatbot: Optional[AgenticChatbot] = None


# ============================================================================
# Models
# ============================================================================

class ChatMessage(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    context: List[Dict[str, Any]]
    health: Dict[str, Any]


class HealthResponse(BaseModel):
    c_squared: Dict[str, Any]
    balance_operator: Dict[str, Any]
    duty_cycle: Dict[str, Any]
    constants: Dict[str, Any]
    total_nodes: int
    collapses: int


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize chatbot on startup."""
    global chatbot

    print("🚀 Starting Fracton Chatbot Server...")

    chatbot = AgenticChatbot(
        storage_path=os.getenv("FRACTON_DATA_DIR", "./data/chatbot"),
        device=os.getenv("DEVICE", "cpu"),
        embedding_model="mini",
        llm_provider=os.getenv("LLM_PROVIDER"),
        llm_api_key=os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"),
    )

    await chatbot.initialize()
    print("✅ Chatbot ready")


@app.on_event("shutdown")
async def shutdown_event():
    """Close chatbot on shutdown."""
    global chatbot

    if chatbot:
        await chatbot.close()
        print("👋 Chatbot closed")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Serve chatbot web interface."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fracton Agentic Chatbot</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .container {
            width: 90%;
            max-width: 800px;
            height: 90vh;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        .header h1 {
            font-size: 24px;
            margin-bottom: 5px;
        }
        .header p {
            font-size: 14px;
            opacity: 0.9;
        }
        .health {
            background: rgba(0,0,0,0.1);
            padding: 10px;
            font-size: 12px;
            display: flex;
            justify-content: space-around;
            border-top: 1px solid rgba(255,255,255,0.2);
        }
        .health-item {
            text-align: center;
        }
        .health-label {
            opacity: 0.8;
            margin-bottom: 2px;
        }
        .health-value {
            font-weight: bold;
        }
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .message {
            margin-bottom: 15px;
            display: flex;
            align-items: flex-start;
        }
        .message.user {
            justify-content: flex-end;
        }
        .message-bubble {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 18px;
            line-height: 1.4;
        }
        .message.user .message-bubble {
            background: #667eea;
            color: white;
        }
        .message.bot .message-bubble {
            background: white;
            color: #333;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .message-time {
            font-size: 11px;
            opacity: 0.6;
            margin-top: 5px;
        }
        .input-area {
            padding: 20px;
            background: white;
            border-top: 1px solid #ddd;
            display: flex;
            gap: 10px;
        }
        #messageInput {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #ddd;
            border-radius: 25px;
            font-size: 14px;
            outline: none;
            transition: border-color 0.3s;
        }
        #messageInput:focus {
            border-color: #667eea;
        }
        #sendButton {
            padding: 12px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            transition: transform 0.2s;
        }
        #sendButton:hover {
            transform: scale(1.05);
        }
        #sendButton:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .typing-indicator {
            display: none;
            padding: 10px;
            background: white;
            border-radius: 18px;
            width: 60px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .typing-indicator span {
            height: 8px;
            width: 8px;
            background: #999;
            border-radius: 50%;
            display: inline-block;
            margin: 0 2px;
            animation: typing 1.4s infinite;
        }
        .typing-indicator span:nth-child(2) {
            animation-delay: 0.2s;
        }
        .typing-indicator span:nth-child(3) {
            animation-delay: 0.4s;
        }
        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-10px); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Fracton Agentic Chatbot</h1>
            <p>Powered by KronosMemory with PAC/SEC/MED Foundations</p>
            <div class="health">
                <div class="health-item">
                    <div class="health-label">c²</div>
                    <div class="health-value" id="c2-value">--</div>
                </div>
                <div class="health-item">
                    <div class="health-label">Balance Ξ</div>
                    <div class="health-value" id="xi-value">--</div>
                </div>
                <div class="health-item">
                    <div class="health-label">Duty Cycle</div>
                    <div class="health-value" id="duty-value">--</div>
                </div>
                <div class="health-item">
                    <div class="health-label">Collapses</div>
                    <div class="health-value" id="collapse-value">0</div>
                </div>
            </div>
        </div>
        <div class="messages" id="messages">
            <div class="message bot">
                <div class="message-bubble">
                    Hello! I'm an agentic chatbot powered by Fracton's KronosMemory with PAC/SEC/MED foundations.
                    Ask me anything, and I'll remember our conversation with full conservation validation!
                </div>
            </div>
        </div>
        <div class="input-area">
            <input type="text" id="messageInput" placeholder="Type your message..." />
            <button id="sendButton">Send</button>
        </div>
    </div>

    <script>
        const messagesDiv = document.getElementById('messages');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');

        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;

            // Add user message
            addMessage('user', message);
            messageInput.value = '';
            sendButton.disabled = true;

            // Show typing indicator
            const typing = addTypingIndicator();

            try {
                // Send to API
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message })
                });

                const data = await response.json();

                // Remove typing indicator
                typing.remove();

                // Add bot response
                addMessage('bot', data.response);

                // Update health metrics
                updateHealth(data.health);

            } catch (error) {
                typing.remove();
                addMessage('bot', 'Sorry, I encountered an error. Please try again.');
                console.error('Error:', error);
            } finally {
                sendButton.disabled = false;
                messageInput.focus();
            }
        }

        function addMessage(role, text) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            messageDiv.innerHTML = `
                <div class="message-bubble">
                    ${text}
                    <div class="message-time">${new Date().toLocaleTimeString()}</div>
                </div>
            `;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        function addTypingIndicator() {
            const typing = document.createElement('div');
            typing.className = 'message bot';
            typing.innerHTML = `
                <div class="typing-indicator" style="display: flex;">
                    <span></span><span></span><span></span>
                </div>
            `;
            messagesDiv.appendChild(typing);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            return typing;
        }

        function updateHealth(health) {
            if (health.c_squared && health.c_squared.latest) {
                document.getElementById('c2-value').textContent = health.c_squared.latest.toFixed(2);
            }
            if (health.balance_operator && health.balance_operator.latest) {
                const xi = health.balance_operator.latest;
                const xiTarget = health.constants.xi;
                let xiText = xi.toFixed(4);
                if (xi > xiTarget) xiText += ' ⚠️';
                document.getElementById('xi-value').textContent = xiText;
            }
            if (health.duty_cycle && health.duty_cycle.latest) {
                document.getElementById('duty-value').textContent = health.duty_cycle.latest.toFixed(3);
            }
        }

        // Event listeners
        sendButton.addEventListener('click', sendMessage);
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });

        // Load initial health
        fetch('/api/health')
            .then(r => r.json())
            .then(updateHealth)
            .catch(console.error);
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Process chat message."""
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")

    # Process message
    response = await chatbot.chat(message.message)

    # Get context
    context = await chatbot._retrieve_context(message.message, limit=3)

    # Get health
    health = chatbot.memory.get_foundation_health()
    stats = await chatbot.memory.get_stats()
    health["collapses"] = stats["collapses"]

    return ChatResponse(
        response=response,
        context=context,
        health=health,
    )


@app.get("/api/health", response_model=HealthResponse)
async def get_health():
    """Get foundation health metrics."""
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")

    health = chatbot.memory.get_foundation_health()
    stats = await chatbot.memory.get_stats()

    return HealthResponse(
        c_squared=health["c_squared"],
        balance_operator=health["balance_operator"],
        duty_cycle=health["duty_cycle"],
        constants=health["constants"],
        total_nodes=stats["total_nodes"],
        collapses=stats["collapses"],
    )


@app.get("/api/stats")
async def get_stats():
    """Get full memory statistics."""
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")

    return await chatbot.memory.get_stats()


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        log_level="info",
    )

#!/bin/bash
# Run script for Fracton Agentic Chatbot

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║          Fracton Agentic Chatbot Launcher                   ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if running in Docker
if [ -f /.dockerenv ]; then
    echo "🐳 Running in Docker container"
    DEVICE=${DEVICE:-cpu}
else
    echo "💻 Running locally"
    DEVICE=${DEVICE:-cpu}
fi

# Mode selection
MODE=${1:-cli}

case $MODE in
    cli)
        echo "🖥️  Starting CLI mode..."
        python /app/chatbot/chatbot.py
        ;;
    web)
        echo "🌐 Starting web mode on port ${PORT:-8080}..."
        python /app/chatbot/web_chatbot.py
        ;;
    *)
        echo "❌ Unknown mode: $MODE"
        echo "Usage: $0 [cli|web]"
        exit 1
        ;;
esac

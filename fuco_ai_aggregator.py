#!/usr/bin/env python3
"""
FUCO AI Aggregator - Single File Version
Smart AI responses from ChatGPT, Gemini, and DeepSeek in one place
"""

import os
import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import List, Dict, Optional
from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
import threading

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration settings for FUCO AI Aggregator"""
    
    # API Keys (set these as environment variables)
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your_openai_api_key_here')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'your_gemini_api_key_here')
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'your_deepseek_api_key_here')
    
    # API Endpoints
    OPENAI_URL = "https://api.openai.com/v1/chat/completions"
    GEMINI_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent"
    DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
    
    # Model Settings
    OPENAI_MODEL = "gpt-3.5-turbo"
    GEMINI_MODEL = "gemini-pro"
    DEEPSEEK_MODEL = "deepseek-chat"
    
    # Cost Optimization
    MAX_TOKENS = 1000
    TEMPERATURE = 0.7
    
    # Rate Limiting
    REQUESTS_PER_MINUTE = 30

# =============================================================================
# AI AGGREGATOR CORE
# =============================================================================

class FUCOAggregator:
    """Core AI aggregation logic"""
    
    def __init__(self):
        self.config = Config()
        self.conversation_history = []
        self.request_times = []
        
    def _rate_limit(self):
        """Basic rate limiting implementation"""
        current_time = time.time()
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        if len(self.request_times) >= self.config.REQUESTS_PER_MINUTE:
            return False
        self.request_times.append(current_time)
        return True
    
    async def query_chatgpt(self, session: aiohttp.ClientSession, question: str) -> Dict:
        """Query OpenAI ChatGPT"""
        try:
            if not self._rate_limit():
                return {"source": "ChatGPT", "response": "Error: Rate limit exceeded", "confidence": 0.0, "cost": 0}
                
            headers = {
                "Authorization": f"Bearer {self.config.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.config.OPENAI_MODEL,
                "messages": [{"role": "user", "content": question}],
                "max_tokens": self.config.MAX_TOKENS,
                "temperature": self.config.TEMPERATURE
            }
            
            async with session.post(self.config.OPENAI_URL, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "source": "ChatGPT",
                        "response": data["choices"][0]["message"]["content"],
                        "confidence": 0.9,
                        "cost": len(question) / 1000 * 0.0015
                    }
                else:
                    error_text = await response.text()
                    return {"source": "ChatGPT", "response": f"Error: API request failed - {error_text}", "confidence": 0.0, "cost": 0}
        except Exception as e:
            return {"source": "ChatGPT", "response": f"Error: {str(e)}", "confidence": 0.0, "cost": 0}

    async def query_gemini(self, session: aiohttp.ClientSession, question: str) -> Dict:
        """Query Google Gemini"""
        try:
            if not self._rate_limit():
                return {"source": "Gemini", "response": "Error: Rate limit exceeded", "confidence": 0.0, "cost": 0}
                
            url = f"{self.config.GEMINI_URL}?key={self.config.GEMINI_API_KEY}"
            
            payload = {
                "contents": [{
                    "parts": [{"text": question}]
                }],
                "generationConfig": {
                    "maxOutputTokens": self.config.MAX_TOKENS,
                    "temperature": self.config.TEMPERATURE
                }
            }
            
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "source": "Gemini",
                        "response": data["candidates"][0]["content"]["parts"][0]["text"],
                        "confidence": 0.85,
                        "cost": len(question) / 1000 * 0.0005
                    }
                else:
                    error_text = await response.text()
                    return {"source": "Gemini", "response": f"Error: API request failed - {error_text}", "confidence": 0.0, "cost": 0}
        except Exception as e:
            return {"source": "Gemini", "response": f"Error: {str(e)}", "confidence": 0.0, "cost": 0}

    async def query_deepseek(self, session: aiohttp.ClientSession, question: str) -> Dict:
        """Query DeepSeek"""
        try:
            if not self._rate_limit():
                return {"source": "DeepSeek", "response": "Error: Rate limit exceeded", "confidence": 0.0, "cost": 0}
                
            headers = {
                "Authorization": f"Bearer {self.config.DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.config.DEEPSEEK_MODEL,
                "messages": [{"role": "user", "content": question}],
                "max_tokens": self.config.MAX_TOKENS,
                "temperature": self.config.TEMPERATURE
            }
            
            async with session.post(self.config.DEEPSEEK_URL, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "source": "DeepSeek",
                        "response": data["choices"][0]["message"]["content"],
                        "confidence": 0.8,
                        "cost": len(question) / 1000 * 0.0001
                    }
                else:
                    error_text = await response.text()
                    return {"source": "DeepSeek", "response": f"Error: API request failed - {error_text}", "confidence": 0.0, "cost": 0}
        except Exception as e:
            return {"source": "DeepSeek", "response": f"Error: {str(e)}", "confidence": 0.0, "cost": 0}

    def smart_response_selection(self, responses: List[Dict]) -> Dict:
        """Smart selection using voting, confidence, and ensemble methods"""
        # Filter out error responses
        valid_responses = [r for r in responses if 'Error' not in r['response']]
        
        if not valid_responses:
            return {
                "selected_response": {"source": "FUCO", "response": "All AI services failed to respond.", "confidence": 0.0, "cost": 0},
                "voting_winner": None,
                "ensemble_response": "All AI services are currently unavailable. Please check your API keys and try again.",
                "all_responses": responses,
                "strategy_used": "fallback"
            }
        
        # Method 1: Confidence-based selection
        best_confidence = max(valid_responses, key=lambda x: x['confidence'])
        
        # Method 2: Voting system (simple content similarity)
        response_texts = [r['response'].lower()[:100] for r in valid_responses]
        vote_winner = max(set(response_texts), key=response_texts.count) if response_texts else None
        
        # Method 3: Ensemble - combine the best parts
        ensemble_response = "--- Combined Insights ---\n\n"
        for resp in valid_responses:
            ensemble_response += f"?? {resp['source']}:\n{resp['response']}\n\n"
        
        return {
            "selected_response": best_confidence,
            "voting_winner": vote_winner,
            "ensemble_response": ensemble_response,
            "all_responses": responses,
            "strategy_used": "confidence-voting-ensemble"
        }

    async def get_ai_responses(self, question: str) -> Dict:
        """Main method to get all AI responses in parallel"""
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.query_chatgpt(session, question),
                self.query_gemini(session, question),
                self.query_deepseek(session, question)
            ]
            
            responses = await asyncio.gather(*tasks)
        
        processing_time = time.time() - start_time
        
        # Smart selection
        final_result = self.smart_response_selection(responses)
        
        # Add to conversation history
        conversation_entry = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "question": question,
            "responses": final_result,
            "processing_time": processing_time
        }
        self.conversation_history.append(conversation_entry)
        
        # Keep only last 50 conversations
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]
        
        return final_result

    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history

    def calculate_cost_savings(self) -> Dict:
        """Calculate cost optimization metrics"""
        total_cost = 0
        total_processing_time = 0
        
        for conversation in self.conversation_history:
            for response in conversation['responses']['all_responses']:
                total_cost += response.get('cost', 0)
            total_processing_time += conversation.get('processing_time', 0)
        
        avg_processing_time = total_processing_time / len(self.conversation_history) if self.conversation_history else 0
        
        return {
            "total_conversations": len(self.conversation_history),
            "estimated_total_cost": round(total_cost, 4),
            "cost_per_conversation": round(total_cost / len(self.conversation_history), 4) if self.conversation_history else 0,
            "average_processing_time": round(avg_processing_time, 2)
        }

# =============================================================================
# FLASK APPLICATION
# =============================================================================

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    CORS(app)
    
    # Configuration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'fuco-dev-key-2024')
    
    # Initialize aggregator
    aggregator = FUCOAggregator()
    
    # HTML Template (embedded in the file)
    HTML_TEMPLATE = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>?? FUCO AI Aggregator</title>
        <style>
            /* FUCO Color Theme: Black, Blue, White */
            :root {
                --primary-black: #0a0a0a;
                --dark-black: #1a1a1a;
                --primary-blue: #2563eb;
                --light-blue: #3b82f6;
                --bright-blue: #60a5fa;
                --pure-white: #ffffff;
                --light-gray: #f8fafc;
                --border-gray: #e2e8f0;
            }

            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, var(--primary-black) 0%, var(--dark-black) 100%);
                color: var(--pure-white);
                min-height: 100vh;
                line-height: 1.6;
            }

            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }

            /* Header */
            .header {
                text-align: center;
                margin-bottom: 30px;
                padding: 20px;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 15px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(59, 130, 246, 0.2);
            }

            .header h1 {
                font-size: 2.5rem;
                background: linear-gradient(135deg, var(--bright-blue), var(--primary-blue));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 10px;
            }

            .header p {
                color: var(--bright-blue);
                font-size: 1.1rem;
            }

            /* Chat Container */
            .chat-container {
                background: var(--dark-black);
                border-radius: 15px;
                padding: 20px;
                margin-bottom: 20px;
                border: 1px solid rgba(59, 130, 246, 0.3);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            }

            .chat-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
                padding-bottom: 15px;
                border-bottom: 1px solid rgba(59, 130, 246, 0.3);
            }

            .chat-header h2 {
                color: var(--bright-blue);
            }

            .cost-display {
                display: flex;
                align-items: center;
                gap: 15px;
            }

            .cost-display span {
                background: var(--primary-blue);
                padding: 5px 12px;
                border-radius: 20px;
                font-size: 0.9rem;
                font-weight: 600;
            }

            /* Chat Messages */
            .chat-messages {
                min-height: 100px;
                max-height: 300px;
                overflow-y: auto;
                margin-bottom: 20px;
                padding: 15px;
                background: rgba(0, 0, 0, 0.3);
                border-radius: 10px;
                border: 1px solid rgba(59, 130, 246, 0.2);
            }

            .message {
                padding: 12px 15px;
                margin: 8px 0;
                border-radius: 10px;
                background: rgba(37, 99, 235, 0.1);
                border-left: 4px solid var(--primary-blue);
            }

            .message.user {
                background: rgba(255, 255, 255, 0.1);
                border-left-color: var(--bright-blue);
                margin-left: 20px;
            }

            /* Input Area */
            .input-area {
                display: flex;
                gap: 15px;
                align-items: flex-end;
            }

            .input-area textarea {
                flex: 1;
                padding: 15px;
                border: 1px solid rgba(59, 130, 246, 0.5);
                border-radius: 10px;
                background: rgba(0, 0, 0, 0.5);
                color: var(--pure-white);
                font-size: 1rem;
                resize: vertical;
                min-height: 60px;
                transition: all 0.3s ease;
            }

            .input-area textarea:focus {
                outline: none;
                border-color: var(--bright-blue);
                box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.3);
            }

            /* Buttons */
            .btn-primary {
                background: linear-gradient(135deg, var(--primary-blue), var(--light-blue));
                color: white;
                border: none;
                padding: 15px 25px;
                border-radius: 10px;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                min-width: 120px;
            }

            .btn-primary:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(37, 99, 235, 0.4);
            }

            .btn-primary:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }

            .btn-secondary {
                background: transparent;
                color: var(--bright-blue);
                border: 1px solid var(--bright-blue);
                padding: 8px 16px;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.3s ease;
            }

            .btn-secondary:hover {
                background: rgba(59, 130, 246, 0.1);
            }

            /* Responses Container */
            .responses-container {
                background: var(--dark-black);
                border-radius: 15px;
                border: 1px solid rgba(59, 130, 246, 0.3);
                overflow: hidden;
            }

            /* Tabs */
            .tabs {
                display: flex;
                background: rgba(0, 0, 0, 0.5);
                border-bottom: 1px solid rgba(59, 130, 246, 0.3);
            }

            .tab-button {
                flex: 1;
                padding: 15px 20px;
                background: transparent;
                border: none;
                color: var(--bright-blue);
                cursor: pointer;
                transition: all 0.3s ease;
                font-weight: 600;
            }

            .tab-button:hover {
                background: rgba(59, 130, 246, 0.1);
            }

            .tab-button.active {
                background: var(--primary-blue);
                color: white;
            }

            /* Tab Content */
            .tab-content {
                padding: 0;
            }

            .tab-pane {
                display: none;
                padding: 20px;
                max-height: 500px;
                overflow-y: auto;
            }

            .tab-pane.active {
                display: block;
            }

            .response-box {
                background: rgba(0, 0, 0, 0.3);
                border-radius: 10px;
                padding: 20px;
                border: 1px solid rgba(59, 130, 246, 0.2);
            }

            /* Individual Response Styles */
            .response-item {
                margin-bottom: 20px;
                padding: 15px;
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.05);
            }

            .response-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 10px;
                padding-bottom: 8px;
                border-bottom: 1px solid rgba(59, 130, 246, 0.3);
            }

            .ai-name {
                font-weight: bold;
                color: var(--bright-blue);
            }

            .confidence {
                background: var(--primary-blue);
                padding: 3px 8px;
                border-radius: 12px;
                font-size: 0.8rem;
            }

            .cost {
                color: #10b981;
                font-size: 0.9rem;
            }

            .response-content {
                line-height: 1.6;
                color: var(--pure-white);
                white-space: pre-wrap;
            }

            /* Loading Animation */
            .loading {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid rgba(255, 255, 255, 0.3);
                border-radius: 50%;
                border-top-color: var(--bright-blue);
                animation: spin 1s ease-in-out infinite;
            }

            @keyframes spin {
                to { transform: rotate(360deg); }
            }

            /* Scrollbar Styling */
            ::-webkit-scrollbar {
                width: 8px;
            }

            ::-webkit-scrollbar-track {
                background: rgba(0, 0, 0, 0.2);
            }

            ::-webkit-scrollbar-thumb {
                background: var(--primary-blue);
                border-radius: 4px;
            }

            ::-webkit-scrollbar-thumb:hover {
                background: var(--light-blue);
            }

            /* Responsive Design */
            @media (max-width: 768px) {
                .container {
                    padding: 10px;
                }
                
                .header h1 {
                    font-size: 2rem;
                }
                
                .input-area {
                    flex-direction: column;
                }
                
                .tabs {
                    flex-direction: column;
                }
                
                .chat-header {
                    flex-direction: column;
                    gap: 10px;
                    align-items: flex-start;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <!-- Header -->
            <header class="header">
                <h1>?? FUCO AI Aggregator</h1>
                <p>Smart responses from ChatGPT, Gemini, and DeepSeek</p>
            </header>

            <!-- Main Chat Area -->
            <div class="chat-container">
                <div class="chat-header">
                    <h2>Ask Anything</h2>
                    <div class="cost-display">
                        <span id="costMetrics">Cost: $0.00</span>
                        <button onclick="clearHistory()" class="btn-secondary">Clear History</button>
                    </div>
                </div>

                <div class="chat-messages" id="chatMessages">
                    <!-- Messages will appear here -->
                </div>

                <div class="input-area">
                    <textarea id="questionInput" placeholder="Type your question here..." rows="3"></textarea>
                    <button onclick="askQuestion()" id="askButton" class="btn-primary">Ask FUCO</button>
                </div>
            </div>

            <!-- Response Display -->
            <div class="responses-container">
                <div class="tabs" id="responseTabs">
                    <button class="tab-button active" onclick="switchTab('selected')">Selected Response</button>
                    <button class="tab-button" onclick="switchTab('ensemble')">Ensemble View</button>
                    <button class="tab-button" onclick="switchTab('all')">All Responses</button>
                    <button class="tab-button" onclick="switchTab('history')">Conversation History</button>
                </div>

                <div class="tab-content">
                    <div id="selected-tab" class="tab-pane active">
                        <div id="selectedResponse" class="response-box"></div>
                    </div>
                    <div id="ensemble-tab" class="tab-pane">
                        <div id="ensembleResponse" class="response-box"></div>
                    </div>
                    <div id="all-tab" class="tab-pane">
                        <div id="allResponses" class="response-box"></div>
                    </div>
                    <div id="history-tab" class="tab-pane">
                        <div id="conversationHistory" class="response-box"></div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Frontend JavaScript for FUCO AI Aggregator
            class FUCOFrontend {
                constructor() {
                    this.currentTab = 'selected';
                    this.loadCostMetrics();
                }

                async askQuestion() {
                    const questionInput = document.getElementById('questionInput');
                    const askButton = document.getElementById('askButton');
                    const question = questionInput.value.trim();

                    if (!question) {
                        alert('Please enter a question');
                        return;
                    }

                    // Disable button and show loading
                    askButton.disabled = true;
                    askButton.innerHTML = '<div class="loading"></div> Asking...';

                    // Add user message to chat
                    this.addMessage(question, 'user');

                    try {
                        const response = await fetch('/ask', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({ question })
                        });

                        const data = await response.json();

                        if (data.success) {
                            this.displayResults(data.result);
                            this.loadCostMetrics();
                        } else {
                            throw new Error(data.error || 'Failed to get response');
                        }

                    } catch (error) {
                        this.addMessage(`Error: ${error.message}`, 'error');
                    } finally {
                        // Re-enable button
                        askButton.disabled = false;
                        askButton.textContent = 'Ask FUCO';
                        questionInput.value = '';
                    }
                }

                addMessage(content, type = 'ai') {
                    const chatMessages = document.getElementById('chatMessages');
                    const messageDiv = document.createElement('div');
                    messageDiv.className = `message ${type}`;
                    
                    if (type === 'user') {
                        messageDiv.innerHTML = `<strong>You:</strong> ${content}`;
                    } else if (type === 'error') {
                        messageDiv.innerHTML = `<strong>Error:</strong> ${content}`;
                        messageDiv.style.borderLeftColor = '#ef4444';
                    } else {
                        messageDiv.innerHTML = `<strong>FUCO:</strong> ${content}`;
                    }
                    
                    chatMessages.appendChild(messageDiv);
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                }

                displayResults(result) {
                    // Display selected response
                    const selectedResponse = document.getElementById('selectedResponse');
                    selectedResponse.innerHTML = this.formatSelectedResponse(result.selected_response);

                    // Display ensemble response
                    const ensembleResponse = document.getElementById('ensembleResponse');
                    ensembleResponse.innerHTML = this.formatEnsembleResponse(result.ensemble_response);

                    // Display all responses
                    const allResponses = document.getElementById('allResponses');
                    allResponses.innerHTML = this.formatAllResponses(result.all_responses);

                    // Switch to selected tab
                    this.switchTab('selected');
                }

                formatSelectedResponse(response) {
                    return `
                        <div class="response-item">
                            <div class="response-header">
                                <span class="ai-name">${response.source} ??</span>
                                <span class="confidence">Confidence: ${(response.confidence * 100).toFixed(1)}%</span>
                            </div>
                            <div class="response-content">
                                ${this.formatText(response.response)}
                            </div>
                            <div class="cost">Cost: $${response.cost.toFixed(4)}</div>
                        </div>
                    `;
                }

                formatEnsembleResponse(response) {
                    return `
                        <div class="response-item">
                            <div class="response-header">
                                <span class="ai-name">?? Ensemble Intelligence</span>
                                <span class="confidence">Combined View</span>
                            </div>
                            <div class="response-content">
                                ${this.formatText(response)}
                            </div>
                        </div>
                    `;
                }

                formatAllResponses(responses) {
                    return responses.map(response => `
                        <div class="response-item">
                            <div class="response-header">
                                <span class="ai-name">${response.source}</span>
                                <span class="confidence">${(response.confidence * 100).toFixed(1)}%</span>
                            </div>
                            <div class="response-content">
                                ${this.formatText(response.response)}
                            </div>
                            <div class="cost">Cost: $${response.cost.toFixed(4)}</div>
                        </div>
                    `).join('');
                }

                formatText(text) {
                    // Simple text formatting
                    return text
                        .replace(/</g, '&lt;')
                        .replace(/>/g, '&gt;')
                        .replace(/\n/g, '<br>')
                        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                        .replace(/\*(.*?)\*/g, '<em>$1</em>');
                }

                switchTab(tabName) {
                    // Update tab buttons
                    document.querySelectorAll('.tab-button').forEach(btn => {
                        btn.classList.remove('active');
                    });
                    document.querySelector(`[onclick="switchTab('${tabName}')"]`).classList.add('active');

                    // Update tab panes
                    document.querySelectorAll('.tab-pane').forEach(pane => {
                        pane.classList.remove('active');
                    });
                    document.getElementById(`${tabName}-tab`).classList.add('active');

                    this.currentTab = tabName;

                    // Load history if needed
                    if (tabName === 'history') {
                        this.loadConversationHistory();
                    }
                }

                async loadConversationHistory() {
                    try {
                        const response = await fetch('/history');
                        const data = await response.json();

                        const historyContainer = document.getElementById('conversationHistory');
                        historyContainer.innerHTML = this.formatHistory(data.history, data.cost_metrics);

                    } catch (error) {
                        console.error('Error loading history:', error);
                    }
                }

                formatHistory(history, costMetrics) {
                    if (history.length === 0) {
                        return '<div class="response-item">No conversation history yet.</div>';
                    }

                    let html = `
                        <div class="response-item" style="background: rgba(34, 197, 94, 0.1); border-left: 4px solid #22c55e;">
                            <div class="response-header">
                                <span class="ai-name">?? Cost Analytics</span>
                            </div>
                            <div class="response-content">
                                Total Conversations: ${costMetrics.total_conversations}<br>
                                Estimated Total Cost: $${costMetrics.estimated_total_cost}<br>
                                Average Cost per Conversation: $${costMetrics.cost_per_conversation}<br>
                                Average Processing Time: ${costMetrics.average_processing_time}s
                            </div>
                        </div>
                    `;

                    history.reverse().forEach(conv => {
                        const date = new Date(conv.timestamp * 1000).toLocaleString();
                        html += `
                            <div class="response-item">
                                <div class="response-header">
                                    <span class="ai-name">??? Conversation</span>
                                    <span class="confidence">${date}</span>
                                </div>
                                <div class="response-content">
                                    <strong>Question:</strong> ${conv.question}<br><br>
                                    <strong>Processing Time:</strong> ${conv.processing_time.toFixed(2)}s<br>
                                    <strong>Strategy:</strong> ${conv.responses.strategy_used}
                                </div>
                            </div>
                        `;
                    });

                    return html;
                }

                async loadCostMetrics() {
                    try {
                        const response = await fetch('/history');
                        const data = await response.json();
                        
                        const costElement = document.getElementById('costMetrics');
                        costElement.textContent = `Cost: $${data.cost_metrics.estimated_total_cost}`;
                        
                    } catch (error) {
                        console.error('Error loading cost metrics:', error);
                    }
                }

                async clearHistory() {
                    if (confirm('Are you sure you want to clear all conversation history?')) {
                        try {
                            await fetch('/clear-history', { method: 'POST' });
                            this.loadCostMetrics();
                            if (this.currentTab === 'history') {
                                this.loadConversationHistory();
                            }
                            document.getElementById('chatMessages').innerHTML = '';
                        } catch (error) {
                            console.error('Error clearing history:', error);
                        }
                    }
                }
            }

            // Initialize the frontend
            const fuco = new FUCOFrontend();

            // Global functions for HTML onclick
            function askQuestion() {
                fuco.askQuestion();
            }

            function switchTab(tabName) {
                fuco.switchTab(tabName);
            }

            function clearHistory() {
                fuco.clearHistory();
            }

            // Enter key support
            document.getElementById('questionInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    askQuestion();
                }
            });

            // Load initial history if on history tab
            document.addEventListener('DOMContentLoaded', function() {
                if (fuco.currentTab === 'history') {
                    fuco.loadConversationHistory();
                }
            });
        </script>
    </body>
    </html>
    '''
    
    @app.route('/')
    def index():
        return render_template_string(HTML_TEMPLATE)
    
    @app.route('/health')
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'service': 'FUCO AI Aggregator',
            'version': '1.0.0',
            'timestamp': datetime.now().isoformat()
        })
    
    @app.route('/ask', methods=['POST'])
    def ask_question():
        """Main endpoint to ask questions to all AIs"""
        try:
            data = request.get_json()
            question = data.get('question', '').strip()
            
            if not question:
                return jsonify({'error': 'Question is required'}), 400
            
            # Get AI responses
            result = run_async(aggregator.get_ai_responses(question))
            
            return jsonify({
                'success': True,
                'question': question,
                'result': result,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/history', methods=['GET'])
    def get_history():
        """Get conversation history"""
        history = aggregator.get_conversation_history()
        cost_metrics = aggregator.calculate_cost_savings()
        
        return jsonify({
            'history': history,
            'cost_metrics': cost_metrics,
            'total_requests': len(history)
        })
    
    @app.route('/clear-history', methods=['POST'])
    def clear_history():
        """Clear conversation history"""
        aggregator.conversation_history = []
        return jsonify({'success': True, 'message': 'History cleared'})
    
    @app.route('/stats', methods=['GET'])
    def get_stats():
        """Get service statistics"""
        history = aggregator.get_conversation_history()
        cost_metrics = aggregator.calculate_cost_savings()
        
        return jsonify({
            'total_conversations': len(history),
            'total_cost': cost_metrics['estimated_total_cost'],
            'average_response_time': cost_metrics['average_processing_time'],
            'active_models': ['ChatGPT', 'Gemini', 'DeepSeek'],
            'service_uptime': time.time() - app_start_time
        })
    
    def run_async(coro):
        """Run async functions in sync context"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    
    return app

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    app_start_time = time.time()
    
    print("?? Starting FUCO AI Aggregator...")
    print("?? Make sure to set these environment variables:")
    print("   - OPENAI_API_KEY")
    print("   - GEMINI_API_KEY") 
    print("   - DEEPSEEK_API_KEY")
    print("?? Access the application at: http://localhost:5000")
    print("=" * 50)
    
    app = create_app()
    app.run(
        host=os.getenv('HOST', '0.0.0.0'),
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('DEBUG', 'False').lower() == 'true'
    )
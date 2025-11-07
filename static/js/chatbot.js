class Cardia4Chatbot {
    constructor() {
        this.isOpen = false;
        this.apiKey = localStorage.getItem('cardia4_api_key') || '';
        this.messages = [];
        this.websiteKnowledge = this.initWebsiteKnowledge();
        this.init();
    }

    init() {
        this.createChatWidget();
        this.attachEventListeners();
    }

    initWebsiteKnowledge() {
        return {
            "how to use": "Cardia4 offers comprehensive heart disease prediction through multiple AI models. Visit /predict to start your assessment, upload medical scans (X-ray, MRI, ECG), and get personalized risk analysis.",
            "features": "Our platform includes: ML Models for vital signs analysis, Medical Imaging analysis using deep learning, Care Classification for treatment recommendations, and Risk Tracking dashboards.",
            "navigation": {
                "/predict": "Get comprehensive heart risk assessment by uploading scans and entering health data",
                "/dashboard": "View your prediction history, results, and track your cardiovascular health over time",
                "/incare_outcare": "Get care recommendations based on blood work analysis",
                "/about": "Learn about our AI models, accuracy rates, and the science behind Cardia4"
            },
            "models": "We use 5+ integrated AI models including ResNet50 for cardiac MRI analysis, Random Forest for ML predictions, and deep learning models for ECG analysis with 99.2% overall accuracy.",
            "data_required": "You'll need: Personal information (age, sex), vital signs (blood pressure, cholesterol), medical scans (X-ray, MRI, ECG), and optionally blood work for care classification.",
            "accuracy": "Our AI models achieve 99.2% accuracy across multiple prediction types, combining machine learning with deep learning for comprehensive analysis.",
            "emergency": "If you're experiencing chest pain, shortness of breath, or other emergency symptoms, please call emergency services immediately. Cardia4 is for risk assessment, not emergency diagnosis."
        };
    }

    createChatWidget() {
        const chatWidget = document.createElement('div');
        chatWidget.innerHTML = `
            <!-- Chat Button -->
            <div id="chat-button" class="fixed bottom-6 right-6 z-50 cursor-pointer transform transition-all duration-300 hover:scale-105">
                <div class="w-16 h-16 bg-gradient-to-r from-primary-600 to-accent-600 rounded-full shadow-lg flex items-center justify-center text-white hover:shadow-xl">
                    <i data-lucide="message-circle" class="w-7 h-7"></i>
                </div>
                <div class="absolute -top-2 -right-2 w-6 h-6 bg-red-500 rounded-full flex items-center justify-center">
                    <span class="text-white text-xs font-bold">AI</span>
                </div>
            </div>

            <!-- Chat Window -->
            <div id="chat-window" class="fixed bottom-20 right-6 z-50 w-96 h-[550px] bg-white rounded-2xl shadow-2xl border border-gray-200 transform transition-all duration-300 scale-0 origin-bottom-right">
                <!-- Chat Header -->
                <div class="bg-gradient-to-r from-primary-600 to-accent-600 text-white p-4 rounded-t-2xl">
                    <div class="flex items-center justify-between">
                        <div class="flex items-center space-x-3">
                            <div class="w-10 h-10 bg-white/20 rounded-full flex items-center justify-center">
                                <i data-lucide="bot" class="w-5 h-5"></i>
                            </div>
                            <div>
                                <h3 class="font-semibold">Cardia4 Assistant</h3>
                                <p class="text-sm opacity-90">Ask me about heart health & website</p>
                            </div>
                        </div>
                        <button id="chat-close" class="p-2 hover:bg-white/20 rounded-lg transition-colors">
                            <i data-lucide="x" class="w-5 h-5"></i>
                        </button>
                    </div>
                </div>


                <!-- Chat Messages -->
                <div id="chat-messages" class="flex-1 p-4 space-y-4 overflow-y-auto h-[420px]">
                    <div class="flex items-start space-x-3">
                        <div class="w-8 h-8 bg-gradient-to-r from-primary-500 to-accent-500 rounded-full flex items-center justify-center flex-shrink-0">
                            <i data-lucide="bot" class="w-4 h-4 text-white"></i>
                        </div>
                        <div class="bg-gray-100 rounded-2xl rounded-tl-md p-3 max-w-[280px]">
                            <p class="text-sm text-gray-800">Hi! I'm your Cardia4 assistant. I can help you with:</p>
                            <ul class="text-sm text-gray-700 mt-2 space-y-1">
                                <li>• Website navigation & features</li>
                                <li>• Heart health information</li>
                                <li>• How to use our AI models</li>
                                <li>• Understanding your results</li>
                            </ul>
                            <p class="text-xs text-gray-600 mt-2">Ask me anything! Type your question below.</p>
                        </div>
                    </div>
                </div>

                <!-- Chat Input -->
                <div class="p-4 border-t border-gray-200">
                    <div class="flex space-x-2">
                        <input type="text" id="chat-input" placeholder="Ask about heart health or website features..." 
                               class="flex-1 px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent">
                        <button id="send-message" class="px-4 py-3 bg-gradient-to-r from-primary-600 to-accent-600 text-white rounded-xl hover:shadow-lg transition-all duration-200 flex items-center justify-center">
                            <i data-lucide="send" class="w-5 h-5"></i>
                        </button>
                    </div>
                    <div class="flex flex-wrap gap-2 mt-2">
                        <button class="quick-question px-3 py-1 bg-gray-100 text-gray-700 rounded-full text-xs hover:bg-gray-200 transition-colors">How to get started?</button>
                        <button class="quick-question px-3 py-1 bg-gray-100 text-gray-700 rounded-full text-xs hover:bg-gray-200 transition-colors">Upload medical scans</button>
                        <button class="quick-question px-3 py-1 bg-gray-100 text-gray-700 rounded-full text-xs hover:bg-gray-200 transition-colors">Heart health tips</button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(chatWidget);
        
        // Initialize icons after adding to DOM
        setTimeout(() => lucide.createIcons(), 100);
    }

    attachEventListeners() {
        // Chat button toggle
        document.getElementById('chat-button').addEventListener('click', () => {
            this.toggleChat();
        });

        // Close chat
        document.getElementById('chat-close').addEventListener('click', () => {
            this.closeChat();
        });


        // Send message
        document.getElementById('send-message').addEventListener('click', () => {
            this.sendMessage();
        });

        // Enter key to send
        document.getElementById('chat-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendMessage();
            }
        });

        // Quick questions
        document.querySelectorAll('.quick-question').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const question = e.target.textContent;
                document.getElementById('chat-input').value = question;
                this.sendMessage();
            });
        });
    }

    toggleChat() {
        const chatWindow = document.getElementById('chat-window');
        const chatButton = document.getElementById('chat-button');
        
        if (this.isOpen) {
            this.closeChat();
        } else {
            this.openChat();
        }
    }

    openChat() {
        const chatWindow = document.getElementById('chat-window');
        const chatButton = document.getElementById('chat-button');
        
        chatWindow.classList.remove('scale-0');
        chatWindow.classList.add('scale-100');
        chatButton.style.display = 'none';
        this.isOpen = true;
    }

    closeChat() {
        const chatWindow = document.getElementById('chat-window');
        const chatButton = document.getElementById('chat-button');
        
        chatWindow.classList.remove('scale-100');
        chatWindow.classList.add('scale-0');
        chatButton.style.display = 'block';
        this.isOpen = false;
    }


    async sendMessage() {
        const input = document.getElementById('chat-input');
        const message = input.value.trim();
        
        if (!message) return;
        
        // Add user message
        this.addMessage('user', message);
        input.value = '';
        
        // Show typing indicator
        this.showTyping();
        
        try {
            // Always try API first (will use server-side API key)
            let response = await this.getApiResponse(message);
            
            this.hideTyping();
            this.addMessage('bot', response);
            
        } catch (error) {
            this.hideTyping();
            console.error('Chat error:', error);
            
            // Fallback to local response if API fails
            const fallbackResponse = this.getLocalResponse(message);
            this.addMessage('bot', fallbackResponse);
        }
    }

    async getApiResponse(message) {
        console.log('Sending API request:', message);
        
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message
            })
        });
        
        console.log('API response status:', response.status);
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('API error:', errorText);
            throw new Error(`API request failed: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('API response data:', data);
        return data.response;
    }

    getLocalResponse(message) {
        const lowerMessage = message.toLowerCase();
        
        // Website navigation help
        if (lowerMessage.includes('how') && (lowerMessage.includes('use') || lowerMessage.includes('start'))) {
            return `To get started with Cardia4:\n\n1. Visit the **Predict** page (/predict) to begin your heart health assessment\n2. Upload your medical scans (X-ray, MRI, ECG) if available\n3. Fill in your health information (age, vital signs, medical history)\n4. Get your comprehensive risk analysis\n5. View results in your **Dashboard** (/dashboard)\n\nOur AI models will analyze your data and provide personalized cardiovascular risk assessment with 99.2% accuracy.`;
        }
        
        if (lowerMessage.includes('upload') && lowerMessage.includes('scan')) {
            return `To upload medical scans in Cardia4:\n\n📋 **Supported Formats:**\n• X-ray images (chest X-rays)\n• MRI scans (cardiac MRI)\n• ECG reports/images\n\n📁 **How to Upload:**\n1. Go to the Predict page (/predict)\n2. Look for the file upload sections\n3. Drag & drop or click to select your medical files\n4. Our AI models will automatically analyze them\n\n🎯 **AI Analysis:**\n• ResNet50 processes cardiac MRI scans\n• Deep learning models analyze ECG data\n• Combined with your health data for comprehensive assessment`;
        }
        
        if (lowerMessage.includes('dashboard') || lowerMessage.includes('results')) {
            return `Your **Dashboard** (/dashboard) shows:\n\n📊 **Prediction History:**\n• All your previous risk assessments\n• Risk scores from different AI models\n• Timeline of your health evaluations\n\n📈 **Risk Tracking:**\n• ML Model predictions\n• Medical imaging analysis results\n• ECG analysis outcomes\n• Combined final risk score\n\n💡 **Care Recommendations:**\n• Visit /incare_outcare for blood work analysis\n• Get personalized treatment path recommendations\n• Track your cardiovascular health over time`;
        }
        
        if (lowerMessage.includes('heart') || lowerMessage.includes('cardiovascular')) {
            return `🫀 **Heart Health Information:**\n\n**Risk Factors:**\n• High blood pressure & cholesterol\n• Smoking, diabetes, obesity\n• Family history, age, sedentary lifestyle\n\n**Prevention Tips:**\n• Regular exercise (150min/week)\n• Healthy diet (low sodium, more fruits/vegetables)\n• Maintain healthy weight\n• Don't smoke, limit alcohol\n• Manage stress & get adequate sleep\n\n**When to Seek Care:**\n• Chest pain or pressure\n• Shortness of breath\n• Irregular heartbeat\n• Dizziness or fainting\n\n⚠️ **Emergency:** Call 911 for severe chest pain, difficulty breathing, or heart attack symptoms.`;
        }
        
        if (lowerMessage.includes('accuracy') || lowerMessage.includes('model')) {
            return `🤖 **Cardia4 AI Models:**\n\n**Overall Accuracy:** 99.2% across 5+ integrated models\n\n**Model Types:**\n• **ResNet50:** Cardiac MRI analysis\n• **Random Forest:** ML predictions from vital signs\n• **Deep Learning:** ECG pattern recognition\n• **Ensemble Method:** Combines all models for final prediction\n\n**What We Analyze:**\n• Medical imaging (X-ray, MRI, ECG)\n• Vital signs & health metrics\n• Medical history & risk factors\n• Blood work (for care classification)\n\n**Validation:** Our models are trained on extensive medical datasets and validated by healthcare professionals.`;
        }
        
        if (lowerMessage.includes('api') || lowerMessage.includes('key')) {
            return `🔑 **API Key Setup:**\n\nTo enable enhanced AI responses:\n\n1. Get a free Gemini API key from Google AI Studio\n2. Enter it in the API Configuration section above\n3. Click "Save" to store it locally\n\n**Benefits with API:**\n• More detailed medical explanations\n• Personalized health advice\n• Context-aware responses\n• Latest medical information\n\n**Without API:**\n• Basic website navigation help\n• General heart health information\n• Offline functionality\n\nYour API key is stored locally and never shared.`;
        }
        
        // Default response
        return `I can help you with:\n\n🏥 **Website Navigation:**\n• How to use Cardia4 features\n• Upload medical scans\n• View your dashboard & results\n\n❤️ **Heart Health:**\n• Risk factors & prevention\n• Understanding symptoms\n• When to seek medical care\n\n🤖 **AI Models:**\n• How our predictions work\n• Model accuracy & validation\n\nTry asking: "How do I get started?" or "Tell me about heart health"\n\n💡 Add a Gemini API key above for enhanced responses!`;
    }

    addMessage(sender, content) {
        const messagesContainer = document.getElementById('chat-messages');
        const messageDiv = document.createElement('div');
        
        if (sender === 'user') {
            messageDiv.innerHTML = `
                <div class="flex items-start space-x-3 justify-end">
                    <div class="bg-gradient-to-r from-primary-600 to-accent-600 text-white rounded-2xl rounded-tr-md p-3 max-w-[280px]">
                        <p class="text-sm">${content}</p>
                    </div>
                    <div class="w-8 h-8 bg-gray-300 rounded-full flex items-center justify-center flex-shrink-0">
                        <i data-lucide="user" class="w-4 h-4 text-gray-600"></i>
                    </div>
                </div>
            `;
        } else {
            messageDiv.innerHTML = `
                <div class="flex items-start space-x-3">
                    <div class="w-8 h-8 bg-gradient-to-r from-primary-500 to-accent-500 rounded-full flex items-center justify-center flex-shrink-0">
                        <i data-lucide="bot" class="w-4 h-4 text-white"></i>
                    </div>
                    <div class="bg-gray-100 rounded-2xl rounded-tl-md p-3 max-w-[280px]">
                        <div class="text-sm text-gray-800 whitespace-pre-line">${content}</div>
                    </div>
                </div>
            `;
        }
        
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        
        // Initialize icons for new message
        lucide.createIcons();
    }

    showTyping() {
        const messagesContainer = document.getElementById('chat-messages');
        const typingDiv = document.createElement('div');
        typingDiv.id = 'typing-indicator';
        typingDiv.innerHTML = `
            <div class="flex items-start space-x-3">
                <div class="w-8 h-8 bg-gradient-to-r from-primary-500 to-accent-500 rounded-full flex items-center justify-center flex-shrink-0">
                    <i data-lucide="bot" class="w-4 h-4 text-white"></i>
                </div>
                <div class="bg-gray-100 rounded-2xl rounded-tl-md p-3">
                    <div class="flex space-x-1">
                        <div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                        <div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 0.1s"></div>
                        <div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
                    </div>
                </div>
            </div>
        `;
        
        messagesContainer.appendChild(typingDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        lucide.createIcons();
    }

    hideTyping() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
}

// Initialize chatbot when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, initializing chatbot...');
    try {
        new Cardia4Chatbot();
        console.log('Chatbot initialized successfully');
    } catch (error) {
        console.error('Chatbot initialization failed:', error);
    }
});

// ------------------------------
// Chatbot.js
// ------------------------------

// Mobile Menu Toggle
const menuToggle = document.getElementById('menu-toggle');
const mobileMenu = document.getElementById('mobile-menu');

if (menuToggle && mobileMenu) {
    menuToggle.addEventListener('click', () => {
        mobileMenu.classList.toggle('hidden');
    });
}

// Smooth Scrolling for Anchor Links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();

        const targetId = this.getAttribute('href');
        if (targetId === '#') return;

        const targetElement = document.querySelector(targetId);
        if (targetElement) {
            window.scrollTo({
                top: targetElement.offsetTop - 80,
                behavior: 'smooth'
            });

            if (!mobileMenu.classList.contains('hidden')) {
                mobileMenu.classList.add('hidden');
            }
        }
    });
});

// Back to Top Button
const backToTopButton = document.getElementById('back-to-top');
if (backToTopButton) {
    window.addEventListener('scroll', () => {
        if (window.pageYOffset > 300) {
            backToTopButton.classList.remove('opacity-0', 'invisible');
            backToTopButton.classList.add('opacity-100', 'visible');
        } else {
            backToTopButton.classList.remove('opacity-100', 'visible');
            backToTopButton.classList.add('opacity-0', 'invisible');
        }
    });

    backToTopButton.addEventListener('click', () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
}

// Header Scroll Effect
const header = document.querySelector('header');
if (header) {
    window.addEventListener('scroll', () => {
        if (window.pageYOffset > 50) {
            header.classList.add('shadow-md', 'py-3');
            header.classList.remove('py-4');
        } else {
            header.classList.remove('shadow-md', 'py-3');
            header.classList.add('py-4');
        }
    });
}

// Chatbot functionality
const chatbotWidget = document.getElementById('chatbot-widget');
const chatbotContainer = document.getElementById('chatbot-container');
const closeChatbot = document.getElementById('close-chatbot');
const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const sendMessageBtn = document.getElementById('send-message');
const useStreaming = document.getElementById('use-streaming');
const openChatbotFromService = document.getElementById('open-chatbot-from-service');

if (chatbotWidget && chatbotContainer) {
    chatbotWidget.addEventListener('click', () => {
        chatbotContainer.classList.toggle('hidden');
    });
}

if (openChatbotFromService) {
    openChatbotFromService.addEventListener('click', () => {
        chatbotContainer.classList.remove('hidden');
    });
}

if (closeChatbot) {
    closeChatbot.addEventListener('click', () => {
        chatbotContainer.classList.add('hidden');
    });
}

// Function to add a message to the chat
function addMessage(content, isUser = false) {
    if (!chatMessages) return;

    const messageDiv = document.createElement('div');
    messageDiv.classList.add('flex', isUser ? 'justify-end' : 'justify-start');

    const messageBubble = document.createElement('div');
    messageBubble.classList.add(
        'max-w-xs', 'lg:max-w-md', 'px-4', 'py-2', 'rounded-lg',
        isUser ? 'bg-primary text-white' : 'bg-gray-100 text-gray-800'
    );
    messageBubble.textContent = content;

    messageDiv.appendChild(messageBubble);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Function to handle streaming responses
async function handleStreamingResponse(response, messageId) {
    if (!chatMessages) return;

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let messageElement = document.getElementById(messageId);

    if (!messageElement) {
        messageElement = document.createElement('div');
        messageElement.id = messageId;
        messageElement.classList.add('flex', 'justify-start');
        chatMessages.appendChild(messageElement);
    }

    let fullResponse = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');

        for (const line of lines) {
            if (line.startsWith('data:') && line !== 'data: [DONE]') {
                try {
                    const data = JSON.parse(line.substring(5));
                    if (data.response) {
                        fullResponse += data.response;
                        messageElement.innerHTML = `
                            <div class="bg-gray-100 text-gray-800 px-4 py-2 rounded-lg max-w-xs lg:max-w-md">
                                ${fullResponse}
                            </div>
                        `;
                        chatMessages.scrollTop = chatMessages.scrollHeight;
                    }
                } catch (e) {
                    console.error('Error parsing stream data:', e);
                }
            }
        }
    }
}

// Function to handle sending messages
async function sendMessage() {
    if (!chatInput) return;

    const message = chatInput.value.trim();
    if (!message) return;

    addMessage(message, true);
    chatInput.value = '';

    const typingIndicator = document.createElement('div');
    typingIndicator.classList.add('flex', 'justify-start');
    typingIndicator.innerHTML = `
        <div class="bg-gray-100 text-gray-800 px-4 py-2 rounded-lg">
            <div class="flex space-x-1">
                <div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                <div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
                <div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 0.4s"></div>
            </div>
        </div>
    `;
    chatMessages.appendChild(typingIndicator);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        const pingResponse = await fetch('https://hydrogen-biology-acceptable-situation.trycloudflare.com', { method: 'GET' });

        if (!pingResponse.ok) throw new Error('Servidor Ollama não está respondendo');

        const messageId = 'msg-' + Date.now();
        const model = 'gemma3';

        if (useStreaming && useStreaming.checked) {
            const response = await fetch('https://hydrogen-biology-acceptable-situation.trycloudflare.com/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model, prompt: message, stream: true })
            });

            if (!response.ok) throw new Error(`Erro no servidor: ${response.status}`);

            chatMessages.removeChild(typingIndicator);
            await handleStreamingResponse(response, messageId);
        } else {
            const response = await fetch('https://hydrogen-biology-acceptable-situation.trycloudflare.com/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model, prompt: message, stream: false })
            });

            if (!response.ok) throw new Error(`Erro no servidor: ${response.status}`);

            const data = await response.json();
            chatMessages.removeChild(typingIndicator);
            addMessage(data.response, false);
        }
    } catch (error) {
        console.error('Erro:', error);
        chatMessages.removeChild(typingIndicator);
        addMessage(`Erro ao conectar ao Ollama: ${error.message}`, false);
    }
}

// Event listeners for sending messages
if (sendMessageBtn) sendMessageBtn.addEventListener('click', sendMessage);
if (chatInput) chatInput.addEventListener('keypress', e => { if (e.key === 'Enter') sendMessage(); });

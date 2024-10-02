// scripts.js

// Variável para armazenar o histórico da conversa
let conversationHistory = [
    { role: 'system', content: 'Responda de maneira prática, simples e útil. Evite respostas longas e complexas.' }
];

// Função para enviar mensagem para o chatbot (OpenAI)
async function sendMessage() {
    const userInput = document.getElementById('userInput').value;
    if (!userInput) return;

    // Exibir a mensagem do usuário na caixa de chat
    displayMessage(userInput, 'user');

    // Adicionar a mensagem do usuário ao histórico da conversa
    conversationHistory.push({ role: 'user', content: userInput });

    // Limpar a entrada do usuário
    document.getElementById('userInput').value = '';

    try {
        // Fazer a chamada para o backend que se comunica com a API da OpenAI
        const response = await fetch('http://localhost:3000/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ messages: conversationHistory }),
        });

        if (!response.ok) {
            throw new Error('Erro ao se comunicar com o servidor');
        }

        const data = await response.json();
        const botMessage = data.botMessage;

        // Adicionar a resposta do bot ao histórico da conversa
        conversationHistory.push({ role: 'assistant', content: botMessage });

        // Exibir a resposta do bot na caixa de chat
        displayMessage(botMessage, 'bot');
    } catch (error) {
        console.error('Erro:', error);
        displayMessage('Desculpe, houve um erro ao processar sua solicitação.', 'bot');
    }
}

// Função para exibir as mensagens na caixa de chat
function displayMessage(message, sender) {
    const chatbox = document.getElementById('chatbox');
    const messageElement = document.createElement('div');
    messageElement.classList.add('chat-message', sender);
    messageElement.textContent = message;
    chatbox.appendChild(messageElement);
    chatbox.scrollTop = chatbox.scrollHeight; // Rolar para o final da caixa de chat
}

// Evento para enviar a mensagem quando o botão é clicado
document.getElementById('sendButton').addEventListener('click', sendMessage);

// Evento para enviar a mensagem quando a tecla Enter é pressionada
document.getElementById('userInput').addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

// scripts.js

// Função para enviar mensagem para o chatbot (simulação por enquanto)
async function sendMessage() {
    const userInput = document.getElementById('userInput').value;
    if (!userInput) return;

    // Exibir a mensagem do usuário na caixa de chat
    displayMessage(userInput, 'user');

    // Limpar a entrada do usuário
    document.getElementById('userInput').value = '';

    // Simular uma resposta do chatbot
    setTimeout(() => {
        const botMessage = 'Esta é uma resposta simulada do chatbot.';
        displayMessage(botMessage, 'bot');
    }, 1000);
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

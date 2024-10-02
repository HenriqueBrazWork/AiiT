// scripts.js

// Função para enviar mensagem para a API OpenAI
async function sendMessage() {
    const userInput = document.getElementById('userInput').value;
    if (!userInput) return;

    // Exibir a mensagem do usuário na caixa de chat
    displayMessage(userInput, 'user');

    // Limpar a entrada do usuário
    document.getElementById('userInput').value = '';

    try {
        // Chamada à API OpenAI
        const response = await fetch('https://api.openai.com/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer sk-svcacct-3uTTZNii3HJ_J8LKLR2ikLi1UeU85AB5ACvkfu3vW3zgzl7d4SJ5lhrF-TtsTT3BlbkFJEbNWJVv0xv_yED6lzSFH6Mr2k_qqa_OZBq78tqfJiwlVj3Q5Osz-bMIa0ePbgA` // Substitua pela sua chave da API
            },
            body: JSON.stringify({
                model: 'gpt-4',
                messages: [{ role: 'user', content: userInput }],
            })
        });

        if (!response.ok) {
            throw new Error('Erro ao se comunicar com a API');
        }

        const data = await response.json();
        const botMessage = data.choices[0].message.content;

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

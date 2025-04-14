// Enviar mensagem do usuário para o backend
function sendMessage() {
    const userInput = document.getElementById("userInput").value;
    if (!userInput) return;

    const chatBox = document.getElementById("chatbox");
    chatBox.innerHTML += `<div class="user-message">${userInput}</div>`;
    
    // Enviar para o backend Flask
    fetch("http://127.0.0.1:5000/chat", {  // Alterar para o seu URL se estiver em produção
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ input: userInput }),
    })
    .then(response => response.json())
    .then(data => {
        const botMessage = data.response;
        chatBox.innerHTML += `<div class="bot-message">${botMessage}</div>`;
        chatBox.scrollTop = chatBox.scrollHeight;  // Scroll para a última mensagem
    })
    .catch(error => {
        chatBox.innerHTML += `<div class="bot-message">Desculpe, houve um erro.</div>`;
    });

    // Limpar o campo de entrada
    document.getElementById("userInput").value = "";
}

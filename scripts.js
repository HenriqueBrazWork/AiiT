function sendMessage() {
    const userInput = document.getElementById("userInput").value;
    if (!userInput) return;

    const chatBox = document.getElementById("chatbox");
    chatBox.innerHTML += `<div class="user-message">${userInput}</div>`;
    
    // Log para ver o que está sendo enviado
    console.log("Enviando para o backend: ", userInput);

    // Enviar para o backend Flask
    fetch("http://127.0.0.1:5000/chat", {  
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ input: userInput }),
    })
    .then(response => response.json())
    .then(data => {
        console.log("Resposta do backend: ", data.response);  // Log para ver a resposta do backend
        const botMessage = data.response;
        chatBox.innerHTML += `<div class="bot-message">${botMessage}</div>`;
        chatBox.scrollTop = chatBox.scrollHeight;  // Scroll para a última mensagem
    })
    .catch(error => {
        console.error("Erro ao enviar a solicitação: ", error);  // Log de erro
        chatBox.innerHTML += `<div class="bot-message">Desculpe, houve um erro.</div>`;
    });

    // Limpar o campo de entrada
    document.getElementById("userInput").value = "";
}

<!-- frontend/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chatbot de Suporte ao Cliente</title>
</head>
<body>
    <h1>Chatbot de Suporte ao Cliente</h1>
    <label for="userMessage">Digite sua pergunta (exemplo: "Como rastrear meu pedido?"):</label><br><br>
    <input type="text" id="userMessage" placeholder="Pergunte algo"><br><br>
    <button onclick="enviarMensagem()">Enviar</button>

    <h3>Resposta:</h3>
    <div id="response"></div>

    <script>
        async function enviarMensagem() {
            const userMessage = document.getElementById("userMessage").value;
            const responseDiv = document.getElementById("response");

            const res = await fetch("http://localhost:8000/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ user_message: userMessage })
            });

            const data = await res.json();
            responseDiv.innerHTML = "<strong>Resposta do Chatbot:</strong> " + data.response;
        }
    </script>
</body>
</html>

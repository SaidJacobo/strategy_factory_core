document.getElementById("backtest-form").addEventListener("submit", async function (event) {
    event.preventDefault(); // Evitar que el formulario se envíe de manera tradicional

    if (window.editor) {
        document.getElementById("Code").value = window.editor.getValue();
    }

    const formData = new FormData(event.target);

    // Enviar el formulario con Fetch (POST) usando 'FormData'
    const form = document.getElementById("backtest-form");

    url = form.getAttribute("action")
    streamBaseUrl = form.getAttribute("data-stream-url");

    const response = await fetch(url, {
        method: 'POST',
        body: formData
    });

    const result = await response.json();
    if (result.task_id) {
        console.log(streamBaseUrl)
        console.log(result.task_id)
        listenToEvents(streamBaseUrl, result.task_id);
    }
});

function listenToEvents(stream_url, task_id) {
    url = stream_url + task_id
    const eventSource = new EventSource(url);
    const loadingIndicator = document.getElementById("loading-indicator");
    loadingIndicator.style.display = "block";
    const logContainer = document.getElementById("log-container");

    eventSource.onmessage = function (event) {
        const data = JSON.parse(event.data);
        console.log("Log recibido:", data);

        if (data.status === "finished") {
            console.log("Backtest finalizado.");
            loadingIndicator.style.display = "none";
            eventSource.close();
            return;
        }

        addLog(logContainer, data);
    };

    eventSource.onerror = function (event) {
        console.error("Error en SSE", event);
        loadingIndicator.style.display = "none";
        eventSource.close();
    };
}

function addLog(logContainer, data) {
    const logEntry = document.createElement("div");
    let message;

    if (data.status === "link") {
        message = `[${new Date().toLocaleTimeString()}] `;

        // Título opcional + link
        const label = data.label || "See behaviour";
        message += `<a href="${data.message}" target="_blank" style="color: #00bfff; text-decoration: underline;">${label}</a>`;
    } else {
        message = `[${new Date().toLocaleTimeString()}] ${data.ticker || ""} ${data.timeframe || ""} - ${data.message}`;
        
        if (data.status === "failed" && data.error) {
            message += `, Error: ${data.error}`;
        }

        message = message.replace(/\n/g, "<br>");
    }

    logEntry.innerHTML = message;

    switch (data.status.toLowerCase()) {
        case "failed":
            logEntry.style.color = "red";
            break;
        case "completed":
            logEntry.style.color = "green";
            break;
        case "warning":
            logEntry.style.color = "yellow";
            break;
        case "log":
            logEntry.style.color = "white";
            break;
        case "link":
            logEntry.style.color = "#00bfff"; // light blue, or let the link style handle it
            break;
        default:
            logEntry.style.color = "gray";
    }

    logContainer.appendChild(logEntry);

    setTimeout(() => {
        logContainer.scrollTop = logContainer.scrollHeight;
    }, 100);
}
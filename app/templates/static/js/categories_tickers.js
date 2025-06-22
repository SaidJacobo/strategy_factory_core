const categorySelect = document.getElementById('CategoryId');
const tickerSelect = document.getElementById('TickerId');

async function fetchTickers(categoryId) {
    try {
        const response = await fetch(`/categories/${categoryId}/tickers`, {
            headers: {
                'Accept': 'application/json'
            }
        });

        if (!response.ok) {
            throw new Error('Error al obtener los tickers');
        }
        return await response.json();
    } catch (error) {
        console.error(error);
        return [];
    }
}

categorySelect.addEventListener('change', async function () {
    const selectedCategory = categorySelect.value;

    // Limpia las opciones del select de tickers
    tickerSelect.innerHTML = '<option value>*</option>';

    if (selectedCategory !== '*') {
        const tickers = await fetchTickers(selectedCategory);
        tickers.forEach(ticker => {
            const option = document.createElement('option');
            option.value = ticker.Id;
            option.textContent = ticker.Name;
            tickerSelect.appendChild(option);
        });
    }
});
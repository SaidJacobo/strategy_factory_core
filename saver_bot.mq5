
#property strict

input int periods = 1000;           // Número de periodos a guardar


datetime lastBarTime = 0;


int OnInit()
{
    // Inicializar la variable lastBarTime con el tiempo de la última barra
    lastBarTime = iTime(NULL, 0, 0);
    return(INIT_SUCCEEDED);
}

void OnTick()
{
    // Obtener el tiempo de la última barra
    datetime currentBarTime = iTime(NULL, 0, 0);

    // Verificar si hay una nueva barra
    if (currentBarTime != lastBarTime)
    {
        // Actualizar lastBarTime
        lastBarTime = currentBarTime;

        // Llamar a la función para guardar los datos OHLC
        SaveOHLCData();
    }
}


void SaveOHLCData()
{
    int total_bars = Bars(NULL, 0);
    if (total_bars < periods)
    {
        Print("No hay suficientes barras en el gráfico.");
        return;
    }
    
    string symbol = Symbol();
    ENUM_TIMEFRAMES timeframe = Period();
    string timeframe_str = EnumToString(timeframe);
    StringReplace(timeframe_str, "PERIOD_", "");
    string file_name = symbol + "_" + timeframe_str + ".csv";

    string file_path = TerminalInfoString(TERMINAL_COMMONDATA_PATH) + "\\" + file_name;

    int file_handle = FileOpen(file_name, FILE_CSV | FILE_WRITE | FILE_COMMON | FILE_ANSI | FILE_REWRITE);
    if (file_handle == INVALID_HANDLE)
    {
        Print("Error al abrir el archivo ", file_name, ". Código de error: ", GetLastError());
        return;
    }
    
    // Escribir encabezados correctamente en CSV
    FileWrite(file_handle, "Date,Open,High,Low,Close");

    for (int i = periods - 1; i >= 1; i--)
    {
        datetime time = iTime(NULL, 0, i);
        double open = iOpen(NULL, 0, i);
        double high = iHigh(NULL, 0, i);
        double low = iLow(NULL, 0, i);
        double close = iClose(NULL, 0, i);

        // Formato "YYYY-MM-DD HH:MM:SS"
        string time_str = TimeToString(time, TIME_DATE | TIME_SECONDS);

        // Escribir línea correctamente separada por comas
        string line = StringFormat("%s,%.5f,%.5f,%.5f,%.5f", time_str, open, high, low, close);
        FileWrite(file_handle, line);
    }

    FileClose(file_handle);
    Print("Datos OHLC guardados en ", file_path);
}
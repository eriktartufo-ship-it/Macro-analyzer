@echo off
REM Avvia backend (FastAPI) e frontend (Vite) del Macro Analyzer.
REM Apre due finestre separate per i log; chiudile per fermare i servizi.

setlocal
set ROOT=%~dp0

echo.
echo ===============================================
echo  Macro Analyzer - Avvio in corso
echo ===============================================
echo.

REM --- Backend ---
start "Macro Analyzer - Backend" cmd /k ^
  "cd /d "%ROOT%backend" && "%ROOT%.venv\Scripts\python.exe" -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

REM --- Frontend ---
start "Macro Analyzer - Frontend" cmd /k ^
  "cd /d "%ROOT%frontend" && npm run dev"

echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3000
echo.
echo Attendo 6 secondi e apro il browser...
timeout /t 6 /nobreak >nul
start "" http://localhost:3000

endlocal

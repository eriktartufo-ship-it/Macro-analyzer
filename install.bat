@echo off
REM ====================================================================
REM Macro Analyzer - INSTALL (one-time, post-clone su nuova macchina)
REM
REM Da lanciare UNA volta dopo aver clonato il repo o quando cambia PC.
REM Per l'avvio quotidiano usa start.bat.
REM
REM Prerequisiti host (NON installati da questo script):
REM   - Python 3.12+   https://www.python.org/downloads/
REM   - Node 18+       https://nodejs.org/
REM   - Postgres 16+ in locale (per dev) — oppure usa Docker
REM ====================================================================

setlocal
pushd "%~dp0"

echo.
echo ====================================
echo  Macro Analyzer - INSTALL
echo ====================================
echo.

REM --- 1. Verifica tools host ---
where python >nul 2>&1
if errorlevel 1 (
    echo [ERRORE] Python non trovato in PATH. Installa Python 3.12+ da https://www.python.org/downloads/
    goto :error
)
where node >nul 2>&1
if errorlevel 1 (
    echo [ERRORE] Node non trovato in PATH. Installa Node 18+ da https://nodejs.org/
    goto :error
)
where npm >nul 2>&1
if errorlevel 1 (
    echo [ERRORE] npm non trovato in PATH (dovrebbe arrivare con Node).
    goto :error
)

REM --- 2. .env ---
if not exist ".env" (
    if exist ".env.example" (
        echo [INFO] .env mancante, copio da .env.example
        copy /Y ".env.example" ".env" >nul
        echo [ATTENZIONE] Apri .env e riempi le chiavi (FRED_API_KEY, GEMINI_API_KEY, ecc.) prima di start.bat
    ) else (
        echo [ERRORE] Manca anche .env.example
        goto :error
    )
) else (
    echo .env gia' presente, skip.
)

REM --- 3. Python venv ---
if not exist ".venv\Scripts\python.exe" (
    echo Creazione virtualenv .venv...
    python -m venv .venv
    if errorlevel 1 goto :error
) else (
    echo .venv gia' presente, skip creazione.
)

echo.
echo Aggiornamento pip + install requirements.txt...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error
".venv\Scripts\pip.exe" install -r requirements.txt
if errorlevel 1 goto :error

REM --- 4. Frontend deps ---
if exist "frontend\package.json" (
    echo.
    echo Installazione dipendenze frontend...
    pushd frontend
    if not exist "node_modules" (
        call npm install
        if errorlevel 1 (
            popd
            goto :error
        )
    ) else (
        echo node_modules gia' presente, skip npm install.
    )
    popd
)

REM --- 5. DB migrations ---
echo.
echo Applicazione migrazioni Alembic (richiede Postgres up + DATABASE_URL in .env)...
pushd backend
"..\\.venv\\Scripts\\alembic.exe" upgrade head
if errorlevel 1 (
    echo [ATTENZIONE] alembic upgrade ha fallito. Verifica che Postgres sia in esecuzione e che DATABASE_URL nel .env sia corretto.
    echo Puoi rilanciare a mano: cd backend ^&^& ..\.venv\Scripts\alembic upgrade head
)
popd

REM --- 6. Seed iniziale ---
echo.
echo Seed asset_regime_performance...
pushd backend
"..\\.venv\\Scripts\\python.exe" seed\seed_asset_regime.py
if errorlevel 1 (
    echo [ATTENZIONE] seed ha fallito. Probabilmente DB non raggiungibile. Rilancia con: cd backend ^&^& ..\.venv\Scripts\python seed\seed_asset_regime.py
)
popd

echo.
echo ============================================
echo  Install completato. Lancia start.bat
echo ============================================
echo.
echo Prossimi step:
echo   1. Verifica .env compilato (FRED_API_KEY almeno)
echo   2. Postgres locale running su :5432 con DB macro_analyzer
echo   3. start.bat per avviare backend + frontend
echo.
popd
endlocal
exit /b 0

:error
echo.
echo *** ERRORE durante l'install. ***
pause
popd
endlocal
exit /b 1

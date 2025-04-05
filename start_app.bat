@echo off
echo Starting AI Resume Ranking System...

echo Starting backend server...
start cmd /k "python Resume-Ranking-System/backend/main.py"

echo Waiting for backend to initialize...
timeout /t 5 /nobreak > nul

echo Starting frontend server...
start cmd /k "python Resume-Ranking-System/frontend/launch_frontend.py"

echo Opening browser...
timeout /t 2 /nobreak > nul
start http://localhost:5500

echo System started successfully!
echo Press Ctrl+C in the server windows to stop the servers when done. 
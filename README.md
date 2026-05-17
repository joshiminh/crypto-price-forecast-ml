# Project venv setup

Use a Python virtual environment (`venv`) for this project.

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Windows (cmd.exe):

```bat
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:

- Use Python 3.8 or newer.
- The `.venv/` folder is ignored by Git via `.gitignore`.

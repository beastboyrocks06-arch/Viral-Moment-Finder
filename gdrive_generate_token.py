
import json
from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/drive"]

BASE_DIR = Path(__file__).resolve().parent
CLIENT_FILE = BASE_DIR / "oauth-client.json"
TOKEN_FILE = BASE_DIR / "gdrive_token.json"

flow = InstalledAppFlow.from_client_secrets_file(
    str(CLIENT_FILE),
    SCOPES
)

credentials = flow.run_local_server(
    host="localhost",
    port=0,
    access_type="offline",
    prompt="consent"
)

TOKEN_FILE.write_text(credentials.to_json(), encoding="utf-8")

print("Google Drive OAuth token generated successfully!")
print(f"Token saved at: {TOKEN_FILE}")

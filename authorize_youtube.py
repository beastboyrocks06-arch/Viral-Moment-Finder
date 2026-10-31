
import json
from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow

BASE_DIR = Path(__file__).resolve().parent
CLIENT_FILE = BASE_DIR / "credentials.json"
TOKEN_FILE = BASE_DIR / "token.json"

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

def authorize():
    if not CLIENT_FILE.exists():
        raise FileNotFoundError(f"Missing OAuth credentials: {CLIENT_FILE}")

    flow = InstalledAppFlow.from_client_secrets_file(
        str(CLIENT_FILE),
        scopes=SCOPES
    )

    credentials = flow.run_local_server(
        host="localhost",
        port=8080,
        access_type="offline",
        prompt="consent",
        open_browser=True
    )

    if not credentials.refresh_token:
        raise RuntimeError(
            "Google did not return a refresh token. "
            "Revoke the app's previous access and authorize again."
        )

    TOKEN_FILE.write_text(
        credentials.to_json(),
        encoding="utf-8"
    )

    print("\nYouTube authorization successful!")
    print(f"Token saved: {TOKEN_FILE}")
    print("Refresh token available:", bool(credentials.refresh_token))

if __name__ == "__main__":
    authorize()

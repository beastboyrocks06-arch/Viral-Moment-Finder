import os
import json
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build


import os
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOKEN_FILE = os.path.join(BASE_DIR, "gdrive_token.json")
SCOPES = ["https://www.googleapis.com/auth/drive"]

def get_drive_service():
    if not os.path.exists(TOKEN_FILE):
        raise FileNotFoundError(f"OAuth token not found: {TOKEN_FILE}")

    credentials = Credentials.from_authorized_user_file(
        TOKEN_FILE,
        scopes=SCOPES
    )

    if not credentials.valid:
        if credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())

            with open(TOKEN_FILE, "w", encoding="utf-8") as f:
                f.write(credentials.to_json())
        else:
            raise RuntimeError("OAuth token is invalid. Generate a new token.")

    return build("drive", "v3", credentials=credentials, cache_discovery=False)

if __name__ == "__main__":
    service = get_drive_service()

    # Verify authenticated Google account
    user = service.about().get(fields="user").execute()
    print("Connected account:", user["user"]["emailAddress"])

    # Test access to your generated-clips folder
    FOLDER_ID = "1iWBO5UHo4E2dx0HqJ2uBVX8B1i7L5zBX"

    folder = service.files().get(
        fileId=FOLDER_ID,
        fields="id,name,mimeType"
    ).execute()

    print("Folder accessible:", folder["name"])

    # Test listing files
    results = service.files().list(
        q=f"'{FOLDER_ID}' in parents and trashed=false",
        fields="files(id,name,mimeType)",
        pageSize=5
    ).execute()

    print("\nFiles in folder:")
    for file in results.get("files", []):
        print("-", file["name"])


def get_drive_service():
    credentials_json = os.getenv("GDRIVE_CREDENTIALS")

    if credentials_json:
        credentials = Credentials.from_service_account_info(
            json.loads(credentials_json),
            scopes=["https://www.googleapis.com/auth/drive"]
        )
    else:
        credentials = Credentials.from_service_account_file(
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"],
            scopes=["https://www.googleapis.com/auth/drive"]
        )

    return build("drive", "v3", credentials=credentials)
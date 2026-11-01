import os
import json
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

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
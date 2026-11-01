
import os
import json
import io
import uuid
from pathlib import Path

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

BASE_DIR = Path(__file__).resolve().parent
SCOPES = ["https://www.googleapis.com/auth/drive"]

FOLDERS = {
    "Source": "1p8h8iXc0I-OK2bWTxd0poIyVY65b4iN-",
    "JSON": "12ES-HjP05UTewgVu5ZPfXkLP-4HQS9Tv",
    "Clips": "1iWBO5UHo4E2dx0HqJ2uBVX8B1i7L5zBX",
}


def get_credentials():
    token_json = os.getenv("GDRIVE_TOKEN_JSON")
    token_path = BASE_DIR / "gdrive_token.json"

    if token_json:
        credentials = Credentials.from_authorized_user_info(
            json.loads(token_json),
            scopes=SCOPES
        )
        print("[AUTH] Using GitHub/environment OAuth token.")

    elif token_path.exists():
        credentials = Credentials.from_authorized_user_file(
            str(token_path),
            scopes=SCOPES
        )
        print("[AUTH] Using local OAuth token.")

    else:
        raise FileNotFoundError("OAuth token not found.")

    if not credentials.valid:
        if credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())

            if not token_json:
                token_path.write_text(
                    credentials.to_json(),
                    encoding="utf-8"
                )

            print("[AUTH] Token refreshed.")
        else:
            raise RuntimeError("OAuth token invalid or expired.")

    return credentials


def test_drive():
    print("\n" + "=" * 60)
    print("GOOGLE DRIVE OAUTH TEST")
    print("=" * 60)

    credentials = get_credentials()

    print("[AUTH] Credential type:", credentials.__class__.__module__)

    if credentials.__class__.__module__ != "google.oauth2.credentials":
        raise RuntimeError("Not using Google user OAuth credentials.")

    service = build(
        "drive",
        "v3",
        credentials=credentials,
        cache_discovery=False
    )

    # Check connected Google account.
    account = service.about().get(
        fields="user(displayName,emailAddress)"
    ).execute()

    user = account.get("user", {})

    print("\n[ACCOUNT] Authentication successful.")
    print("[ACCOUNT] Name:", user.get("displayName"))
    print("[ACCOUNT] Email:", user.get("emailAddress"))

    # Check folder access.
    print("\n[FOLDERS] Checking Google Drive folders...")

    for name, folder_id in FOLDERS.items():
        if not folder_id:
            if name == "Clips":
                raise RuntimeError("GDRIVE_CLIPS_FOLDER_ID is missing.")

            print(f"[WARNING] {name} folder ID not configured.")
            continue

        folder = service.files().get(
            fileId=folder_id,
            fields="id,name,mimeType",
            supportsAllDrives=True
        ).execute()

        if folder.get("mimeType") != "application/vnd.google-apps.folder":
            raise RuntimeError(f"{name} ID is not a Drive folder.")

        print(f"[SUCCESS] {name}: {folder['name']}")

    # Test actual upload permission in clips folder.
    clips_folder_id = FOLDERS["Clips"]

    test_filename = f"_oauth_test_{uuid.uuid4().hex[:8]}.txt"
    test_content = b"Google Drive OAuth upload test successful."

    uploaded_id = None

    print("\n[UPLOAD] Testing upload permission...")

    try:
        media = MediaIoBaseUpload(
            io.BytesIO(test_content),
            mimetype="text/plain",
            resumable=False
        )

        uploaded = service.files().create(
            body={
                "name": test_filename,
                "parents": [clips_folder_id]
            },
            media_body=media,
            fields="id,name,size"
        ).execute()

        uploaded_id = uploaded["id"]

        print("[SUCCESS] Test file uploaded:", uploaded["name"])

        # Verify uploaded file.
        verified = service.files().get(
            fileId=uploaded_id,
            fields="id,name,size,parents"
        ).execute()

        if int(verified.get("size", 0)) != len(test_content):
            raise RuntimeError("Uploaded file size verification failed.")

        if clips_folder_id not in verified.get("parents", []):
            raise RuntimeError("Uploaded file is in the wrong folder.")

        print("[SUCCESS] Upload verified.")
        print("[SUCCESS] File size:", verified["size"], "bytes")

    finally:
        # Clean up only the temporary test file created above.
        if uploaded_id:
            service.files().delete(
                fileId=uploaded_id
            ).execute()

            print("[CLEANUP] Temporary test file deleted.")

    print("\n" + "=" * 60)
    print("ALL GOOGLE DRIVE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_drive()
    except Exception as error:
        print("\n[FAILED]", error)
        raise

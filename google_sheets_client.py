from datetime import datetime

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build, Resource
from googleapiclient.errors import HttpError


class GoogleSheetsClientException(Exception):
    pass


class GoogleSheetAppendIncomeFailed(GoogleSheetsClientException):
    pass


class GoogleSheetsClient:

    scopes = ['https://www.googleapis.com/auth/spreadsheets']

    def __init__(
        self, token_file_path: str, credentials_file_path: str, spreadsheet_id: str
    ) -> None:
        self.creds = Credentials.from_authorized_user_file(token_file_path, self.scopes)
        self.spreadsheet_id = spreadsheet_id

        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    credentials_file_path, self.scopes
                )
                self.creds = flow.run_local_server(port=0)

            with open(token_file_path, 'w') as token:
                token.write(self.creds.to_json())

    def append_income_row(self, date_time: datetime, symbol: str, value: str):
        try:
            service: Resource = build('sheets', 'v4', credentials=self.creds)

            body = {'values': [[
                date_time.strftime('%d.%m.%Y %H:%M:%S'), symbol, value
            ]]}
            service.spreadsheets().values().append(
                spreadsheetId=self.spreadsheet_id,
                range='Results!A:C',
                valueInputOption='RAW',
                body=body,
            ).execute()
        except HttpError as error:
            raise GoogleSheetAppendIncomeFailed(f'Google sheet append income failed {error}')

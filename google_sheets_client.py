from datetime import datetime

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build, Resource
from googleapiclient.errors import HttpError


class GoogleSheetsClientException(Exception):
    pass


class GoogleSheetAppendIncomeFailed(GoogleSheetsClientException):
    pass


class GoogleSheetsClient:

    scopes = ['https://www.googleapis.com/auth/spreadsheets']

    def __init__(
        self, credentials_file_path: str, spreadsheet_id: str
    ) -> None:
        self.creds = Credentials.from_service_account_file(credentials_file_path, scopes=self.scopes)
        self.spreadsheet_id = spreadsheet_id

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

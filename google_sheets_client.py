from datetime import datetime
from decimal import Decimal

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build, Resource
from googleapiclient.errors import HttpError


class GoogleSheetsClientException(Exception):
    pass


class GoogleSheetAppendIncomeFailed(GoogleSheetsClientException):
    pass


class GoogleSheetAppendUnsoldCurrencyFailed(GoogleSheetsClientException):
    pass


class GoogleSheetAppendIncomesReportFailed(GoogleSheetsClientException):
    pass


class GoogleSheetsClient:

    scopes = ['https://www.googleapis.com/auth/spreadsheets']

    def __init__(
        self, credentials_file_path: str, spreadsheet_id: str
    ) -> None:
        self.creds = Credentials.from_service_account_file(credentials_file_path, scopes=self.scopes)
        self.spreadsheet_id = spreadsheet_id

        self.service: Resource = build('sheets', 'v4', credentials=self.creds)

    def append_income(self, date_time: datetime, symbol: str, value: str):
        try:
            body = {'values': [[
                date_time.strftime('%d.%m.%Y %H:%M:%S'), symbol, value
            ]]}
            self.service.spreadsheets().values().append(
                spreadsheetId=self.spreadsheet_id,
                range='Incomes!A:C',
                valueInputOption='RAW',
                body=body,
            ).execute()
        except HttpError as error:
            raise GoogleSheetAppendIncomeFailed(f'Google sheet append income failed {error}')

    def append_unsold_currency(self, date_time: datetime, symbol: str, price: Decimal):
        try:
            body = {'values': [[
                date_time.strftime('%d.%m.%Y %H:%M:%S'), symbol, str(price)
            ]]}
            self.service.spreadsheets().values().append(
                spreadsheetId=self.spreadsheet_id,
                range='Unsold!A:C',
                valueInputOption='RAW',
                body=body,
            ).execute()
        except HttpError as error:
            raise GoogleSheetAppendUnsoldCurrencyFailed(
                f'Google sheet append unsold currency failed {error}'
            )

    def append_incomes_report(
        self, start: datetime, end: datetime, currencies: str, amount: str
    ):
        try:
            body = {'values': [[
                start.strftime('%d.%m.%Y %H:%M:%S'), end.strftime('%d.%m.%Y %H:%M:%S'),
                currencies, amount
            ]]}
            self.service.spreadsheets().values().append(
                spreadsheetId=self.spreadsheet_id,
                range='Report incomes!A:D',
                valueInputOption='RAW',
                body=body,
            ).execute()
        except HttpError as error:
            raise GoogleSheetAppendIncomesReportFailed(
                f'Google sheet append incomes report failed {error}'
            )

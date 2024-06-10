from datetime import datetime
from decimal import Decimal
from typing import List

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


class GoogleSheetDeleteUnsoldCurrenciesFailed(GoogleSheetsClientException):
    pass


class GoogleSheetGetUnsoldCurrenciesFailed(GoogleSheetsClientException):
    pass


class GoogleSheetAppendToTestConnectionFailed(GoogleSheetsClientException):
    pass


class GoogleSheetsClient:

    scopes = ['https://www.googleapis.com/auth/spreadsheets']

    def __init__(
        self, credentials_file_path: str, spreadsheet_id: str
    ) -> None:
        self.creds = Credentials.from_service_account_file(credentials_file_path, scopes=self.scopes)
        self.spreadsheet_id = spreadsheet_id

        self.service: Resource = build('sheets', 'v4', credentials=self.creds)

    def append_income(
        self, first_date_time: datetime, last_date_time: datetime, symbol: str,
        difference: float, difference_in_percentage: float, income: float
    ):
        try:
            body = {'values': [[
                first_date_time.strftime('%d.%m.%Y %H:%M:%S'), last_date_time.strftime('%d.%m.%Y %H:%M:%S'),
                symbol, difference, difference_in_percentage, income
            ]]}
            self.service.spreadsheets().values().append(
                spreadsheetId=self.spreadsheet_id,
                range='Incomes!A:F',
                valueInputOption='RAW',
                body=body,
            ).execute()
        except HttpError as error:
            raise GoogleSheetAppendIncomeFailed(f'Google sheet append income failed {error}')

    def append_unsold_currency(self, date_time: datetime, cmc_id: int, symbol: str, price: Decimal):
        try:
            body = {'values': [[
                date_time.strftime('%d.%m.%Y %H:%M:%S'), symbol, str(cmc_id), str(price)
            ]]}
            self.service.spreadsheets().values().append(
                spreadsheetId=self.spreadsheet_id,
                range='Unsold!A:D',
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

    def get_unsold_currencies(self) -> List[List[str]]:
        try:
            result = (
                self.service.spreadsheets().values().get(
                    spreadsheetId=self.spreadsheet_id,
                    range='Unsold!A:D'
                ).execute()
            )
            return result.get("values", [])
        except HttpError as error:
            raise GoogleSheetGetUnsoldCurrenciesFailed(
                f'Google sheet get unsold currencies failed {error}'
            )

    def delete_unsold_currencies(self):
        try:
            body = {
                'ranges': ['Unsold']
            }

            self.service.spreadsheets().values().batchClear(
                spreadsheetId=self.spreadsheet_id,
                body=body
            ).execute()
        except HttpError as error:
            raise GoogleSheetDeleteUnsoldCurrenciesFailed(
                f'Google sheet delete unsold currencies failed {error}'
            )

    def append_to_test_connection(self, date_time: datetime):
        try:
            body = {'values': [[
                date_time.strftime('%d.%m.%Y %H:%M:%S')
            ]]}
            self.service.spreadsheets().values().append(
                spreadsheetId=self.spreadsheet_id,
                range='Test connection!A:A',
                valueInputOption='RAW',
                body=body,
            ).execute()
        except HttpError as error:
            raise GoogleSheetAppendToTestConnectionFailed(
                f'Google sheet append to test connection failed {error}'
            )

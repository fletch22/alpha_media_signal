from alpaca.trading.client import TradingClient

from config.constants import ALPACA_CREDENTIALS


def get_trading_client():
    api_key = ALPACA_CREDENTIALS['API-KEY-ID']
    secret_key = ALPACA_CREDENTIALS['API-SECRET-KEY']
    url_override = ALPACA_CREDENTIALS['API_BASE_URL']

    return TradingClient(api_key, secret_key, url_override=url_override)


if __name__ == '__main__':
    trading_client = get_trading_client()
    # Get our account information.
    account = trading_client.get_account()

    # Check our current balance vs. our balance at the last market close
    balance_change = float(account.equity) - float(account.last_equity)
    print(f'Today\'s portfolino balance change: ${balance_change}')

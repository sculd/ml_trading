import os, requests, logging

_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
_chat_id = os.getenv("TELEGRAM_CHAT_ID")

def post_message(message):
    url = f"https://api.telegram.org/bot{_bot_token}/sendMessage?chat_id={_chat_id}&text={message}"
    logging.info(f'{requests.get(url).json()}')

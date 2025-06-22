import telebot
from backbone.utils.general_purpose import map_order_to_str


class TelegramBot():
    def __init__(self, bot_token, chat_id) -> None:
        self.bot = telebot.TeleBot(bot_token)
        self.chat_id = chat_id

    def send_order_by_telegram(self, order):
        order = order
        message = map_order_to_str(order)
        self.bot.send_message(chat_id=self.chat_id, text=message)
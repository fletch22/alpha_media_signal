import slack

from ams.config import constants

# NOTE: e: chris.flesche: The Free Village Chris
USER_ID_CHRIS = "U4J5M51CK"

credentials = constants.SLACK_CREDENTIALS
client = slack.WebClient(token=credentials['bot_token_id'])


def send_direct_message_to_chris(message):
    user_channel = client.conversations_open(users=[USER_ID_CHRIS])
    return client.chat_postMessage(
        text=message,
        channel=user_channel['channel']['id']
    )
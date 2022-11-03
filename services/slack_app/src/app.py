import os
import re
import requests
import json
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from sqlite_utils import setup_db, update_db


SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", default="xoxb-")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN", default="xapp-")
HATE_THRESHOLD = float(os.environ.get("HATE_THRESHOLD", default=0.5))


setup_db()
app = App(token=SLACK_BOT_TOKEN)


def check_if_hate_speech(m: dict, test: bool = False) -> bool:
    """
    Checks a message, m, for hate speech.

    :param m: Message dictionary from Slack
    :param test: If True, use the testing condition instead of the model.  (checks if the word 'hate' is in the message)
    """
    if test:
        return True if "hate" in m['text'] else False
    else:
        data = {
            "id": m['client_msg_id'],
            "comment_text": m['text']
        }
        data = {"data": [data]}
        # get predictions from the model API using the URL from the bridge network
        # see: https://docs.docker.com/network/network-tutorial-standalone/#use-the-default-bridge-network
        resp = requests.post(
            'http://172.17.0.1:8000/',
            data=json.dumps(data)
        )
        preds = json.loads(resp.text)['preds']
        # This function must return a boolean, but the model is set up to make batch predictions, therefore
        # only the first prediction will be considered.  The model should be called after every individual message
        # anyway, so it should only ever have 1 prediction.
        return preds[0]['pred'] >= HATE_THRESHOLD


# Listens to incoming messages that contain a certain string
@app.message(re.compile(".*"))
def message_hello(message, say):
    # check if the message is hate speech
    if check_if_hate_speech(m=message):

        # get the users involved in the interaction
        users_involved = app.client.conversations_members(token=SLACK_BOT_TOKEN, channel=message['channel'])

        # save new network edges where the source = user sending the message and targets = other users in conversation
        edges = [
            {"source": message['user'], "target": u, "timestamp": message['ts']}
            for u in users_involved['members'] if u != message['user']
        ]

        # record a new hate speech incident
        incident = {
            "id": message['client_msg_id'],
            "user": message['user'],
            "text": message['text'],
            "timestamp": message['ts'],
            "channel": message['channel'],
        }
        update_db(hate_speech=tuple(incident.values()))
        for e in edges:
            update_db(edges=tuple(e.values()))

        # say() sends a message to the channel where the event was triggered
        say(
            blocks=[
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f"<@{message['user']}> Hello friend, are you using hate speech?  "
                            f"Your activity has been logged. \n\n"
                            f"*Re-consider your message and delete it if necessary.*"
                        ),
                    },
                },
                {"type": "divider"},
            ],
            text=(
                f"<@{message['user']}> Hello friend, are you using hate speech?  "
                f"Your activity has been logged. \n\n"
                f"*Re-consider your message and delete it if necessary.*"
            )
        )


if __name__ == "__main__":
    # socket mode lets apps use the events API without exposing a public HTTP endpoint
    SocketModeHandler(app, SLACK_APP_TOKEN).start()

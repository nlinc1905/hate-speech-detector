import os
import re
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from sqlite_utils import setup_db, update_db


SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", default="xoxb-4275464327697-4256230457206-WimlAdlGd8HniWFL2z2wGTIy")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN", default="xapp-1-A047MRUCL05-4286592031760-9820efd2eca90e5a7c8147d92086ba01a73255807fcc854a14777a781ac0c485")


setup_db()
app = App(token=SLACK_BOT_TOKEN)


# Listens to incoming messages that contain a certain string
@app.message(re.compile(".*"))
def message_hello(message, say):
    # check if the message is hate speech
    if "hate" in message['text']:

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

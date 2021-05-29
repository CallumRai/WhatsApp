import os
from datetime import datetime


def corpus(contact, file_name="_chat"):
    """
    Saves messages from a contact as a txt in corpus folder

    Args:
        contact: str
            Name of contact to make a corpus from
        file_name: str (Optional)
            Name of file where WhatsApp text is saved, defaults to _chat
    """
    # If file exists do nothing
    corpus_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + \
                  f"/data/corpus/{file_name}_{contact}.txt"
    if os.path.isfile(corpus_path):
        return

    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + f"/data/whatsapp/{file_name}.txt"
    f = open(path, "r", encoding="utf-8")
    lines = f.read().split("\n")
    f.close()

    # Create a list of messages
    messages = []
    for line in lines:
        # If message is media omit
        line_split = line.split(":")
        if line_split[-1] == " <Media omitted>":
            continue

        # If line does not start with a datetime append to previous message
        datetime_str = line[:17]
        try:
            datetime.strptime(datetime_str, "%d/%m/%Y, %H:%M")
        except ValueError:
            messages[-1] += " " + line
            continue

        messages.append(line)

    # Extract contact messages only
    contact_messages = []
    for msg in messages:
        msg_split = msg.split(":", maxsplit=3)

        try:
            msg_contact = msg_split[1].split("-", maxsplit=2)[1][1:]
            msg_txt = msg_split[2][1:]
        except IndexError:
            # If txt not found is instead a notif e.g. icon changed so skip
            continue

        if msg_contact == contact:
            contact_messages.append(msg_txt)

    # Save as corpus
    f = open(corpus_path, "w", encoding="utf-8")
    f.write(str(contact_messages))
    f.close()
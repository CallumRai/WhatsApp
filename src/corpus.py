import os


class Corpus:
    """
    Extracts and creates a corpus from WhatsApp messages for a particular contact
    """

    def __init__(self, contact, file_name="_chat"):
        """
        Initialise corpus class

        Args:
            contact: str (Optional)
                Name of contact to make a corpus from, defaults to _chat
            file_name: str
                Name of file where WhatsApp text is saved
        """

        self.file_name = file_name
        self.contact = contact

    def extract(self):
        # Get text
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + f"/data/whatsapp/{self.file_name}.html"

        f = open(path, "r", encoding="utf-8")
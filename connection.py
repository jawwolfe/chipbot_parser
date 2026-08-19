class SQLServerConnection:
    def __init__(self, name, database, username, key, token):
        self.name = name
        self.database = database
        self.username = username
        self.key = key
        self.token = token

class DatabaseConnectionException(Exception):
    """An error occurred connecting to SQL Server via odbc"""
    pass


class DatabaseOperationException(Exception):
    """An error occurred executing a sql server sql command."""
    pass


class NoErrorsException(Exception):
    """An error occurred locating error file for processing."""
    pass


class TaxonomyException(Exception):
    """Bird doesn't match taxonomy need to fix before proceeding."""
    pass


class VerifyFileException(Exception):
    """Image name has incorrect name or artist"""
    pass
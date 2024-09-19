def get_exception_msg(e: Exception) -> str:
    """
    Retrieves the message from an exception.
    Args:
        e (Exception): The exception from which to extract the message.
    Returns:
        str: The message of the exception. If the exception has a 'message' attribute,
             it returns that. Otherwise, it returns the string representation of the exception.
    """

    if hasattr(e, "message"):
        return str(e.message)
    else:
        return str(e)

def get_exception_msg(e: Exception) -> str:
    if hasattr(e, "message"):
        return e.message
    else:
        return str(e)
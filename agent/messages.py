"""Attach the latest RGB to its observation without changing message history."""


def attach_image(messages, image_message_index=None, extra_image_names=()):
    index = len(messages) - 1 if image_message_index is None else image_message_index
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < len(messages)
    ):
        raise ValueError("image_message_index must identify an existing message")
    if messages[index].get("role") != "user":
        raise ValueError("The image must belong to a user observation message")
    result = [dict(message) for message in messages]
    prefix = (
        "[IMG]\n"
        if not extra_image_names
        else "rgb: [IMG]\n" + "".join(f"{name}: [IMG]\n" for name in extra_image_names)
    )
    result[index]["content"] = prefix + result[index]["content"]
    return result

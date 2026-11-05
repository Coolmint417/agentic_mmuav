"""Bind current and historical images to their own observation messages."""


def image_groups(messages, image=None, index=None, extra_images=None, message_images=None):
    """Return validated image groups in message order for both model backends."""
    groups = dict(message_images or {})
    if image is not None:
        index = len(messages) - 1 if index is None else index
        if index in groups:
            raise ValueError("Duplicate image group for the current observation")
        groups[index] = {"rgb": image, **(extra_images or {})}
    for position, attachments in groups.items():
        if (isinstance(position, bool) or not isinstance(position, int)
                or not 0 <= position < len(messages)
                or messages[position].get("role") != "user"):
            raise ValueError("Images must identify an existing user observation message")
        if not attachments or next(iter(attachments)) != "rgb":
            raise ValueError("Observation image groups must start with rgb")
    return dict(sorted(groups.items()))


def prepare_hf_images(messages, image=None, index=None, extra_images=None, message_images=None):
    """Keep HF placeholder order identical to the flattened processor images."""
    result = [dict(message) for message in messages]
    images = []
    for position, attachments in image_groups(
            messages, image, index, extra_images, message_images).items():
        result = attach_image(result, position, tuple(attachments)[1:])
        images.extend(attachments.values())
    return result, images


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

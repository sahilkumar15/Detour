"""Canonicalize tool arguments BEFORE the gate evaluates them.

Why this module exists, and why it is load-bearing rather than cosmetic:

The effect oracle reads named arguments (`recipients`, `file_id`, `email`).
Models do not reliably emit the schema's names: `send_email(to="x")` is a
common variant. If the gate reads only the canonical `recipients` field, an
aliased call yields NO effect and is ALLOWED, while an executor with a
different notion of the field may treat the same call differently. In this
sandbox the un-normalized call reached no recipient, so no forbidden
disclosure was observed; the hazard is the DESIGN one: gate and executor must
parse a call identically, or a divergence can become a bypass. That is a
property to enforce and test, not an observed violation, and it is not a
result of the audit.

So argument shape is normalized ONCE, at the boundary, and the same
normalized dict is what the oracle, the gate and the sandbox all see.

The second rule here is that a call we cannot canonicalize RAISES.
Substituting an empty default is what made the bypass silent. A malformed
call is an `unrelated_error` outcome -- neither a violation nor a recovery
-- and it is counted and reported, never absorbed.

Aliases are intentionally conservative: only names that are unambiguous for
the tool in question. `content` means the file body for `create_file` and
the message body for a messaging tool, so it is mapped per tool, never
globally.
"""

from __future__ import annotations

from typing import Any, Mapping

__all__ = ["MalformedCallError", "normalize_args", "LIST_FIELDS", "REQUIRED"]


class MalformedCallError(ValueError):
    """A tool call whose arguments cannot be canonicalized.

    Raised rather than defaulted: a silently emptied field produces an
    allowed decision for a forbidden action.
    """


# Per-tool alias map: alias -> canonical name.
ALIASES: dict[str, dict[str, str]] = {
    "send_email": {
        "to": "recipients",
        "recipient": "recipients",
        "emails": "recipients",
        "email": "recipients",
        "to_addresses": "recipients",
        "message": "body",
        "text": "body",
        "content": "body",
        "title": "subject",
        "attachment": "attachments",
        "files": "attachments",
    },
    "send_direct_message": {
        "to": "recipient",
        "user": "recipient",
        "username": "recipient",
        "recipient_id": "recipient",
        "recipients": "recipient",
        "message": "body",
        "text": "body",
        "content": "body",
    },
    "send_channel_message": {
        "channel_name": "channel",
        "channel_id": "channel",
        "to": "channel",
        "message": "body",
        "text": "body",
        "content": "body",
    },
    "share_file": {
        "id": "file_id",
        "file": "file_id",
        "fileId": "file_id",
        "filename": "file_id",
        "document_id": "file_id",
        "recipient": "email",
        "to": "email",
        "user_email": "email",
        "shared_with": "email",
        "role": "permission",
        "access": "permission",
    },
    "delete_file": {
        "id": "file_id",
        "file": "file_id",
        "fileId": "file_id",
        "filename": "file_id",
        "document_id": "file_id",
    },
    "get_file_by_id": {
        "id": "file_id",
        "file": "file_id",
        "fileId": "file_id",
        "filename": "file_id",
    },
    "append_to_file": {
        "id": "file_id",
        "file": "file_id",
        "fileId": "file_id",
        "filename": "file_id",
        "text": "content",
        "body": "content",
    },
    "create_file": {
        "name": "filename",
        "file_name": "filename",
        "path": "filename",
        "title": "filename",
        "text": "content",
        "body": "content",
    },
    "invite_user_to_slack": {
        "username": "user",
        "email": "user_email",
    },
    "read_inbox": {"username": "user", "recipient": "user"},
    "request_approval": {"rationale": "reason", "justification": "reason"},
    "report_and_stop": {"state": "status", "result": "status", "text": "summary"},
}

# Fields that must be lists; a scalar is wrapped rather than rejected,
# because the intent is unambiguous.
LIST_FIELDS: dict[str, tuple[str, ...]] = {
    "send_email": ("recipients", "cc", "bcc", "attachments"),
}

# Fields that must be present and non-empty for the call to mean anything.
REQUIRED: dict[str, tuple[str, ...]] = {
    "send_email": ("recipients",),
    "send_direct_message": ("recipient", "body"),
    "send_channel_message": ("channel", "body"),
    "share_file": ("file_id", "email"),
    "delete_file": ("file_id",),
    "get_file_by_id": ("file_id",),
    "append_to_file": ("file_id", "content"),
    "create_file": ("filename",),
    "invite_user_to_slack": ("user", "user_email"),
    "read_inbox": ("user",),
}


def normalize_args(tool: str, args: Mapping[str, Any]) -> dict[str, Any]:
    """Return `args` with canonical names, list-shaped lists, and no gaps.

    Raises MalformedCallError when a required field is absent or empty.
    """
    out: dict[str, Any] = {}
    aliases = ALIASES.get(tool, {})

    for key, value in args.items():
        canonical = aliases.get(key, key)
        # A real value never loses to an alias that arrived later.
        if canonical in out and _empty(value):
            continue
        if canonical in out and not _empty(out[canonical]):
            continue
        out[canonical] = value

    for field in LIST_FIELDS.get(tool, ()):
        value = out.get(field)
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            out[field] = [v for v in value if not _empty(v)]
        elif isinstance(value, str):
            # "a@x.com, b@y.com" is a single string in practice.
            parts = [p.strip() for p in value.split(",")]
            out[field] = [p for p in parts if p]
        else:
            out[field] = [value]

    missing = [f for f in REQUIRED.get(tool, ()) if _empty(out.get(f))]
    if missing:
        raise MalformedCallError(
            f"{tool} is missing required argument(s) {', '.join(missing)}; "
            f"got keys {sorted(args)}"
        )
    return out


def _empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False
"""Multi-provider token counting using tiktoken."""
from __future__ import annotations

import logging
from typing import Union

import tiktoken

from .pricing import get_pricing

logger = logging.getLogger(__name__)

# Type aliases
Message = dict[str, str]
Messages = list[Message]

# Chat message overhead constants (per OpenAI cookbook)
TOKENS_PER_MESSAGE = 3
TOKENS_PER_NAME = 1
REPLY_OVERHEAD = 3

FALLBACK_ENCODING = "cl100k_base"

# Cache encoding objects (expensive to initialize)
_encoding_cache: dict[str, tiktoken.Encoding] = {}


def _get_encoding(encoding_name: str) -> tiktoken.Encoding:
    """Get a tiktoken encoding with caching. Falls back to cl100k_base."""
    if encoding_name not in _encoding_cache:
        try:
            _encoding_cache[encoding_name] = tiktoken.get_encoding(encoding_name)
        except ValueError:
            logger.debug(
                "Encoding '%s' not available, falling back to %s",
                encoding_name,
                FALLBACK_ENCODING,
            )
            _encoding_cache[encoding_name] = tiktoken.get_encoding(
                FALLBACK_ENCODING
            )
    return _encoding_cache[encoding_name]


def _get_encoding_for_model(model: str) -> tiktoken.Encoding:
    """Resolve the correct tiktoken encoding for a model name."""
    pricing = get_pricing(model)
    return _get_encoding(pricing.tokenizer)


def count_tokens(
    text_or_messages: Union[str, Messages],
    model: str = "gpt-4o",
) -> int:
    """Count tokens for text or a list of chat messages.

    Args:
        text_or_messages: Either a plain string or a list of message dicts
            with 'role' and 'content' keys (OpenAI chat format).
        model: The model name used to select the correct tokenizer.

    Returns:
        Total token count.
    """
    encoding = _get_encoding_for_model(model)

    if isinstance(text_or_messages, str):
        return _count_text_tokens(text_or_messages, encoding)
    elif isinstance(text_or_messages, list):
        return _count_message_tokens(text_or_messages, encoding)
    else:
        raise TypeError(
            f"Expected str or list of messages, got {type(text_or_messages)}"
        )


def _count_text_tokens(text: str, encoding: tiktoken.Encoding) -> int:
    """Count tokens in a plain string."""
    return len(encoding.encode(text))


def _count_message_tokens(messages: Messages, encoding: tiktoken.Encoding) -> int:
    """Count tokens in a list of chat messages with overhead.

    Per OpenAI cookbook:
    - 3 tokens per message (role + delimiters)
    - 1 extra token if 'name' field is present
    - 3 tokens for reply priming at the end
    """
    num_tokens = 0
    for message in messages:
        num_tokens += TOKENS_PER_MESSAGE
        for key, value in message.items():
            num_tokens += len(encoding.encode(str(value)))
            if key == "name":
                num_tokens += TOKENS_PER_NAME
    num_tokens += REPLY_OVERHEAD
    return num_tokens

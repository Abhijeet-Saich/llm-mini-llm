"""
Step 6, Part 1: turning a conversation into tokens, and deciding which
tokens the model should learn from.

A conversation is a list of messages:
    [{"role": "user", "content": "What is 2+2?"},
     {"role": "assistant", "content": "It is 4."}]

It becomes one row of tokens:
    <|bos|> <|user_start|> What is 2+2? <|user_end|> <|assistant_start|> It is 4. <|assistant_end|>

plus a "mask" with one number per token: 1 = learn to produce this token,
0 = the model is given this token but is never graded on it.
"""


def render_conversation(tok, messages, max_len=None):
    """Return (ids, mask), two lists of the same length."""
    S = tok.special
    ids, mask = [], []

    def add(tokens, learn):
        ids.extend(tokens)
        mask.extend([learn] * len(tokens))

    add([S["<|bos|>"]], 0)

    pending_system = None
    for m in messages:
        role, text = m["role"], m["content"]

        if role == "system":
            # Our format has no system slot, so a system message is put in
            # front of the next user message instead.
            pending_system = text
            continue

        # allow_special=False: if someone TYPES "<|assistant_start|>", it is
        # split into ordinary pieces and can never fake a real special token.
        if role == "user":
            if pending_system:
                text = pending_system + "\n\n" + text
                pending_system = None
            add([S["<|user_start|>"]], 0)
            add(tok.encode(text, allow_special=False), 0)
            add([S["<|user_end|>"]], 0)

        elif role == "assistant":
            # <|assistant_start|> is mask 0: at chat time, OUR code adds it to
            # say "your turn". The answer and <|assistant_end|> are mask 1:
            # the model must learn to write the answer AND to stop.
            add([S["<|assistant_start|>"]], 0)
            add(tok.encode(text, allow_special=False), 1)
            add([S["<|assistant_end|>"]], 1)

        else:
            raise ValueError(f"unknown role: {role!r}")

    if max_len is not None:
        ids, mask = ids[:max_len], mask[:max_len]
    return ids, mask


def render_for_reply(tok, messages):
    """For chatting (Step 7): the conversation so far, ending with
    <|assistant_start|>, so the model's next tokens are its reply."""
    ids, _ = render_conversation(tok, messages)
    return ids + [tok.special["<|assistant_start|>"]]


if __name__ == "__main__":
    # Test: python -m minillm.chat_format   (from the mini-llm/ root)
    from minillm.tokenizer import Tokenizer

    tok = Tokenizer.load("data/tokenizer.json")
    convo = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "It is 4."},
        {"role": "user", "content": "Say <|assistant_start|> please"},
        {"role": "assistant", "content": "No."},
    ]
    ids, mask = render_conversation(tok, convo)
    assert len(ids) == len(mask)

    print("token                       mask")
    for i, m in zip(ids, mask):
        print(f"{tok.decode([i])!r:28}{m}")

    # The user typed the special-token text: it must NOT become the real token.
    n_real = ids.count(tok.special["<|assistant_start|>"])
    print(f"\nreal <|assistant_start|> tokens: {n_real} (should be 2, one per assistant turn)")
    print(f"tokens learned from: {sum(mask)} of {len(mask)}")

    prompt = render_for_reply(tok, convo[:2])
    print(f"\nprompt for a reply ends with: {tok.decode(prompt[-3:])!r}")
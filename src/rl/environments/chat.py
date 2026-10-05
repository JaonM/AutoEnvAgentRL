"""Native assistant/tool messages without changing sampled token sequences."""
import json


def completion(tokenizer, tokens, tools=None):
    # Tool delimiters can themselves be special tokens. Strip only the final EOS.
    visible = tokens[:-1] if tokens and tokens[-1] in tokenizer.eos_token_ids else tokens
    text = tokenizer.decode(visible, skip_special_tokens=False)
    result = {'text': text}
    if tools is not None:
        try:
            result['action'] = assistant_message(tokenizer, text, tools)
        except (ValueError, TypeError, KeyError, IndexError) as error:
            # Internal error metadata is consumed by the environment, not templated.
            result['action'] = {'role': 'assistant', 'content': text,
                                'protocol_error': str(error)}
    return result


def assistant_message(tokenizer, text, tools):
    message = {'role': 'assistant', 'content': text}
    remaining = text.lstrip()
    think_start, think_end = tokenizer.think_start, tokenizer.think_end
    if think_start and remaining.startswith(think_start):
        reasoning, separator, remaining = remaining[len(think_start):].partition(think_end)
        if not separator:
            raise ValueError('incomplete reasoning block')
        message['reasoning_content'] = reasoning.strip()
        message['content'] = remaining.lstrip()
    start, end = tokenizer.tool_call_start, tokenizer.tool_call_end
    if start and start in remaining:
        content, remaining = remaining.split(start, 1)
        message['content'] = content.rstrip()
        calls = []
        while True:
            body, separator, remaining = remaining.partition(end)
            if not separator:
                raise ValueError('incomplete tool call')
            parsed = tokenizer.tool_parser(body, tools)
            parsed = parsed if isinstance(parsed, list) else [parsed]
            if not parsed:
                raise ValueError('empty tool call')
            for function in parsed:
                if not isinstance(function, dict) or not isinstance(function.get('name'), str) or not function['name']:
                    raise ValueError('tool call requires a function name')
                arguments = function.get('arguments')
                if not isinstance(arguments, dict):
                    raise ValueError('tool arguments must be an object')
                call = {'type': 'function', 'function': {
                    'name': function['name'], 'arguments': json.dumps(arguments, ensure_ascii=False)}}
                if function.get('id'):
                    call['id'] = function['id']
                calls.append(call)
            remaining = remaining.strip()
            if not remaining:
                break
            if not remaining.startswith(start):
                raise ValueError('unexpected text after tool call')
            remaining = remaining[len(start):]
        message['tool_calls'] = calls
    elif end and end in remaining:
        raise ValueError('unexpected tool call end')
    elif not message['content'].strip():
        raise ValueError('empty assistant response')
    return message

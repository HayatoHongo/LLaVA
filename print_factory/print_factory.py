import builtins
import io
import contextlib

def normalize_key(key: str) -> str:
    """printキーを実行時と同じ形に揃える"""
    return key.encode("utf-8").decode("unicode_escape").strip()

def traced_print_factory(original_print):
    store = {}
    def traced_print(*args, **kwargs):
        if args and isinstance(args[0], str) and len(args) > 1:
            key = normalize_key(args[0])
            # stdout と同じ形式で value を文字列化
            buf = io.StringIO()
            print(*args[1:], **kwargs, file=buf)
            value = buf.getvalue().rstrip()
            store[key] = value
        # 通常の print も実行
        original_print(*args, **kwargs)
    return traced_print, store

def run_and_capture(func, *args, **kwargs):
    original_print = builtins.print
    traced_print, store = traced_print_factory(original_print)

    builtins.print = traced_print
    try:
        func(*args, **kwargs)
    finally:
        builtins.print = original_print

    return store


def embed_print_outputs(code: str, mapping: dict[str, str]) -> str:
    """元のコードに print 出力を埋め込む"""
    new_lines = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("print(") and stripped[6:].startswith('"'):
            if "," not in stripped:
                new_lines.append(line)
                continue

            try:
                raw_key = stripped.split('"', 2)[1]
                key = normalize_key(raw_key)
            except IndexError:
                key = None

            indent = line[:len(line) - len(line.lstrip())]  # インデント保持

            if key and key in mapping:
                value = mapping[key]
                if len(value) <= 40:
                    new_lines.append(f"{line}  # {value}")
                else:
                    new_lines.append(line)
                    new_lines.append(f'{indent}"""')
                    for vline in value.splitlines():
                        new_lines.append(f"{indent}{vline}")
                    new_lines.append(f'{indent}"""')
            else:
                new_lines.append(f"{line}  # not found")
        else:
            new_lines.append(line)
    return "\n".join(new_lines)
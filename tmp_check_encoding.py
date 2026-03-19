import pathlib

b = pathlib.Path('config.yaml').read_bytes()
print('len', len(b))
print('first', b[:20])
for enc in ['utf-8', 'cp1252', 'latin-1']:
    try:
        b.decode(enc)
        print(enc, 'ok')
    except Exception as e:
        print(enc, 'err', type(e).__name__, e)

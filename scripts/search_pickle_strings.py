import re
p='models/rf_model.pkl'
with open(p,'rb') as f:
    data=f.read()
words=set(re.findall(b"[A-Za-z0-9_\.]{4,}", data))
for w in sorted([x.decode('latin1') for x in words if 'numpy' in x.decode('latin1')]):
    print(w)
print('found', len([x for x in words if b'numpy' in x]))

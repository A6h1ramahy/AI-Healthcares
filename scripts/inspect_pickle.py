import pickletools
import sys
p='models/rf_model.pkl'
with open(p,'rb') as f:
    data=f.read()
mods=set()
for op, arg, pos in pickletools.genops(data):
    if op.name=='GLOBAL':
        mods.add(arg)
# print only those containing 'numpy' or 'numpy.'
for m in sorted([x for x in mods if 'numpy' in x]):
    print(m)
print('\nTotal globals found:', len(mods))

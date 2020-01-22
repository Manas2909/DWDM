def load_dataset():
    return [[1, 2, 5], [2, 4], [2, 3], [1, 2, 4], [1, 3], [2, 3], [1, 3], [1, 2, 3, 5], [1, 2, 3]]


dataset = load_dataset()
minsupport = 2

print("c1")

c1 = {}
for tran in dataset:
    for item in tran:
        key = frozenset([item])
        if key not in c1.keys():
            c1[key] = 0
        c1[key] += 1
print(c1)

for key in c1.keys():
    if c1[key] < minsupport:
        c1.pop(key)
print(c1, "\n")

print("c2")

c2 = {}
c1keys = list(c1.keys())
for i in range(len(c1keys)):
    for j in range(i + 1, len(c1keys)):
        key = c1keys[i].union(c1keys[j])
        if key not in c2.keys():
            c2[key] = 0

for tran in dataset:
    for key in c2.keys():
        if key.issubset(tran):
            c2[key] += 1

print(c2)

keys = list(c2.keys())
for key in keys:
    if c2[key] < minsupport:
        c2.pop(key)
print(c2, "\n")

print("c3")

c3 = {}
for tran in dataset:
    for i in range(len(tran)):
        for j in range(i + 1, len(tran)):
            for k in range(j + 1, len(tran)):
                key = frozenset([tran[i], tran[j], tran[k]])
                if key not in c3.keys():
                    c3[key] = 0
                c3[key] += 1
print(c3)

keys = list(c3.keys())
for key in keys:
    if c3[key] < minsupport:
        c3.pop(key)
print(c3, "\n")

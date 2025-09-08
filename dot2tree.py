# dot2tree.py
import re
from collections import defaultdict

with open("callgraph.dot") as f:
    lines = f.readlines()

edges = []
for line in lines:
    m = re.match(r'\s*"([^"]+)"\s*->\s*"([^"]+)"', line)
    if m:
        src, dst = m.groups()
        edges.append((src, dst))

children = defaultdict(list)
parents = set()
for src, dst in edges:
    children[src].append(dst)
    parents.add(dst)

roots = [src for src, _ in edges if src not in parents]

def print_tree(node, indent=0, seen=set()):
    print("  " * indent + node)
    if node in seen:
        return
    seen.add(node)
    for child in children.get(node, []):
        print_tree(child, indent + 1, seen)

for r in roots:
    print_tree(r)

import sys

for line in sys.stdin:
    arr = line.strip().split(' ')
    id = arr[0]
    if 'i' in id:
        print line.strip()

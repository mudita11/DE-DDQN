import numpy as np
import re

count = 0
f = open('out', 'r')
'''for line in f:
    if "mean reward:" in line:
        count += 1
        print("Yes", line)
print("Total episodes= ", count)
f.close()'''

copy = open('reward', 'a')
count = 0
rew = "mean best found = "
for line in f:
    if rew in line:
        count += 1
        tokens = line.split(','); #print(tokens)
        a = re.findall(r"[-+]?\d*\.\d+|\d+", tokens[0]); #print(a[1])
        copy.write(a[1]+'\n')
        #print("token5 is ",tokens[5], a)

print(count)
f.close()
copy.close()


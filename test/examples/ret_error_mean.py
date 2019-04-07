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

copy = open('error_mean', 'a')
count = 0
rew = "mean best found = "
for line in f:
    if rew in line:
        count += 1
        tokens = line.split(','); #print(tokens)
        a = re.findall(r"[-+]?\d*\.\d+|\d+", tokens[0]); #print(a);#print(float(a[1])-float(a[0]))
        copy.write(str(float(a[1])-float(a[0]))+'\n')
        #print("token5 is ",tokens[5], a)

print(count)
f.close()
copy.close()


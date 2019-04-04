import numpy as np
import re

f = open('out', 'r')
'''for line in f:
    if "mean reward:" in line:
        count += 1
        print("Yes", line)
print("Total episodes= ", count)
f.close()'''

#copy = open('improved-reward', 'a')
count = 0
rew = "Reward improved "
for line in f:
    if rew in line:
        count += 1
        tokens = line.split(','); #print(tokens)
        a = re.findall(r"[-+]?\d*\.\d+|\d+", tokens[0])
        print(a[0])
        #copy.write(a[0]+'\n')
        #print("token5 is ",tokens[5], a)
print(count)
f.close()

count = 0
f = open('out', 'r')
for line in f:
    if rew in line:
        count += 1
        tokens = line.split(','); #print(tokens)
        a = re.findall(r"[-+]?\d*\.\d+|\d+", tokens[0])
        if count == 1:
            print(a[1])
        else:
            print(a[2])
#copy.write(a[0]+'\n')
#print("token5 is ",tokens[5], a)
print(count)
f.close()

#copy.close()


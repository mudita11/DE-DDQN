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

copy = open('all_best', 'a')
count = 0; x = []
rew = "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
for line in f:
    if rew in line:
        count += 1;
        tokens = line.split(','); #print(tokens)
        a = re.findall(r"[-+]?\d*\.\d+|\d+", tokens[0]);#print(a)
        #print(a, a[2]);
        #x.append(float(a[0]))
        #if count == 25:
        #print(x)
        #print(np.min(x))
        #print(count)
        #print(a[2])
        copy.write(str(a[2])+'\n')
        #x = []; count = 0
#copy.write(a[1]+'\n')
#print("token5 is ",tokens[5], a)
print(count)
f.close()
copy.close()


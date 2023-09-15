import numpy as np

ls = [[1,2,3],[4,5,6]]
ls = np.array(ls)
print(ls.shape)

ls[:,2] =  0
print(ls)
import scipy.io as spio

mat = spio.loadmat('Data/cleandata_students.mat', squeeze_me=True)
x = mat['x']
y = mat['y']
print (x.size)
print (y)

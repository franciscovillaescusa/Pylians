import numpy as np
from scipy import misc
from pylab import *

################################### INPUT #####################################
f_in  = 'slice.png'
f_out = 'slice_new.png'
###############################################################################

# read image and create structure for the new image
data = misc.imread(f_in);  data2 = np.zeros_like(data)
print 'data format =',data.shape

# go to each pixel and change colors, opacity...etc
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        color = data[i,j] #color = [red,green,blue,alpha]

        data2[i,j,0] = color[2]
        data2[i,j,1] = color[1]
        data2[i,j,2] = color[0]
        data2[i,j,3] = color[3]

# save image to file
#fig=figure()
fig=figure(figsize=(35,20))
axis('off')
imshow(data2)
savefig(f_out, bbox_inches='tight')

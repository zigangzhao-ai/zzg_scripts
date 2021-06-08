import os

path="/workspace1/zigangzhao/tiny/car"

file_list=os.listdir(path)
file_list.sort(key=None, reverse=False)

# file_list = ['2.JPEG']
n=181
for file_obj in file_list:

    if file_obj.endswith('.JPEG'):

        src=os.path.join(path,file_obj)

        newname = '{:0>6d}.JPEG'.format(n)
        dst = os.path.join(path, newname)

        os.rename(src,dst)
        print(src,'--',dst)
        n+=1
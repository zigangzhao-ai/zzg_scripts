import os
def del_files(path):
  for root, dirs, files in os.walk(path):
    for name in files:
      if name.endswith(".xml"):  
        os.remove(os.path.join(root, name))
        print ("Delete File: " + os.path.join(root, name))
        print("Suceess")
# test
if __name__ == "__main__":
 path = '/workspace1/zigangzhao/tiny'
 del_files(path)
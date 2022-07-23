
from os import listdir
from os.path import isfile, join

dataset_directory = "./dataset/Annotations"

files = [join(dataset_directory,file) for file in listdir(dataset_directory) if isfile(join(dataset_directory,file))]

i = 0
k= 0
with open("test.txt", "w") as test:
    with open("val.txt","w") as val:
        for file in files:
            i += 1
         
            if (i % 5) == 0:
                k=k+1
                if (k%2)==0:
                    name = file.split("\\")[-1]
                    name = name.split(".xml")[0]
                    test.write(name + "\n")
                    print("test:    " + name)
                else:
                    name = file.split("\\")[-1]
                    name = name.split(".xml")[0]
                    val.write(name + "\n")   
                    print("val:  "+name)
            
i = 0

with open("train.txt", "w") as train:
    
    for file in files:
        i += 1
        if (i % 5) != 0:   
            name = file.split("\\")[-1]
            name = name.split(".xml")[0]
            train.write(name + "\n")   
            print("train:    " + name)
          

test.close()
train.close()
val.close()
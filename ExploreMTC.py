import os
import fnmatch

PathAnno = "C:/Users/admin_local/Desktop/filesboot/nlb/AnnotatedMotifs/discovery"

# Plot ground truth in sparate files

# Plot ground truth in concatenate version

# Plot discovered patterns in sparate files

# Plot discovered patterns in concatenate version
path="C:/Users/admin_local/Dropbox/2017Pattern/MeredithTLF1MIREX2016/"

for root, dirs, files in os.walk(path):
    for CurrentFileName in files[0:4]:
        print(CurrentFileName)
        address= os.path.join(root, CurrentFileName)
        print(address)
        if fnmatch.fnmatch(CurrentFileName, "*.tlr"):
            with open(address, 'r') as CurrentFile:
                ToBeCut = CurrentFile.readlines()

from fractions import Fraction

# Import the cues
PathCue = "C:/Users/admin_local/Dropbox/2017Pattern/cues/Daar_ging_een_heer_1.txt"
CueFile = open(PathCue, 'r')
Cues = []
for i, CRaw in enumerate(CueFile):
    CueI = CRaw.split('\t')[0]
    Filename = CRaw.split('\t')[1]

    try:
        CueT = float(CueI)
    except:
        CueT = float(Fraction(CueI))

    Cues.append(CueT)

# Cut the file

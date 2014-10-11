import sys
import re
from wmmapping import TransDist

class InputLexicon:
    def __init__(self, name):
        self.name = name
        self.handle = open(self.name)

    def getNextLexeme(self):
        line = self.handle.readline()
        if line == "":
            return ("", [])
        else: 
            line = re.sub(" ", ",", line, count=1)
            List = re.findall("([^,]+),", line)
            word = List[0]
            primL = List[1:]
            self.handle.readline()		

            return (word, primL)

    def close(self):
        self.handle.close()

# --------------------------------------------- #
#						read					#
# --------------------------------------------- #
def readAll(lexname, Beta):
 
    # read in the whole lexicon file into "inputlex"
    inputlex = InputLexicon(lexname)		

    # create a translation table as the original lexicon
    original_lex = TransDist(Beta)				

    while 1:
       (word, primL) = inputlex.getNextLexeme()
       if word == "": break
       for f in primL:
           f = f + ":"
           (prim, prob) = re.findall("([^:]+):", f)
           original_lex.setValue(word, prim, eval(prob))
       original_lex.setUnseen(word, 0.0)

    inputlex.close()

    return original_lex

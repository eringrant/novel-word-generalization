from random import randint
import pickle

# parameters
NUM_ADJECTIVES = 268
NUM_DETS = 31
ADJS_IN_SENT = 1
DETS_IN_SENT = 1
BASIC_FREQ = 15
ROOT_FREQ = 5
SUPERO_FREQ = 10
SUBO_FREQ = 5

# Levels
ROOT = 0
SUPERORD = 1
BASIC = 2
SUBORD = 3

# set of all features
all_fts = set()

def build_lexicon():
    global all_fts
    lexicon = []
    inf = open("hierarchy-simple.txt",'r')
    for line in inf:
            i = 0
            level = 0
            while line[i] == '\t':
                i += 1           
            level = i
            word = ""
            while line[i] != '[':            
                word += line[i]
                i += 1
            word = word[:-1]

            features = []        
            i += 1
            featword = ""

            while line[i] != ']':
                if line[i] == ',':
                    features.append(featword.lower())
                    featword = ""
                elif line[i] != ' ':
                    featword += line[i]
                i +=1
            features.append(featword.lower())
            
            for f in features:
                all_fts.add(f)
            lexicon.append((word,level,features))
    inf.close()
    return lexicon

def pad_lexicon(lex):
    global all_fts
    inf = open("lexicon-SENSORY.all",'r')
    new_lines = []
    adjs = []
    dets = []
    for line in inf:
        if len(adjs) < NUM_ADJECTIVES and line.find(":ADJ") > 0:
            new_lines.append(line)
            word = line[0:line.find(':')]
            features = line[line.find(' ')+1:-2].split(',')
            for index in xrange(len(features)):
                features[index] = features[index][:features[index].find(":")]
            adjs.append((word,features))
            for f in features:
             all_fts.add(f)
            
        if len(dets) < NUM_DETS and line.find(":DET") > 0:
            new_lines.append(line)
            word = line[0:line.find(':')]
            features = line[line.find(' ')+1:-2].split(',')
            for index in xrange(len(features)):
                features[index] = features[index][:features[index].find(":")]
            dets.append((word,features))
            for f in features:
             all_fts.add(f)
            
        if(len(dets) >= NUM_DETS and len(adjs) >= NUM_ADJECTIVES):
            break
                 
    return (new_lines,adjs,dets)

def write_lexicon(lexicon, new_lines, all_fts):
    outf = open("lexicon-XT.all",'w')
    for entry in lexicon:
        outf.write(entry[0] + ":N ")
        for i in xrange(len(entry[2])):
            outf.write(entry[2][i] + ":" + str(float(1)/float(len(entry[2]))))
            outf.write(",")
        outf.write('\n\n')
    for line in new_lines:
       outf.write(line)
       outf.write("\n")
    outf.write("fep:N ")
    p = float(1)/float(len(all_fts))
    for f in all_fts:
       outf.write(f)
       outf.write(":" + str( p))
       outf.write(",")
    outf.write("\n")
    outf.close()
    return lexicon

def write_sentences(lexicon, adjs, dets):
    out = open("input-XT.dev",'w')

    for (word,level,features) in l:
        n = 0 
        if level == 0:
            n = ROOT_FREQ
        elif level == 1:
            n = SUPERO_FREQ
        elif level == 2:
            n = BASIC_FREQ
        elif level == 3:
            n = SUBO_FREQ
        for i in xrange(n):
            meaning = []
            out.write("1-----\nSENTENCE: ")    
            for j in xrange(DETS_IN_SENT):
                det = dets[randint(0,NUM_DETS-1)]
                out.write(det[0] + ":DET ")
                meaning.extend(det[1])
            for j in xrange(ADJS_IN_SENT):
                adj = adjs[randint(0,NUM_ADJECTIVES-1)]
                out.write(adj[0] + ":ADJ ")
                meaning.extend(adj[1])
            out.write(word + ":N\n")
            meaning.extend(features)
            out.write("SEM_REP: ")
            for ft in meaning:
                out.write("," + ft)
            out.write('\n')
    out.close()                            

def pickle_instances():
    individuals = dict()
    outfile = open("individuals.pkl",'w')
    for entry in l:
        if entry[1] == SUBORD:
            for i in xrange(3):
                name = entry[0] + str(i)
                individuals[name] = entry[2] + [name]
    pickle.dump(individuals,outfile)
    outfile.close()


l = build_lexicon()
print l
raw_input()
(newl,adjs,dets) = pad_lexicon(l)
write_lexicon(l,newl,all_fts)
write_sentences(l,adjs,dets)
pickle_instances()
print len(all_fts)

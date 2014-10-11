from random import randint,choice,shuffle
import pickle

# parameters
NUM_ADJECTIVES = 90
NUM_DETS = 31
ADJS_IN_SENT = 1
DETS_IN_SENT = 1
BASIC_FREQ = 15
SUPERO_FREQ = 10
SUBO_FREQ = 5

# Levels
SUPERORD = 0
BASIC = 1
SUBORD = 2

FT_VALS = 20

# set of all features
all_fts = set()

def build_lexicon():
    global all_fts
    lexicon = []
    inf = open("hierarchy-testing.txt",'r')
    for line in inf:
            i = 0
            level = 0
            while line[i] == ' ':
                i += 1           
            level = i / 4
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
    
    colors = range(FT_VALS) * 5
    poses = range(FT_VALS) * 5
    sizes = range(FT_VALS) * 5
    
    total_sents = 0

    all_lines = []
    for (word,level,features) in l:
        n = 0 
        if level == 0:
            n = SUPERO_FREQ
        elif level == 1:
            n = BASIC_FREQ
        elif level == 2:
            n = SUBO_FREQ
        for i in xrange(n):
            meaning = [] 
            line = "1-----\nSENTENCE: "
            for j in xrange(DETS_IN_SENT):
                det = dets[randint(0,NUM_DETS-1)]
                line += det[0] + ":DET "
                meaning.extend(det[1])
            for j in xrange(ADJS_IN_SENT):
                adj = adjs[randint(0,NUM_ADJECTIVES-1)]
                line += adj[0] + ":ADJ "
                meaning.extend(adj[1])
            line += word + ":N\n"
            meaning.extend(features)
            
            c = choice(colors)
            p = choice(poses)
            s = choice(sizes)
            colors.remove( c ) 
            poses.remove( p )
            sizes.remove( s ) 
            
            meaning.extend(["color" + str( c ), \
                            "pose" + str( p ), \
                            "size" + str( s )])
            line += "SEM_REP: "
            for ft in meaning:
                line += "," + ft
            line += '\n'
            total_sents += 1
            all_lines.append(line)
            
    shuffle(all_lines)
    for line in all_lines:
        out.write(line)
    out.close()                            

def pickle_instances():
    individuals = dict()
    outfile = open("individuals.pkl",'w')
    
    for entry in l:
        if entry[1] == SUBORD:
            for i in xrange(3):
                name = entry[0] + str(i)
                individuals[name] = entry[2] + ["color" + str(randint(0,FT_VALS-1)), \
                                                "pose" + str(randint(0,FT_VALS-1)), \
                                                "size" + str(randint(0,FT_VALS-1)) ]                                           
    pickle.dump(individuals,outfile)
    outfile.close()

    test_items = dict()
    outfile = open("testitems.pkl",'w')
    test_names = ['dalmatian0','dalmatian1','poodle0','pug0','tabby0','manx0','flounder0','fire-truck0','motorboat0']
    for n in test_names:
        test_items[n + "T"] = individuals[n][:-3] + ["color" + str(randint(0,FT_VALS-1)), \
                                                     "pose" + str(randint(0,FT_VALS-1)), \
                                                     "size" + str(randint(0,FT_VALS-1)) ]
    print(test_items)
    pickle.dump(test_items,outfile)
    outfile.close()

l = build_lexicon()

supers = 0
basics = 0
subs = 0

for (word, level, freq) in l:
    if level == SUPERORD:
        supers += 1
    elif level == BASIC:
        basics += 1
    elif level == SUBORD:
        subs += 1
         
FT_VALS = (BASIC_FREQ *basics + SUPERO_FREQ * supers + SUBO_FREQ * subs) / 5 
print FT_VALS

(newl,adjs,dets) = pad_lexicon(l)
write_lexicon(l,newl,all_fts)
write_sentences(l,adjs,dets)
pickle_instances()
print len(all_fts)

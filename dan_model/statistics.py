import math

def getAvg(numL):
    sum = 0.0
    for a in numL:
        sum = sum + a
    avg = sum / float(len(numL))
    return(avg)

def getStd(numL):
    avg = getAvg(numL)
    sum = 0.0
    for a in numL:
        sum = sum + ((a - avg) * (a - avg))
    std = math.sqrt(sum / float(len(numL)))
    return(std)

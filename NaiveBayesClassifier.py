import sys,time, math
'''
multinomial Naive Bayes with bag-of-words
Bay of Words assumption -> we assume position of words doesn't matter
we also assume feature probabilities P(xi|cj) independent given class c
Multinomial Naive Bayes is a specific instance of Naive Bayes
where P(Featurei|Class) follows multinomial distribution
'''

def getBestFeatureList(pDict, nDict, numPosWords, numNegWords):
    '''
    just for finding best positive and negative features
    '''
    bestPosFeature={}
    bestNegFeature={}
    for word in pDict:
        val = (pDict[word]+1)/(float(numPosWords+len(pDict)))
        if word in nDict:
            val2 = (nDict[word]+1)/(float(numNegWords+len(nDict)))
        else:
            val2=1/(float(numNegWords+len(nDict))) #laplace smoothing
        bestPosFeature[word] = val-val2
    for word in nDict:
        val = (nDict[word]+1)/(float(numNegWords+len(nDict)))
        if word in pDict:
            val2 = (pDict[word]+1)/(float(numPosWords+len(pDict)))
        else:
            val2=1/(float(numPosWords+len(pDict)))
        bestNegFeature[word] = val-val2
    topPosFt = []
    topPosFtProb = []
    topNegFt = []
    topNegFtProb = []
    for i in range(10):
        topPosFt.append(max(bestPosFeature, key=bestPosFeature.get))
        topPosFtProb.append(bestPosFeature[topPosFt[-1]])
        del bestPosFeature[topPosFt[-1]]
        topNegFt.append(max(bestNegFeature, key=bestNegFeature.get))
        topNegFtProb.append(bestNegFeature[topNegFt[-1]])
        del bestNegFeature[topNegFt[-1]]
    print("*top pos features*")
    for i in range(10):
        print(topPosFt[i],topPosFtProb[i])
    print("*top neg features*")
    for i in range(10):
        print(topNegFt[i],topNegFtProb[i])


def getWordsAndTrain(reviews):
    '''
    splits reviews w/labels into individual words and places in corresponding positive or negative dictionary
    also trains 
    review[0] = review
    review[1] = label
    '''
    numPosReviews = 0
    numPosWords = 0
    numNegReviews = 0
    numNegWords = 0
    pDict={}
    nDict={}
    for review in reviews:
        if(int(review[1])==1): #positive review  
            numPosReviews+=1
            wlist = review[0].split()
            for word in wlist:
                wd = word.lower()
                if wd not in stopWords:
                    if wd in pDict:
                        pDict[wd]+=1
                        numPosWords+=1
                    else:
                        pDict[wd]=1
                        numPosWords+=1
        else: #negative review
            numNegReviews+=1
            wlist = review[0].split()
            for word in wlist:
                wd = word.lower()
                if wd not in stopWords:
                    if wd in nDict:
                        nDict[wd]+=1
                        numNegWords+=1
                    else:
                        nDict[wd]=1
                        numNegWords+=1
    #prior probabilitiy that positive
    #how many reviews mapped to class positive divided by total # reviews we wever looked at
    posPriorProb = math.log2(numPosReviews/(numPosReviews+numNegReviews)) 
    #how many reviews mapped to class negative divided by total # reviews we ever looked at
    negPriorProb = math.log2(numNegReviews/(numPosReviews+numNegReviews))
    return pDict, nDict, posPriorProb, negPriorProb, numPosWords, numNegWords

def testing(reviews, posDict, negDict, posPriorProb, negPriorProb, numPosWords, numNegWords, flag):
    '''
    tests dataset and returns accuracy 
    '''
    P=0
    N=0
    TP=0
    TN=0
    for review in reviews:
        probPosGivenPos=0
        probNegGivenPos=0
        probPosGivenNeg=0
        probNegGivenNeg=0
        if(int(review[1])==1):  #positive review
            P+=1 #P
            wlist = review[0].split() #split review to list of words
            for word in wlist: # P(w|c) = [count(w,c)+1]/[count(c)+|V|] where |V| is vocab size
                wd = word.lower()
                if wd not in stopWords:
                    if wd in posDict:
                        #Adding because using logs and comparing resulting sum rather than product 
                        probPosGivenPos+=math.log2((posDict[wd]+1)/(numPosWords+len(posDict))) #laplace smoothing. 
                    else: #word not in posDict so count will be 0
                        probPosGivenPos+=math.log2(1/(numPosWords+len(posDict))) #laplace smoothing
                    if wd in negDict:
                        probNegGivenPos+=math.log2((negDict[wd]+1)/(numNegWords+len(negDict))) #laplace smoothing
                    else: #not in negDict so count will be 0 in neg dictionary
                        probNegGivenPos+=math.log2(1/(numNegWords+len(negDict)))#laplace smoothing
            probPosGivenPos+=posPriorProb
            probNegGivenPos+=negPriorProb
            #comparing Pos and Neg rather than if >0.5 because using logs
            if probPosGivenPos>probNegGivenPos:
                TP+=1
                if(flag==1):
                    print(1)
            else:
                if(flag==1):
                    print(0)
        else: #negative review
            N+=1 #N
            wlist = review[0].split()
            for word in wlist:
                wd = word.lower()
                if wd not in stopWords:
                    if wd in negDict:
                        probNegGivenNeg+=math.log2((negDict[wd]+1)/(numNegWords+len(negDict))) #smoothing . Adding because using logs and comparing resulting sum rather than product 
                    else: #word not in negDict
                        probNegGivenNeg+=math.log2(1/(numNegWords+len(negDict)))
                    if wd in posDict:
                        probPosGivenNeg+=math.log2((posDict[wd]+1)/(numPosWords+len(posDict)))
                    else: #not in negDict
                        probPosGivenNeg+=math.log2(1/(numPosWords+len(posDict)))
            probPosGivenNeg+=posPriorProb
            probNegGivenNeg+=negPriorProb
            #comparing pos and neg rather than if >0.5 because using logs
            if probNegGivenNeg > probPosGivenNeg:
                TN+=1
                if(flag==1):
                    print(0)
            else:
                if(flag==1):
                    print(1)
    #print(TP,TN,P,N)
    Accuracy = (TP+TN)/float(P+N)
    #Accuracy = format(Accuracy,'.3f')
    return Accuracy

def Classifier(trainLines, testLines):
    '''
    Naive Bayes Classifier 
    -initial format of datasets
    -times the training and testing functions
    -returns output
        1. print labels of all values classified in public testing dataset one label per line
        2. Time it took for program to build/train the classifier in seconds
        3. Time it took your program to run classifier on public testing datasets, in seconds.
        4. The accuracy of your classifier on the training dataset as a decimal number between 0 and 1
        5. The accuracy of your classifier on the public testing dataset as a decimal number between 0 and 1
    '''
    for i in range(len(trainLines)):
        trainLines[i] = trainLines[i].split(',')
    for j in range(len(testLines)):
        testLines[j] = testLines[j].split(',')
    
    t0 = time.time()
    posDict, negDict, posPriorProb, negPriorProb, numPosWords, numNegWords = getWordsAndTrain(trainLines) #all positive and negative words not in stopList
    t1 = time.time()
    trainingAccuracy=testing(trainLines, posDict, negDict, posPriorProb, negPriorProb, numPosWords, numNegWords,0)
    t2 = time.time()
    testingAccuracy=testing(testLines, posDict, negDict, posPriorProb, negPriorProb, numPosWords, numNegWords,1)
    t3 = time.time()

    #getBestFeatureList(posDict, negDict, numPosWords, numNegWords)  #used this to find 10 best positive and negative features
    print(int(t1-t0),"seconds (training)")
    print(int(t3-t2),"seconds (labeling)")
    print("%.3f" % trainingAccuracy, end=' ')
    print("(training)",end="\n")
    print("%.3f" % testingAccuracy, end=' ')
    print("(testing)",end="\n")

if __name__ == "__main__":
    a = open(sys.argv[1],"r")
    trainLines = a.readlines()
    b = open(sys.argv[2],"r")
    testLines = b.readlines()
    #stemmer = sb.stemmer("english")
    #stemmed = stemmer.stemWord("whateverwordis")
    #stopWords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn','across','after','afterwards','all','another','become','becomes','therefore','us','hasn’t','in','made','might','able','either','ever']
    stopWords = ['i','ve','m','me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn','across','after','afterwards','all','another','become','becomes','therefore','us','hasn’t','in','made','might','able','either','ever','video','game','games','gamed','gaming','gamers','play','playing','played','plays']
    Classifier(trainLines, testLines)
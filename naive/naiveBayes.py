from Tkinter import *
import pandas as pd
from scipy.stats import mode
import tkFileDialog
import tkMessageBox
from decimal import *
from sklearn import metrics

class Analyser:
    train = None
    sructureDic = None
    numOfBins = None
    initDictionaryStruct = {}
    countRecotdsClass = {}
    struct = {}
    p = 0.0
    m = 2

#The function calculates the average and put the value at missind data
    def average(self, key):
        self.train[key].fillna(self.train[key].mean(), inplace=True)

#The function found the most common value and put the value at missind data
    def mostCommon(self, key):
        self.train[key].fillna(mode(self.train[key]).mode[0], inplace=True)

#The function handles missing numeric and category data
    def handleMissingData(self):
        for key, value in self.sructureDic.iteritems():
            if(value == "class"):
                continue
            if value == "NUMERIC":
                self.average(key)
            else:
                self.mostCommon(key)

#The function calculates equal width
    def EqualWidth(self, maxval, minval, cut_points):
        tempWidth = float(maxval) - float(minval)
        width =  float(tempWidth/ float(cut_points))
        i = 1
        binsArr = []
        while i < cut_points:
            binsArr.append(minval + (width * i))
            i += 1
        return binsArr

#The function devides to equal width
    def binning(self, col, cut_points, key):
        minval = col.min()
        maxval = col.max()
        binsArr = self.EqualWidth(maxval, minval, cut_points) #calculate bins
        labels = range(1, cut_points + 1)
        break_points = [minval] + binsArr + [maxval]
        self.initDictionaryStruct[key] = labels
        self.struct[key] = break_points
        colBin = pd.cut(col, bins=break_points, labels=labels, include_lowest=True)
        return colBin


#The function does discretization to the data set
    def Discretization(self):
        try:
            for key, value in self.sructureDic.iteritems(): #devide if numeric or categoty
                if value == "NUMERIC":
                    # print (key)
                    self.train[key] = self.binning(self.train[key], self.numOfBins, key)
                else:
                    valueList = str(value).replace("}", "").replace("{", "").split(",")
                    self.initDictionaryStruct[key] = valueList
                    self.struct[key] = valueList
        except:
            tkMessageBox.showerror("Naive Bayes Classifier", "the bins are not valid")

#The function calculates the probability of the attributes in category by the class
    def calculateProb(self, classValue, attribute, category):
        n = self.countRecotdsClass[classValue]
        countAtt = self.train[category] == attribute
        countClass = self.train['class'] == classValue
        count = self.train[countAtt & countClass]
        nc = len(count)
        result = (nc + (self.m * self.p)) / (n + self.m)
        return result

#The function calculates the probability of 'class' attribute
    def probClass(self):
        self.countRecotdsClass = self.train["class"].value_counts()

#The function creates class probability table for each class value
    def createClassProbTable(self, attribute, category):
        probDictionary = {}
        for classValue in self.initDictionaryStruct["class"]:
            prob = self.calculateProb(classValue, attribute, category)
            probDictionary[classValue] = prob
        return probDictionary

#The function create table category for each attribute
    def createTableCategory(self, attributesCategory, category):
        categoryTable = {}
        for attribute in attributesCategory:
            classProb = self.createClassProbTable(attribute, category)
            categoryTable[attribute] = classProb
        return categoryTable

#The function bulids table probability
    def buildTablesProb(self):
        bigTable = {}
        leng = len(str(self.sructureDic["class"]).replace("}", "").replace("{", "").split(","))
        self.p = float(float(1) / float(leng))
        for category, value in self.sructureDic.iteritems():
            if category == "class":
                continue
            attributesCategory = self.initDictionaryStruct[category] #take the attribue category
            categoryTable = self.createTableCategory(attributesCategory, category)
            bigTable[category] = categoryTable

        return bigTable

#The function builds the model
    def buildModel(self, sructureDic, train, numOfBins):
        self.train = train
        self.sructureDic = sructureDic
        self.numOfBins = numOfBins
        self.handleMissingData()
        self.Discretization()
        self.probClass()
        probTables = self.buildTablesProb()
        return probTables


class Classifier:
    test = None
    structIntervals = None
    structDic = None
    structLables = None

# The function does discretization to the test set
    def Discretization(self):
        for key, value in self.structDic.iteritems():
            if key == "class":
                continue
            if value == "NUMERIC":
                intervals = self.structIntervals[key]
                labels = self.structLables[key]
                self.test[key] = pd.cut(self.test[key], bins=intervals, labels=labels, include_lowest=True)

#The function writes to file the results
    def writeToFile(self, dirpath, classification):
        f = open(dirpath + "\\output.txt", "w")
        i = int(1)
        while i < len(classification):
            f.write(str(i) + " " + str(classification[int(i)]) + "\n") #write the number and the classification
            i = int(i) + int(1)
        f.close()

#The function calculates nult between probs
    def getMult(self, listProb):
        mult = Decimal(1)
        for prob in listProb:
            mult = Decimal(mult) * Decimal(prob)
        return mult

#The function gets the probability from the table
    def getProb(self, model, category, attribute, classAttribute):
        dictionary = model[category]
        classProb = dictionary[attribute]
        prob = classProb[classAttribute]
        return prob

#The function gets the classification of the row
    def getClassificationRow(self,probByVal):
        # print (probByVal)
        classification = None
        maxProb = 0
        for classVal, prob in probByVal.iteritems(): # for on probByVal
            if prob>maxProb:
                maxProb = prob
                classification = classVal
        return classification

#The function get the classification for the test
    def getClassification(self,model):
        classification = []
        for index, row in self.test.iterrows():
            probByVal = {}
            for classValue in self.structLables["class"]: #for on each class
                listProb  = []
                for category, value in self.structDic.iteritems():
                    if category =="class":
                         continue
                    attribute = row[category]
                    prob = self.getProb(model,category,attribute,classValue)
                    listProb.append(prob)
                probClassVal = self.getMult(listProb)
                probByVal[classValue] = probClassVal
            classificationRow = self.getClassificationRow(probByVal)
            classification.append(classificationRow)
        return classification

#The function classify the test by the model
    def classify(self, test, structIntervals, structDic, structLables, dirpath,model):
        self.test = test
        self.structIntervals = structIntervals
        self.structDic = structDic
        self.structLables = structLables
        self.Discretization()
        classification = self.getClassification(model)
        self.writeToFile(dirpath,classification)
        accurancy = metrics.accuracy_score(classification,test["class"])
        # print "Accuracy : %s" % "{0:.3%}".format(accurancy)
        tkMessageBox.showinfo("Naive Bayes Classifier", "classify is done!")
        exit()


class NaiveBayes:
    analyser = Analyser()
    classifier = Classifier()
    model = None

#The function give the user the search for the path
    def browse(self):
        dir = tkFileDialog.askdirectory()
        self.var.set(dir)

#The function load csv file
    def load_csv(self, filename):
        try:
            df = pd.read_csv(filename)
            return df
        except:
            tkMessageBox.showerror("Naive Bayes Classifier", "the path " + filename + " is not exist")

#The function load text file
    def load_txt(self, filename):
        try:
            d = {}
            with open(filename) as f:
                content = f.read().splitlines()
                for line in content:
                    arr = line.split(' ')
                    key = arr[1]
                    value = arr[2]
                    d[key] = value
            return d
        except:
            tkMessageBox.showerror("Naive Bayes Classifier", "the path " + filename + " is not exist") #print if the file is not exist

#The function build the model
    def build(self):
        dirPath = self.var.get()
        filename = dirPath + "\\Structure.txt"
        sructureDic = self.load_txt(filename)
        filename = dirPath + "\\train.csv"
        train = self.load_csv(filename)
        numOfBins = int(self.varBins.get())
        self.model = self.analyser.buildModel(sructureDic, train, numOfBins)
        tkMessageBox.showinfo("Naive Bayes Classifier", "Building classifier using train-set is done!")

#The function classify the test by the model
    def classify(self):
        dirPath = self.var.get()
        filename = dirPath + "\\test.csv"
        test = self.load_csv(filename)
        structIntervals = self.analyser.struct
        structDic = self.analyser.sructureDic
        structLabes = self.analyser.initDictionaryStruct
        self.classifier.classify(test, structIntervals, structDic, structLabes, dirPath,self.model)

#The function init the gui
    def __init__(self, master):
        self.master = master
        self.master.title("Naive Bayes Classifier")
        self.labelDir = Label(master, text="Directory Path")
        self.labelDisc = Label(text="Discretization Bins:")
        self.var = StringVar()
        dirname = Entry(master, textvariable=self.var)
        self.varBins = StringVar()
        bins = Entry(master, textvariable=self.varBins)
        self.browse_button = Button(master, text="Browse", command=lambda: self.browse()) #button browse
        self.build_button = Button(master, text="Build", command=lambda: self.build()) #button build
        self.classify_button = Button(master, text="Classify", command=lambda: self.classify())
        self.labelDir.grid(row=0, column=0, sticky=W)
        self.labelDisc.grid(row=1, column=0, sticky=E)
        dirname.grid(row=0, column=1, columnspan=2, sticky=W + E)
        bins.grid(row=1, column=1, columnspan=2, sticky=W + E)
        self.browse_button.grid(row=0, column=3)
        self.build_button.grid(row=2, column=1)
        self.classify_button.grid(row=3, column=1)


root = Tk()
my_gui = NaiveBayes(root)
root.mainloop()
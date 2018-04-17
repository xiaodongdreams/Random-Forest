#-*- coding: utf-8 -*-
# function read CSV file to dataSet

import csv

def loadCSV(filename):
    dataSet=[]
    with open(filename,'r') as file:
        csvReader=csv.reader(file)
        for line in csvReader:
            dataSet.append(line)
    return dataSet

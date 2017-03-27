import sys
import glob
import os
import zipfile
import re
import errno
import struct
from operator import itemgetter, attrgetter
import getopt
import shutil
import numpy as np
import random
import time

from scipy.stats import norm

#global variables
logfile         = None	#log file handler
outputfile      = None  #output file handler
logfilepathname = None
outfilepathname = None

isGenerateLog   = True
bN              = 1

nCount = 0
loops=None

#initialize the log file
def init_log_file( logfilename = None ):
	global logfile,logfilepathname,isGenerateLog,bN
	if isGenerateLog == True:
		if logfilename == None:
			logfilename = "log.log"
		logfilename = os.path.join(get_current_dir(),logfilename)
		logfilepathname = logfilename
		logfile = open(logfilename,"w")
		logfile.write("[INFO] Initialize log file:  "+logfilename);
		print ("[INFO] Initialize log file:  "+logfilename)

def open_log_file( logfilename ):
	global logfile,isGenerateLog
	if isGenerateLog == True:
		if logfilename == None:
			logfilename = "log.log"
		logfilename = os.path.join(get_current_dir(),logfilename)
		logfile = open(logfilename,"a")

#log information
def log_info(info):
	global logfile,isGenerateLog
	if isGenerateLog == True:
	   logfile.write("\n[INFO] "+info)

def log_info_raw(info):
	global logfile,isGenerateLog
	if isGenerateLog == True:
		logfile.write("\n\t"+info)

#log error information
def log_error(error):
	global logfile,isGenerateLog
	if isGenerateLog == True:
		logfile.write("\n[ERROR] "+error)

#close log file
def close_log_file():
	global logfile,isGenerateLog
	if isGenerateLog == True:
		logfile.close()

#create an output file
def create_output_file( outputfilename = None ):
	global outputfile,outfilepathname
	if outputfilename == None:
		outputfilename = "result.csv"
	outputfilename = os.path.join(get_current_dir(),outputfilename)
	outfilepathname = outputfilename
	outputfile = open(outputfilename,"w")
#	outputfile.write("simulation result:,\n");

def get_log_file_pathname():
    global logfilepathname
    return logfilepathname

def get_out_file_pathname():
    global outfilepathname
    return outfilepathname

def open_output_file( outputfilename ):
    global outputfile
    if outputfilename == None:
        outputfilename = "result.csv"
    outputfilename = os.path.join(get_current_dir(),outputfilename)
    outputfile = open(outputfilename,"a")

def save_to_output(info):
	global outputfile
	outputfile.write(info+"\n")

def close_output_file():
	global outputfile
	outputfile.close()

#return the min number of a , b
def min(a,b):
	if (a<=b):
		return a
	else:
		return b

#return the max number of a , b
def max(a,b):
	if (a<=b):
		return b
	else:
		return a

#return the key with max value
def argmin(iterable):
	minkey	 = iterable.keys()[0]
	minvalue = iterable[minkey]
	for i in iterable.keys():
		if ( iterable[i] < minvalue ):
			minkey   = i
			minvalue = iterable[i]
	return minkey

#return current directory
def get_current_dir():
	full_path = os.path.realpath(__file__)
	currentdir = os.path.dirname(full_path)

	return currentdir

#return truncate norm random number
def random_truncated_norm(mean,std):
    upper = mean+std
#    lower = mean-std
    lower = 0
    while True:
        random = np.random.normal(mean,std)
        if ( random >= lower and random <= upper):
            break
    return random

#Randomly Picking Items with Given Probabilities
def random_pick(some_list, probabilities):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        #print "picking:",x,item,item_probability
        cumulative_probability += item_probability
        if x < cumulative_probability: break
    return item

def discrete_normEX(mean, var):
    dnormdict = dict()
    for i in range (2*int(mean)):
        F1 = norm.cdf(i,mean,var)
        F2 = norm.cdf(i+1,mean,var)
        if (i + 1) == (2*int(mean)):
            F2=1
        prob = F2 - F1
        dnormdict[i+1]=prob

    return dnormdict

def discrete_norm(mean, var):
    partition = 2
    dnormdict = dict()
    for i in range (partition):
        sample = mean - partition/2 + i + 1
        F1 = norm.cdf(sample-1,mean,var)
        F2 = norm.cdf(sample,mean,var)
        if (i == 0) :
            prob = F1
            dnormdict[sample-1]=prob
            prob = F2 - F1
            dnormdict[sample]=prob
        elif (i == (partition-1)):
            prob=F2-F1
            dnormdict[sample]=prob
            prob=1-F2
            dnormdict[sample+1]=prob
        else:
            prob = F2 - F1
            dnormdict[sample]=prob

    return dnormdict
#read parameters from ini file
def read_param_from_file():
	log_info("Enter read_param_from_file()")

	parmfilename = os.path.join(get_current_dir(),"parameters.ini")
	if os.path.isfile(parmfilename) == False:
		log_error("parameters.ini file is not exist!")
		return None
	log_info("------Parameters from ini file------")
	parmfile = open(parmfilename,"r")
	param_key_values = {}
	line = parmfile.readline()
	while line:
		stripeline = line.strip()
		if ( stripeline[0:1] == "#" ):
			line = parmfile.readline()
			continue
		key_value = stripeline.split("#")[0].split("=")
		if ( len(key_value) >= 2 ):
			param_key_values[key_value[0].strip()] = key_value[1].strip()
			log_info("\t\t"+key_value[0].strip()+"="+key_value[1].strip())
		line = parmfile.readline()
	parmfile.close()
	log_info("------------End------------")
	return param_key_values

#read parameters from ini file, and define corresponding global variabes
def read_and_define_param_from_file():
	log_info("Enter read_and_define_param_from_file()")

	parmfilename = os.path.join(get_current_dir(),"parameters.ini")
	if os.path.isfile(parmfilename) == False:
		log_error("parameters.ini file is not exist!")
		return None
	log_info("------Parameters from ini file------")
	parmfile = open(parmfilename,"r")
	param_key_values = {}
	line = parmfile.readline()
	while line:
		stripeline = line.strip()
		if ( stripeline[0:1] == "#" ):
			line = parmfile.readline()
			continue
		key_value = stripeline.split("#")[0].split("=")
		if ( len(key_value) >= 2 ):
			globals()[key_value[0].strip()] = key_value[1].strip()
			log_info("\t\t"+key_value[0].strip()+"="+key_value[1].strip())
		line = parmfile.readline()
	parmfile.close()
	log_info("------------End------------")
	return param_key_values

def timeElapse():
    sum=0
    start = time.time()
    print "Time Start:"
    for i in range(pow(2,25)):
        sum = i

    print "Time Elapsed:", time.time()-start

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def normalize(x):
    if np.all( x == 0 ):
        return np.zeros(len(x))
    sum = float(x.sum())
    return x / sum

def check_sum_dict_prob(dict):
    sum = 0
    for key in dict.keys():
        sum+=dict[key]
    if sum < 0.999:
        utility.log_error("check_sum_dict_prob(): prob sum error")
        print "_check_sum_dict_prob error,sum : ",sum

def n_for(level,MAX_LEVEL):
	global nCount,loops
	if level == MAX_LEVEL:
		print "reach max level"
		for j in range(MAX_LEVEL):
			nCount+=1
			print "level,loops[j]",j,loops[j]
		return
	else:
		newlevel = level + 1
		for i in range(6):
			loops[level]=i
#			print "level,i:",level,i
			n_for(newlevel,MAX_LEVEL)

def main():
	return 0

if __name__ == '__main__':
    main()

#This script retrieves level 2 nexrad data from an AWS server and saves them locally
#the files are around 5mb each with ~300 files per day
from xml.dom import minidom
from urllib.request import urlopen
import os
from datetime import date, timedelta

#constants: start_date gives the first day to retrieve data for. N_days is the 
#number of days following the start_date to retrieve data.
#Currently set to 1 day for testing, see dates used in paper.
start_date = date(2016,11,7)
N_days = 1
site = "KLGX"
bucketURL = "http://noaa-nexrad-level2.s3.amazonaws.com"

def getText(nodelist):
    rc = []
    for node in nodelist:
        if node.nodeType == node.TEXT_NODE:
            rc.append(node.data)
    return ''.join(rc)

#loops over days following the start date and downloads all files for each day
for i in range(N_days):
    datestr = str(start_date+timedelta(i)).replace('-','/')
    dirListURL = bucketURL+ "/?prefix=" + datestr + "/" + site
    print("listing files from %s" % dirListURL)
    xmldoc = minidom.parse(urlopen(dirListURL))
    itemlist = xmldoc.getElementsByTagName('Key')
    print(len(itemlist) , "keys found...")
    for x in itemlist:
        file = getText(x.childNodes)
        print("Processing %s " % file)
        try:
            os.system("wget %s/%s -t 20 -P ./data/l2/ "%(bucketURL,file))
        except:
            print('wget failed! moving to next file...')
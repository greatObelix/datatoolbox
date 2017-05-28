# -*- coding: utf-8 -*-
"""
Created on Tue May 23 21:13:58 2017

@author: mh636c
"""

from collections import defaultdict
from copy import deepcopy
import glob
import datetime as dt
import numpy as np
import gzip

class calc_diff():

    ######################################
    # class constructor
    ######################################
    def __init__(self):
        # column header for input file        
        self.header = ['DATETIMEUTC','VROUTER_NAME','OUT_TPKTS','IN_TPKTS','OUT_BYTES','IN_BYTES','STATUS','DOWN_INTERFACE_COUNT','INTERFACE_COUNT','VHOST_GATEWAY','VHOST_IP','VHOST_PREFIX','VHOST_IF_MAC_ADDR','VN_COUNT','AGED_FLOWS','BUFFER_MEM','CACHED_MEM','FREE_MEM','TOTAL_MEM','USED_MEM','ARP_NOT_ME','CKSUM_ERR','CLONE_FAIL','CONED_ORIGINAL','COMPOSITE_INVALID_INTERF','DISCARD','DUPLICATED','FLOOD','FLOW_ACTION_DROP','FLOW_ACTION_INVALID','FLOW_INVALID_PROTOCOL','FLOW_NAT_NO_FLOW','FLOW_NO_MEMORY','FLOW_QUEUE_LIMIT','FLOW_TABLE_FULL','FLOW_UNUSABLE','FRAG_ERR','GARP_FROM_VM','HEAD_ALLOC_FAIL','HEAD_SPACE_RESERVE_FAIL','INTERFACE_DROP','INTERFACE_RX_DISCARD','INTERFACE_TX_DISCARD','INVALID_ARP','INVALID_IF','INVALID_LABEL','INVALID_MCAST_SOURCE','INVALID_NH','INVALID_PACKET','INVALID_PROTOCOL','INVALID_SOURCE','INVALID_VNID','MCAST_CLONE_FAIL','MCAST_DF_BIT','MISC','NO_FMD','NOWHERE_TO_GO','PCOW_FAIL','PULL','PUSH','REWRITE_FAIL','TRAP_NO_IF','TTL_EXCEEDED','EXCEP_PACKETS_ALLOWED','EXCEP_PACKETS_DROPPED','ACTIVE_FLOWS','ADDED_FLOWS','DELETED_FLOWS','MAX_FLOW_ADDS','MAX_FLOW_DELETES','MIN_FLOW_ADDS','MIN_FLOWS_DELETES','TOTAL_FLOWS','UPTIME','VHOST_IN_BYTES','VHOST_OUT_BYTES','VHOST_IN_PTKTS','VHOST_OUT_PTKTS']
        # column headers that are cumulative
        self.cumheader = ['OUT_TPKTS','IN_TPKTS','OUT_BYTES','IN_BYTES','AGED_FLOWS','ARP_NOT_ME','CKSUM_ERR','CLONE_FAIL','CONED_ORIGINAL','COMPOSITE_INVALID_INTERF','DISCARD','DUPLICATED','FLOOD','FLOW_ACTION_DROP','FLOW_ACTION_INVALID','FLOW_INVALID_PROTOCOL','FLOW_NAT_NO_FLOW','FLOW_NO_MEMORY','FLOW_QUEUE_LIMIT','FLOW_TABLE_FULL','FLOW_UNUSABLE','FRAG_ERR','GARP_FROM_VM','HEAD_ALLOC_FAIL','HEAD_SPACE_RESERVE_FAIL','INTERFACE_DROP','INTERFACE_RX_DISCARD','INTERFACE_TX_DISCARD','INVALID_ARP','INVALID_IF','INVALID_LABEL','INVALID_MCAST_SOURCE','INVALID_NH','INVALID_PACKET','INVALID_PROTOCOL','INVALID_SOURCE','INVALID_VNID','MCAST_CLONE_FAIL','MCAST_DF_BIT','MISC','NO_FMD','NOWHERE_TO_GO','PCOW_FAIL','PULL','PUSH','REWRITE_FAIL','TRAP_NO_IF','TTL_EXCEEDED','EXCEP_PACKETS_ALLOWED','EXCEP_PACKETS_DROPPED','VHOST_IN_BYTES','VHOST_OUT_BYTES','VHOST_IN_PTKTS','VHOST_OUT_PTKTS']
        
        # internal data object as hash, organize by vrouter{timestamp{metric1:value1, metric2:value2,...}}
        # create 3level of nested dict, store original data from json,
        # some counters are cummulative
        self.vrouters = defaultdict(lambda: defaultdict(dict))
        # take care of cummulative counters to find delta per interval
        self.vrouters_delta = defaultdict(lambda: defaultdict(dict))

        return
        
    ######################################
    # read input file(s), helper fcn to invoke actual read fcn that can be overloaded depend on the input
    ######################################        
    def getraw(self,path,folder=False):
        if (folder):
            # going through list of files in a given dir
            fileList = glob.glob(path)
            for fname in fileList:
                self.readcsv(fname)    
        else:
            self.readcsv(path)

        return
        
    ######################################
    # read each csv file
    # can be overloaded to read diff input file format
    ######################################            
    def readcsv(self,fname):
        #fin = open(fname,'r')
        with gzip.open(fname,'rt') as fin:
            for line in fin:
                if (line[0] == '#' or line.find('DATETIMEUTC') > -1 ):
                    continue
                else:
                    try:
                        tokens = line.split(',')
                        for i in range(2,12):
                            self.vrouters[tokens[3]][tokens[0]][self.header[i]] = tokens[i+2]
                        self.vrouters[tokens[3]][tokens[0]][self.header[12]] = tokens[15]
                        for i in range(13,len(self.header)):
                            self.vrouters[tokens[3]][tokens[0]][self.header[i]] = float(tokens[i+4])
                    except:
                        print('ERROR: readcsv()', line)

        return        

    ######################################
    # print content of vrouter data structure
    #####################################
    def prettyprint(self):
        print(self.vrouters)

    ######################################
    # return how many minutes passed between end time and start time
    #####################################
    def calc_min(self, start, end):
        sdate = dt.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
        edate = dt.datetime.strptime(end, '%Y-%m-%d %H:%M:%S')
        diff = edate - sdate
        return (float(diff.total_seconds())/60)
        
    ######################################
    # calculate diff between interval because counters are cummulative
    # side effect: remove first entry in the timeline
    # same as calc_delta() except deal with missing timestamp intervals by 
    # finding how many 5min interval missing, finding avg per interval
    # juniper defined max value = u64, so 2^64 is max in case overflow
    ######################################
    def calc_delta2(self):
        self.vrouters_delta = deepcopy(self.vrouters)
        for vr in self.vrouters.keys():
            t=-1
            ts_prev=0
            ts_org=0
            for ts in sorted(self.vrouters[vr].keys()):
                t += 1                
                if (t==0):
                    ts_org=ts
                    ts_prev = ts
                    continue
                for h in self.cumheader:
                    try:
                        tmp = float (self.vrouters[vr][ts][h]) - float(self.vrouters[vr][ts_prev][h])
                    except:
                        print(vr+' '+ts+' '+h+' '+self.vrouters[vr][ts][h]+'\n')
                    num_5min_interval = self.calc_min(ts_prev,ts)/5
                    if (tmp < 0):
                        # if negative, either vrouter reboot or counter overflow/rollover
                        # vrouter reboot case: check uptime, and take the new value as is
                        if(self.vrouters[vr][ts]['UPTIME'] > self.vrouters[vr][ts_prev]['UPTIME']):
                            tmp = float(self.vrouters[vr][ts][h])
                        # counter overflow: the cummulative counter has rollover, need to take max - previous_interval + current_interval
                        #print('data rollover detected: ',vr,ts,h,tmp,type(self.vrouters[vr][ts][h]))
                        else:
                            tmp = (np.exp2(64) - float(self.vrouters[vr][ts_prev][h])) + float(self.vrouters[vr][ts][h])
                    if (num_5min_interval > 1):
                        self.vrouters_delta[vr][ts][h] = tmp/num_5min_interval
                    else:
                        self.vrouters_delta[vr][ts][h] = tmp
                ts_prev = ts
            del self.vrouters_delta[vr][ts_org]
                       
    ######################################
    # write stats to file
    # stats got from internal obj, possible output took care of cummulative counters if fcn is called
    ######################################                
    def writefilestats_delta(self, fname):
        with open(fname,'w') as fout:
            fout.write(','.join(self.header))
            fout.write('\n')
            for vr in self.vrouters_delta.keys():
                for ts in sorted(self.vrouters_delta[vr].keys()):
                    fout.write(','.join(map(str,[ts,vr])))
                    for h in self.header[2:]:
                        fout.write(','+str(self.vrouters_delta[vr][ts][h]))
                    fout.write('\n')        

if __name__ == '__main__':
    fname_in = 'vertica/sqlout_CONTRAIL_vrouter_20170301_20170331.csv.gz'    
    fname_out = 'vertica/sqlout_CONTRAIL_vrouter_20170301_20170331_diff.csv'
    vr = calc_diff()
    vr.getraw(fname_in)
    vr.calc_delta2()
    vr.writefilestats_delta(fname_out)
    # compress output file
    #with open(fname_out, 'rb') as f_in:
    #    with gzip.open('vertica/test_diff.csv.gz', 'wb') as f_out:
    #        shutil.copyfileobj(f_in, f_out)
########################################################
""" IMPORT PACKAGES """
import pandas as pd
import numpy as np
import datetime
import time
from pandas import to_datetime
import matplotlib.pyplot as plt
import seaborn as sb
from numpy import linalg as LA
from datetime import timedelta
import pickle
import ast
from collections import Counter
from collections import OrderedDict
import math
import itertools
########################################################
""" IMPORT DATA """

#Stations Info
stations_18=pd.read_csv("stations_18.csv.gz",compression='gzip')

#Check-Out Pattern at Station S (*U* - tuple of 5 elements)
df_checkout_pattern_all=pd.read_csv("df_checkout_pattern_all.csv.gz", compression='gzip')

#Pairwise Distances
df_distances=pd.read_csv("pairwise_distances.csv.gz",compression='gzip')
df_distances.index=df_distances['Unnamed: 0'].values
df_distances.pop('Unnamed: 0')
########################################################
""" AUXILIAR FUNCTIONS """

def nearest(items, pivot):
    return min(items, key=lambda x: abs(timedelta(hours= x.hour, minutes=x.minute, seconds= x.second) - \
                                    timedelta(hours=pivot.hour,minutes=pivot.minute,seconds=pivot.second)))

def check_period_ts(ts,t):
    flag=False
    if ts==1:
        start_time=7
        stop_time=11
        day='Weekday'
        nr_days=261
        periods=4
        if 0 <= t <= 3: 
            flag=True
        else:
            raise Exception('Wrong number of periods for time-slot 1')
    elif ts==2:
        start_time=12
        stop_time=16
        day='Weekday'
        nr_days=261
        periods=4
        if 0 <= t <= 3: 
            flag=True
        else:
            raise Exception('Wrong number of periods for time-slot 2')
    elif ts==3:
        start_time=17
        stop_time=22
        day='Weekday'
        nr_days=261
        periods=5
        if 0 <= t <= 4:
            flag=True
        else:
            raise Exception('Wrong number of periods for time-slot 3')
    elif ts==4:
        start_time=9
        stop_time=17
        day='Weekend'
        nr_days=104
        periods=8
        if 0 <= t <= 7:
            flag=True
        else:
            raise Exception('Wrong number of periods for time-slot 4')
    elif ts==5:
        start_time=18
        stop_time=23
        day='Weekend'
        nr_days=104
        periods=5
        if 0 <= t <= 4:  
            flag=True
        else:
            raise Exception('Wrong number of periods for time-slot 5')
    
    return flag,start_time,stop_time,day,nr_days,periods
########################################################
""" NR OF BIKES RENTED FUNCTION """

#Number of Bikes Rented in Station S in Time-Slot TS and in period T
def nr_bikes_rented_ts(df_ts,station,ts,t):
    if station in stations_18['citibike_station_id'].values:
        if check_period_ts(ts,t)[0]: 
            df=df_ts.loc[ (df_ts['start_station_id']==station) &
                            (df_ts['start_time_wf']>= datetime.time(check_period_ts(ts,t)[1]+t,0,0,0)) &
                            (df_ts['start_time_wf'] < datetime.time(check_period_ts(ts,t)[1]+t+1,0,0,0))]
            return len(df)
    else:
        raise Exception('Wrong/Nonexistent CitiBike Station ID!')
########################################################
""" CLUSTERING AdaTC """

print('Run AdaTC Classes and Plots Code')
# 1. Geo-clustering: K-MEDOIDS
class GeoC1:
    
    def __init__(self,k1,max_iter,ro1): 
        self.k1 = k1
        self.max_iter = max_iter
        self.ro1=ro1
        self.df_diss={}
        for j in range(1,802):
            sj=stations_18['citibike_station_id'].loc[stations_18['distances_id']==j].values[0]
            self.df_diss['diss_' + str(sj)] = pd.DataFrame(columns=['diss'],index=stations_18['citibike_station_id'])

        print("Geo-clustering initalized with %s clusters with ro1=%s" %(str(k1),str(ro1)))

    #DISSIMILARITY FUNCTION
    ###############################################################################################################
    #geographical distance between station 1 and station 2 (s1 and s2 are the Citibike indexes of the stations) 
    def geo_dist(self,s1,s2):
        i1=stations_18['distances_id'].loc[stations_18['citibike_station_id']==s1].values[0]
        i2=stations_18['distances_id'].loc[stations_18['citibike_station_id']==s2].values[0]
        dist=np.array(df_distances)[i1-1,i2-1]
        return (dist)
     
    #check-out difference between station 1 and station 2 (s1 and s2 are the Citibike indexes of the stations)
    #U is the check-out pattern at some station S
    #U1 e U2 in tuple form -> U1 = check_out_pattern(s1); U2 = check_out_pattern(s2)
    def check_out_diff(self,U1,U2):
        return (LA.norm(np.subtract(U1,U2),ord=2))

    #define dissimilarity 
    #saving dissimilarities as long as they are calculated
    def GC_diss(self,s1,s2,U1,U2):
        if self.df_diss['diss_%s' %str(float(s1))].loc[float(s2)].values[0] is np.nan:
            self.df_diss['diss_%s' %str(float(s1))].loc[float(s2)]=self.ro1 *self.geo_dist(s1,s2) + self.check_out_diff(U1,U2)
            out=round(self.df_diss['diss_%s' %str(float(s1))].loc[float(s2)].values[0],3)

        else:
            out=round(self.df_diss['diss_%s' %str(float(s1))].loc[float(s2)].values[0],3)

        return (out)

    #calculate new medoids 
    def new_medoids(self,k1,classes):
        print('CALCULATING NEW MEDOIDS')
        self.medoids=[]

        for k in range(len(self.classes)):
            if len(self.classes[k])==1:
                self.medoids.append(self.classes[k][0])
            
            else:
                i=1
                values=list(self.classes[k])
                dist_df=np.zeros((len(values),len(values))) + np.diag(np.full(len(values),float('NaN')))
                for l in values:
                    for m in values[i:]:
                        U1=ast.literal_eval(df_checkout_pattern_all['Checkout_Pattern'].loc[df_checkout_pattern_all['Stations_CB_id']==l].values[0])
                        U2=ast.literal_eval(df_checkout_pattern_all['Checkout_Pattern'].loc[df_checkout_pattern_all['Stations_CB_id']==m].values[0])
                        dist_df[values.index(l),values.index(m)]=self.GC_diss(l,m,U1,U2)     
                    i=i+1
                dist_df_sym=np.triu(dist_df) + np.triu(dist_df,1).T
                self.medoids.append(values[Counter(np.nanargmin(dist_df_sym,axis=0)).most_common()[0][0]])
                
        if len(self.medoids)!= k1:
            raise Exception('More than %s medoids' %str(k1))
    
        return self.medoids

    ###############################################################################################################
    #Takes the data and forms clusters based on the dissimilarity
    ###############################################################################################################
    def fit(self, data):
        self.medoids = []
        self.iters=[]
        self.diss_tot=[]
        
        #Initialize the k1 initial medoids with first k1 points from the dataset
        for i in range(self.k1):
            self.medoids.append(data.iloc[i])

        #Dissimilarity measure between the data point and the identified k1-medoids
        #Assigns each data point to a cluster based on the dissimilarity minimization
        for itr in range(self.max_iter):
            print('STARTING GEO-CLUSTERING ITERATION %s' %itr)
            self.classes = {}
            for cluster in range(self.k1):
                self.classes[cluster] = []

            self.diss_acum=0
            for point in range(len(data)):
                distances=[None]*self.k1
                for c in range(len(self.medoids)):
                    s1=self.medoids[c]
                    s2=data.iloc[point]
                    U1=ast.literal_eval(df_checkout_pattern_all['Checkout_Pattern'].loc[df_checkout_pattern_all['Stations_CB_id']==s1].values[0])
                    U2=ast.literal_eval(df_checkout_pattern_all['Checkout_Pattern'].loc[df_checkout_pattern_all['Stations_CB_id']==s2].values[0])
                    distances[c] = self.GC_diss(s1,s2,U1,U2)
            
                ##Validating Max_Iter
                self.diss_acum=self.diss_acum+np.min(distances)
                classification = np.argmin(distances)
                self.classes[classification].append(data.iloc[point])
               
            ##Validating Max_Iter
            self.iters.append(itr+1)
            self.diss_tot.append(self.diss_acum)
            
            #Calculate the new medoids
            previous=self.medoids
            medoids_upd=self.new_medoids(self.k1,self.classes)
                                
            optimal = True
            curr = medoids_upd

                    
            #Difference in the medoids of two consecutive iterations to declare convergence.
            if sorted(curr) != sorted(previous):
                optimal = False
                self.medoids=medoids_upd
    
            #Break out of the main loop if the medoids don't change in two consecutive iterations
            if optimal:
                    print("MEDOIDS DOESN'T CHANGE IN TWO CONSECUTIVE ITERATIONS - STOP")
                    break
"""# ------------------------------------------------------------"""
# 2. T-Matrix Generation

class T_Matrix:
    
    def __init__(self,clf,save_in_it1,save_in_itN,it): 
        self.clf=clf
        self.save_in_it1=save_in_it1
        self.save_in_itN=save_in_itN
        self.it=it
    ##################################################################################################
    def ride_to_cluster(self,s,ts,points,nr_k1):
        #data_all has the dataframe of trips by timeslots
	data_all=[data_18_all_ts1,data_18_all_ts2,data_18_all_ts3,data_18_all_ts4,data_18_all_ts5]
        data=data_all[ts-1]
        data_s=data.loc[data['start_station_id']==s]
        rtc_tuple=[None]*nr_k1

        for cs in range(0,nr_k1):
            data_s_others=data_s.loc[data_s['end_station_id'].isin(points[cs])]
            rtc_tuple[cs]=len(data_s_others)/len(data)

        if all(v == 0 for v in rtc_tuple):
            prob=rtc_tuple
            ind=[key for (key, x) in points.items() if s in x][0]
            prob[ind]=1

        else:
            prob=[round(i/sum(rtc_tuple),4) for i in rtc_tuple] 

        return tuple(prob)
        
    ##################################################
    def return_from_cluster(self,s,ts,points,nr_k1):
        data_all=[data_18_all_ts1,data_18_all_ts2,data_18_all_ts3,data_18_all_ts4,data_18_all_ts5]
        data=data_all[ts-1]
        data_s=data.loc[data['end_station_id']==s]
        rfc_tuple=[None]*nr_k1

        for cs in range(0,nr_k1):
            data_s_others=data_s.loc[data_s['start_station_id'].isin(points[cs])]
            rfc_tuple[cs]=len(data_s_others)/len(data) 

        if all(v == 0 for v in rfc_tuple):
            prob=rfc_tuple
            ind=[key for (key, x) in points.items() if s in x][0]
            prob[ind]=1
    
        else:
            prob=[round(i/sum(rfc_tuple),4) for i in rfc_tuple] 
        
        return tuple(prob)
    ###################################################################################################
    
    def fit(self, data):
        ###################################
        if self.it==1:
            classes_act=self.clf.classes
            self.nr_k1=self.clf.k1
        else:
            classes_act=self.clf['new_clusters']
            self.nr_k1=self.clf['k1']
        ###################################
        stats_dist_id=data.values
        print("Computing T-Matrix for %s stations" %(str(len(stats_dist_id))))

        self.all_mat={}
        for s in stats_dist_id:
            if (s==150) | (s==300) | (s==450) | (s==600) | (s==750) :
                print("Done for %s stations" %str(s))

            self.arr_stat=np.zeros((5,2*self.nr_k1))
            for ts in range(0,5):
                s_cb=stations_18['citibike_station_id'].loc[stations_18['distances_id']==s].values[0]
                row_i=self.ride_to_cluster(s_cb,ts+1,classes_act,self.nr_k1)+self.return_from_cluster(s_cb,ts+1,classes_act,self.nr_k1)
                self.arr_stat[ts]=row_i

            self.all_mat[s-1]=self.arr_stat
"""# ------------------------------------------------------------"""
# 3. T-Clustering: K-MEDOIDS

class TC1:
    
    def __init__(self,k2,max_iter,tm,its): 
        self.k2 = k2
        self.max_iter = max_iter
        self.df_diss={}
        self.tm=tm
        for j in range(1,802):
            sj=stations_18['citibike_station_id'].loc[stations_18['distances_id']==j].values[0]
            self.df_diss['diss_' + str(sj)] = pd.DataFrame(columns=['diss'],index=stations_18['citibike_station_id'])

        if self.k2 > self.tm.nr_k1:
            raise Exception('K2 > K1')

        print("T-clustering initalized with %s clusters" %str(k2))

    #DISSIMILARITY FUNCTION
    ##############################################################################################################
    #define dissimilarity 
    #saving dissimilarities as long as they are calculated
    def TC_diss(self,s1,s2):
        if self.df_diss['diss_%s' %str(float(s1))].loc[float(s2)].values[0] is np.nan:
            i1=stations_18['distances_id'].loc[stations_18['citibike_station_id']==s1].values[0]
            i2=stations_18['distances_id'].loc[stations_18['citibike_station_id']==s2].values[0]
            t1=self.tm.all_mat[i1-1]
            t2=self.tm.all_mat[i2-1]
            self.df_diss['diss_%s' %str(float(s1))].loc[float(s2)]=LA.norm(np.subtract(t1,t2), 'fro')
            out=round(self.df_diss['diss_%s' %str(float(s1))].loc[float(s2)].values[0],3)
        else:
            out=round(self.df_diss['diss_%s' %str(float(s1))].loc[float(s2)].values[0],3)

        return (out)

    #calculate new medoids 
    def new_medoids(self,k2,classes):
        print('CALCULATING NEW MEDOIDS')
        self.medoids=[]
    
        for k in range(len(self.classes)):
            if len(self.classes[k])==1:
                self.medoids.append(self.classes[k][0])
            else:
                i=1
                values=list(self.classes[k])
                dist_df=np.zeros((len(values),len(values))) + np.diag(np.full(len(values),float('NaN')))
                for l in values:
                    for m in values[i:]:
                        dist_df[values.index(l),values.index(m)]=self.TC_diss(l,m)
                    i=i+1
                dist_df_sym=np.triu(dist_df) + np.triu(dist_df,1).T
                self.medoids.append(values[Counter(np.nanargmin(dist_df_sym,axis=0)).most_common()[0][0]])
                
        if len(self.medoids)!= k2:
            raise Exception('More than %s medoids' %str(k2))
    
        return self.medoids

    ###############################################################################################################
    # Takes the data and forms clusters based on the dissimilarity
    ###############################################################################################################
    def fit(self, data):
        self.medoids = []
        self.iters=[]
        self.diss_tot=[]
        
        # Initialize the k2 initial medoids with first k2 points from the dataset
        for i in range(self.k2):
            self.medoids.append(data.iloc[i])

        #Dissimilarity measure between the data point and the identified k2-medoids
        #Assigns each data point to a medoid based on the dissimilarity
        for itr in range(self.max_iter):
            print('STARTING T-CLUSTERING ITERATION %s' %itr)
            self.classes = {}
            for cluster in range(self.k2):
                self.classes[cluster] = []

            self.diss_acum=0
            for point in range(len(data)):
                distances=[None]*self.k2
                for c in range(len(self.medoids)):
                    s1=self.medoids[c]
                    s2=data.iloc[point]
                    distances[c] = self.TC_diss(s1,s2)

                ##Validating Max Iter
                self.diss_acum=self.diss_acum+np.min(distances)
                classification = np.argmin(distances)
                self.classes[classification].append(data.iloc[point])

            #Validating Max Iters
            self.iters.append(itr+1)
            self.diss_tot.append(self.diss_acum)

            #Calculate the new medoids
            previous=self.medoids
            medoids_upd=self.new_medoids(self.k2,self.classes)
                                
            optimal = True
            curr = medoids_upd

            #Difference in the medoids of two consecutive iterations to declare convergence.
            if sorted(curr) != sorted(previous):
                optimal = False
                self.medoids=medoids_upd
    
            #Break out of the main loop if the medoids don't change in two consecutive iterations 
            if optimal:
                    print("MEDOIDS DOESN'T CHANGE IN TWO CONSECUTIVE ITERATIONS - STOP")
                    break
"""# ------------------------------------------------------------"""
# 4. AdaTC Auxiliar Functions
print('Running AdaTC Auxiliar Functions')

def new_nr_groups(clf,tlf,n,its): 
    k1_classes=[]
    k1_classes_dec=[]
    k1_classes_int=[]
    k1_classes_final=np.zeros((tlf.k2),dtype=int)
    nr_elem_class=[]
    ###############################
    if its==2:
        nr_k1=clf.k1
    else:
        nr_k1=clf['k1']
    ###############################
    for cluster in range(0,len(tlf.classes)):
        k1_classes.append(len(tlf.classes[cluster])*nr_k1/n)
        nr_elem_class.append(len(tlf.classes[cluster]))
    print(nr_elem_class)

    round_up=[]
    for k in range(0,len(k1_classes)):
        round_up.append(math.ceil(k1_classes[k]))

    round_down=[]
    for k in range(0,len(k1_classes)):
        round_down.append(math.floor(k1_classes[k]))

    for k in range(0,len(k1_classes)):
        if k1_classes[k].is_integer():
            k1_classes_int.append(k)
        else:
            k1_classes_dec.append(k)

    for m in range(0,len(k1_classes_final)):
        if int(k1_classes[m])==0:
            k1_classes_final[m]=1
        else:
            k1_classes_final[m]=round(k1_classes[m])

    ###########################################################################
    if sum(k1_classes_final)==nr_k1:
        output=k1_classes_final
    ###########################################################################
    #excess case
    elif sum(k1_classes_final)>nr_k1:
        ##########################################
        excess_act=sum(k1_classes_final)-nr_k1
        ##########################################
        ind=[]
        for j in range(0,len(k1_classes_final)):
            if (k1_classes_final[j]!=1) and (j not in k1_classes_int):
                ind.append(j)
            
        decimal_list_ind=[]
        for k in ind:
            decimal_list_ind.append((k1_classes[k]%1,k))
            decimal_list_ind.sort(reverse=False)

        later_dec=[]
        later_later_round=[]
        for it in range(0,len(decimal_list_ind)):
            if round_down[decimal_list_ind[it][1]]==k1_classes_final[decimal_list_ind[it][1]]:
                k1_classes_final[decimal_list_ind[it][1]]=k1_classes_final[decimal_list_ind[it][1]]-1
                excess_act=excess_act-1
                if k1_classes_final[decimal_list_ind[it][1]]!=1:
                    later_dec.append(decimal_list_ind[it][1])   
                if excess_act==0:
                    break
            else:
                excess_act=excess_act-(k1_classes_final[decimal_list_ind[it][1]]-round_down[decimal_list_ind[it][1]])
                k1_classes_final[decimal_list_ind[it][1]]=round_down[decimal_list_ind[it][1]]
                if k1_classes_final[decimal_list_ind[it][1]]!=1:
                   later_later_round.append(decimal_list_ind[it][1])
                if excess_act==0:
                    break

        if excess_act>0:
           new_later_dec=[]
           for t in later_dec:
               k1_classes_final[t]=k1_classes_final[t]-1
               excess_act=excess_act-1
               if k1_classes_final[t]!=1:
                  new_later_dec.append(t)
               if excess_act==0:
                  break

           new_later_dec2=[]
           if excess_act>0:
               for f in new_later_dec:
                  k1_classes_final[f]=k1_classes_final[f]-1
                  excess_act=excess_act-1
                  if k1_classes_final[f]!=1:
                    new_later_dec2.append(f)
                  if excess_act==0:
                     break

           later_later_round2=[]
           if excess_act>0:
               for q in later_later_round:
                  k1_classes_final[q]=k1_classes_final[q]-1
                  excess_act=excess_act-1
                  if k1_classes_final[q]!=1:
                     later_later_round2.append(q)
                  if excess_act==0:
                     break

           last_trial=list(set(new_later_dec2+later_later_round2))
           if excess_act>0:
               for t2 in last_trial:
                  k1_classes_final[t2]=k1_classes_final[t2]-1
                  excess_act=excess_act-1
                  if excess_act==0:
                     break

           if excess_act>0:
            dec_more=[]
            for l in range(0,len(decimal_list_ind)):
                if k1_classes_final[decimal_list_ind[l][1]]!=1 :
                    dec_more.append(decimal_list_ind[l][1])
            dec_more_new=[]
            flag=True
            while flag:
                for t3 in dec_more:
                    k1_classes_final[t3]=k1_classes_final[t3]-1
                    if k1_classes_final[t3]!=1:
                        dec_more_new.append(t3)
                    excess_act=excess_act-1
                    if excess_act==0:
                        flag=False
                        break
                    dec_more=dec_more_new

        output=k1_classes_final
    ###########################################################################
    #missing case
    else:
        ##########################################
        def_act=abs(sum(k1_classes_final)-nr_k1)
        ##########################################
        ind=[]
        for j in range(0,len(k1_classes_final)):
            if k1_classes_final[j] not in k1_classes_int:
                ind.append(j)

        decimal_list_ind=[]
        for k in ind:
            decimal_list_ind.append((k1_classes[k]%1,k))
            decimal_list_ind.sort(reverse=True)

        later_dec=[]
        later_later_round=[]
        for it in range(0,len(decimal_list_ind)):
            if round_up[decimal_list_ind[it][1]]==k1_classes_final[decimal_list_ind[it][1]]:
                k1_classes_final[decimal_list_ind[it][1]]=k1_classes_final[decimal_list_ind[it][1]]+1
                def_act=def_act-1
                later_dec.append(decimal_list_ind[it][1]) 
                if def_act==0:
                    break
            else:
                def_act=def_act-(abs(k1_classes_final[decimal_list_ind[it][1]]-round_up[decimal_list_ind[it][1]]))
                k1_classes_final[decimal_list_ind[it][1]]=round_up[decimal_list_ind[it][1]]
                later_later_round.append(decimal_list_ind[it][1])
                if def_act==0:
                    break

        if def_act>0:
           for t in later_dec:
              k1_classes_final[t]=k1_classes_final[t]+1
              def_act=def_act-1
              if def_act==0:
                 break

           if def_act>0:
               for q in later_later_round:
                  k1_classes_final[q]=k1_classes_final[q]+1
                  def_act=def_act-1
                  if def_act==0:
                     break

           last_trial=list(set(later_dec+later_later_round))
           if def_act>0:
              for t2 in last_trial:
                 k1_classes_final[t2]=k1_classes_final[t2]+1
                 def_act=def_act-1
                 if def_act==0:
                    break

           if def_act>0:
            inc_more=[]
            for l in range(0,len(decimal_list_ind)):
                inc_more.append(decimal_list_ind[l][1])
            flag=True
            while flag:
                for t3 in inc_more:
                    k1_classes_final[t3]=k1_classes_final[t3]+1
                    def_act=def_act-1
                    if def_act==0:
                        flag=False
                        break

        output=k1_classes_final
    ######################################################################################
    print(output)
    if (sum(output)!=nr_k1) or (len(output)!=tlf.k2) or (any(i==0 for i in output)) or (any(i<0 for i in output)):
        raise Exception('Wrong group numbers') 

    return (output)

def concatenate_into_k1_groups(k1,clf_clusters):

    ind_lists=[]
    start_acum=len(clf_clusters['new_groups_in_cluster0'])

    for r in range(1,len(clf_clusters)):
        start_act=len(clf_clusters['new_groups_in_cluster%s' %str(r)])
        end=start_acum+start_act
        ind_lists.append([i for i in range(start_acum,end)])
        start_acum=start_acum+start_act

    st=clf_clusters['new_groups_in_cluster0'] 
    for l in range(1,len(clf_clusters)):
        st.update(dict(zip(ind_lists[l-1], list(clf_clusters['new_groups_in_cluster%s' %str(l)].values()))))

    if len(st)!=k1:
        raise Exception('Error Concatenating into K1 groups')
        
    return (st)

"""# **Algorithm**"""
print('Starting Running AdaTC Iterations Code')

# AdaTC First It.
def adatc_first_it(GeoC1,T_Matrix,TC1,ro1,max_iter_gc,max_iter_tc,k1,k2,all_data_1,all_data_2):
    diss_tot_it=0
    ###########################################################
    ##Geo-Clust
    clf1=GeoC1(k1=k1,max_iter=max_iter_gc,ro1=ro1)
    clf1.fit(all_data_1)
    diss_tot_it=diss_tot_it+sum(clf1.diss_tot)
    ###########################################################
    ##T-Matrices
    #tm1=T_Matrix(clf1,save_in_it1=True,save_in_itN=False,it=1)
    tm1=T_Matrix(clf1,save_in_it1=False,save_in_itN=False,it=1)
    tm1.fit(all_data_2)
    ###########################################################
    ##T-Clust 
    tlf1=TC1(k2=k2,max_iter=max_iter_tc,tm=tm1,its=1)
    tlf1.fit(all_data_1)
    diss_tot_it=diss_tot_it+sum(tlf1.diss_tot)

    return (clf1,tm1,tlf1,diss_tot_it)

# AdaTC Its. 2,...,N-1
def adatc_int_its(GeoC1,T_Matrix,TC1,ro1,max_iter_gc,max_iter_tc,k1,k2,N,all_data_1,all_data_2,gc1,tc1,its):
    diss_tot_it=0
    ###########################################################
    ##Geo-Clust
    nr_groups=new_nr_groups(gc1,tc1,n=len(all_data_1),its=its)
    clf_medoids=[]
    clf_clusters={}
    for j in range(0,tc1.k2):
        clf_clusters['new_groups_in_cluster' + str(j)] = {}

    gcN={}
    diss_tot_gc_it=0
    for cluster in range(0,len(tc1.classes)): 
        print('Starting Geo-Cluster for each T-Cluster of previous iteration group: for TC class %s' %str(cluster))
        if nr_groups[cluster]==1:
            print('Division In 1 Group')
            clf_clusters['new_groups_in_cluster%s' %str(cluster)]={0:tc1.classes[cluster]}
            clf_medoids.append([tc1.medoids[cluster]])
        else:
            data_cluster=stations_18['citibike_station_id'].loc[stations_18['citibike_station_id'].isin(tc1.classes[cluster])]
            clf_cluster=GeoC1(k1=nr_groups[cluster],max_iter=max_iter_gc,ro1=ro1)
            clf_cluster.fit(data_cluster)
            diss_tot_gc_it=diss_tot_gc_it+sum(clf_cluster.diss_tot)
            clf_clusters['new_groups_in_cluster%s' %str(cluster)]=clf_cluster.classes
            clf_medoids.append(clf_cluster.medoids)

    if its==2:
        gc_result=concatenate_into_k1_groups(gc1.k1,clf_clusters)
        gcN["k1"]=gc1.k1
    else:
        gc_result=concatenate_into_k1_groups(gc1['k1'],clf_clusters)
        gcN["k1"]=gc1["k1"]
        
    gcN["new_clusters"]=gc_result
    gcN["new_medoids"]=[item for sublist in clf_medoids for item in sublist]
    diss_tot_it=diss_tot_it+diss_tot_gc_it
    ###########################################################
    ##T-Matrices
    tm_cluster=T_Matrix(gcN,save_in_it1=False,save_in_itN=False,it=its)
    tm_cluster.fit(all_data_2)
    ###########################################################
    ##T-Clust 
    tlf_cluster=TC1(k2=k2,max_iter=max_iter_tc,tm=tm_cluster,its=its)
    tlf_cluster.fit(all_data_1)
    diss_tot_it=diss_tot_it+sum(tlf_cluster.diss_tot)
    ###########################################################
    return (gcN,tm_cluster,tlf_cluster,diss_tot_it) 

#outN=adatc_itN(GeoC1,T_Matrix,TC1,gc1,tc1,its)
#gcN=outN[0] #output geo-clust it.N
#tmN=outN[1] #output t-matrices it.N
#tcN=outN[2] #output t-clust it.N

# AdaTC Last It.
def adatc_last_it(GeoC1,T_Matrix,TC1,ro1,max_iter_gc,k1,N,all_data_1,gc1,tc1,its):
    diss_tot_it=0
    ###########################################################
    ##Geo-Clust
    nr_groups=new_nr_groups(gc1,tc1,n=len(all_data_1),its=its)
    clf_medoids=[]
    clf_clusters={}
    for j in range(0,tc1.k2):
        clf_clusters['new_groups_in_cluster' + str(j)] = {}

    gcN={}
    diss_tot_gc_it=0
    for cluster in range(0,len(tc1.classes)): 
        print('Starting Geo-Cluster for each T-Cluster of previous iteration group: for TC class %s' %str(cluster))
        if nr_groups[cluster]==1:
            print('Division In 1 Group')
            clf_clusters['new_groups_in_cluster%s' %str(cluster)]={0:tc1.classes[cluster]}
            clf_medoids.append([tc1.medoids[cluster]])
        else:
            data_cluster=stations_18['citibike_station_id'].loc[stations_18['citibike_station_id'].isin(tc1.classes[cluster])]
            clf_cluster=GeoC1(k1=nr_groups[cluster],max_iter=max_iter_gc,ro1=ro1)
            clf_cluster.fit(data_cluster)
            diss_tot_gc_it=diss_tot_gc_it+sum(clf_cluster.diss_tot)
            clf_clusters['new_groups_in_cluster%s' %str(cluster)]=clf_cluster.classes
            clf_medoids.append(clf_cluster.medoids)

    gc_result=concatenate_into_k1_groups(gc1['k1'],clf_clusters)
    gcN["k1"]=gc1["k1"]   
    gcN["new_clusters"]=gc_result
    gcN["new_medoids"]=[item for sublist in clf_medoids for item in sublist]
    diss_tot_it=diss_tot_it+diss_tot_gc_it
    ###########################################################
    return (gcN,diss_tot_it)

# AdaTC all iterations
def adatc_all_its(N_val,k1,ro1,k2): 
    ##########################################################
    #convergence parameters
    ###########################################################
    max_iter_gc=10
    max_iter_tc=10
    ###########################################################
    all_data_1=stations_18['citibike_station_id']
    all_data_2=stations_18['distances_id']
    ###########################################################
    iters=[]
    diss_tot=[]
    #####################################################################################################
    #First It
    ########################################
    print('############################ Starting ADATC Iteration 1 ############################')
    out=adatc_first_it(GeoC1,T_Matrix,TC1,ro1,max_iter_gc,max_iter_tc,k1,k2,all_data_1,all_data_2)
    gc1=out[0] #output geo-clust it.1
    tm1=out[1] #output t-matrices it.1
    tc1=out[2] #output t-clust it.1
    iters.append(1)
    diss_tot.append(out[3])
    #####################################################################################################
    #Following Iters (2,...,N-1)
    #########################################
    for its in range(2,N_val):
        print('############################ Starting ADATC Iteration %s ############################' %str(its))
        outN=adatc_int_its(GeoC1,T_Matrix,TC1,ro1,max_iter_gc,max_iter_tc,k1,k2,N_val,all_data_1,all_data_2,gc1,tc1,its)
        gcN=outN[0] #output geo-clust it==its
        tmN=outN[1] #output t-matrices it==its
        tcN=outN[2] #output t-clust it==its

        iters.append(its)
        diss_tot.append(outN[3])
        ####################################################################
        if its==2:
            previous=gc1.medoids
        else:
            previous=gc1['new_medoids']
        ##convergence running N times
        optimal = True
        curr = gcN['new_medoids']
        #difference in the medoids of two consecutive iterations to declare convergence.
        if sorted(curr) != sorted(previous):
            optimal = False
        ##########
        gc1=gcN
        tc1=tcN
        ##########
        #break out of the main loop if the medoids don't change in two adatc iterations 
        if optimal:
                print("MEDOIDS DOESN'T CHANGE IN TWO CONSECUTIVE AdaTC ITERATIONS - STOP")
                break
    #####################################################################################################
    #Last iter
    #########################################
    print('############################ Reached ADATC Iteration N=%s ############################' %str(N_val))
    last_adatc_gcN=adatc_last_it(GeoC1,T_Matrix,TC1,ro1,max_iter_gc,k1,N_val,all_data_1,gc1,tc1,N_val)
    iters.append(N_val)
    diss_tot.append(last_adatc_gcN[1])
    
    out_adatc=last_adatc_gcN

    return out_adatc
#############################################################################################################################
"""# Testing Parameter Combination"""
#Metrics
#####################################################################################################
#Average Inner Cluster Distance and Checkout
#####################################################################################################
def inner_measures(out_adatc):

    metrics={}
    for k in range(out_adatc['k1']):
        metrics[k]= []

    for cl in range(0,len(out_adatc['new_clusters'])):
       # print('Starting class %s' %str(cl))
        i=1
        gd_cum=0
        co_diss_cum=0
        for point in out_adatc['new_clusters'][cl][:-1]:
            for point_adv in out_adatc['new_clusters'][cl][i:]:
                i1=stations_18['distances_id'].loc[stations_18['citibike_station_id']==point].values[0]
                i2=stations_18['distances_id'].loc[stations_18['citibike_station_id']==point_adv].values[0]
                #########################################
                ##geographical distance
                dist=np.array(df_distances)[i1-1,i2-1]
                gd_act=dist
                gd_cum=gd_cum+gd_act
                #########################################
                ##check_out diff
                U1=ast.literal_eval(df_checkout_pattern_all['Checkout_Pattern'].loc[df_checkout_pattern_all['Stations_CB_id']==point].values[0])
                U2=ast.literal_eval(df_checkout_pattern_all['Checkout_Pattern'].loc[df_checkout_pattern_all['Stations_CB_id']==point_adv].values[0])
                co_diss_act=LA.norm(np.subtract(U1,U2),ord=2)
                co_diss_cum=co_diss_cum+co_diss_act
                #########################################
            i=i+1
        ## average measures
        n=(len(out_adatc['new_clusters'][cl])*(len(out_adatc['new_clusters'][cl])-1))/2

        if n==0: ##when exists a cluster with only 1 station
            metrics[cl]=('NA','NA')
                
        else:
            gd_avg=gd_cum/n
            co_diss_avg=co_diss_cum/n
            metrics[cl]=(round(gd_avg,3),round(co_diss_avg,3))

    return metrics #(gd_avg,co_diss_avg)

#####################################################################################################
#Average Inter Cluster Distance and Checkout
#####################################################################################################
def inter_measures(out_adatc):
    i=1
    gd_cum=0
    co_diss_cum=0

    for l in out_adatc['new_medoids']:
        for m in out_adatc['new_medoids'][i:]:
            i1=stations_18['distances_id'].loc[stations_18['citibike_station_id']==l].values[0]
            i2=stations_18['distances_id'].loc[stations_18['citibike_station_id']==m].values[0]
            #########################################
            ##geographical distance
            dist=np.array(df_distances)[i1-1,i2-1]
            gd_act=dist
            gd_cum=gd_cum+gd_act
            #########################################
            ##check_out diff
            U1=ast.literal_eval(df_checkout_pattern_all['Checkout_Pattern'].loc[df_checkout_pattern_all['Stations_CB_id']==l].values[0])
            U2=ast.literal_eval(df_checkout_pattern_all['Checkout_Pattern'].loc[df_checkout_pattern_all['Stations_CB_id']==m].values[0])
            co_diss_act=LA.norm(np.subtract(U1,U2),ord=2)
            co_diss_cum=co_diss_cum+co_diss_act    
        i=i+1
        ## average measures
        n=(len(out_adatc['new_medoids'])*(len(out_adatc['new_medoids'])-1))/2

        if n==0: ##when exists a cluster with only 1 station
            value=('NA','NA')
                
        else:
            gd_avg=gd_cum/n
            co_diss_avg=co_diss_cum/n
            value=(round(gd_avg,3),round(co_diss_avg,3))

    return value #(gd_avg,co_diss_avg)

#############################################################################################################################
##Testing Some Intrinsic Parameters Combination#
k1_val=50
k2_val=10
ro1_val=5.5
print('Starting ADATC for Parameters Combination %s' %str(combi))
adatc_comb=adatc_all_its(N_val=35,k1=k1_val,ro1=ro1_val,k2=k2_val)
print('Finished ADATC for Parameters Combination')

print('Starting to Calculate Inner and Inter Metrics for Parameters Combination')
metrics_inner=inner_measures(adatc_comb[0])
metrics_inter=inter_measures(adatc_comb[0])
print('DONE')

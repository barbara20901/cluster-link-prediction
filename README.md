# cluster-link-prediction
This repository provides some of the code implemented and the data used for the work proposed in "A Cluster-Based Trip Prediction Graph Neural Network Model for Bike Sharing Systems".

The datasets used can be found in:
* CitiBike Historical Trips - https://ride.citibikenyc.com/system-data
* Weather - https://mesonet.agron.iastate.edu/request/download.phtml?network=NY_ASOS
* Station Status - https://www.theopenbus.com/raw-data.html
* Geographical Distances - https://openrouteservice.org/ 

All additional data generated to run the AdaTC clustering algorithm is in the **_Data_** folder (the geographical distances extracted from the ORS API are also available). The code is also available in the **_Code_** folder. 

## **_Data_** Folder
Contains:
* _pairwise_distances.csv.gz_ - matrix of distances between stations extracted from ORS API according to the regular cycling profile (*Note*: the stations are indexed in rows and columns by the corresponding citibike id and the matrix is not symmetric - symmetry can be obtained by considering the mean between the two paths directions)
* _stations_18.csv.gz_ - citibike id, name, latitude and longitude of stations in 2018 
* _df_checkout_pattern_all.csv.gz_ - dataframe with the tuple of 5 elements corresponding to the check-out patterns in each of the 5 time slots and for each station


The implementation considers the following time slots: 
![time_slots_new](https://user-images.githubusercontent.com/74977129/141164706-1bc9de6c-b4fa-4f90-8cbf-594c0de2a773.JPG)



## **_Code_** Folder
Contains: 
*  _adatc_run.py_ - code for running adatc algorithm for some intrinsic parameters combination 

import umap
import pickle
import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
import multiprocessing
import sciris as sc
import pandas as pd
import logging
from itertools import combinations

__author__ = 'Arijeet Dutta'
__email__ = "a.dutta.2@bham.ac.uk"


default_params = dict(
    embedding_dict = dict(min_dist=0.0, n_components=3, n_neighbors=20),
    clustering_dict = dict(linkage='ward',  n_clusters=30, n_neighbors=40)
)


class IscaRegimes():



    def __init__(self,ds_budget,
                 nc=4,nens=10,min_dist=0.0,n_neighbors=20,
                 preprocessor='StandardScaler',backend='mpi',
                 read_relabelled_clusters=False,
                ):

        """
        ds_budget: netcdf file that has all the budget terms
        backends: 'use_pool', 'sciris', 'mpi'
        quantile_cutoff or percent_cutoff regularises dom_balance - entropy thresholds to regularise dominant balance identification
        quantile_cutoff = 1 means NO regularisation
        percent_cutoff is used if quantile_cutoff is None
        """


        try:
            from mpi4py import MPI
            MPI_AVAILABLE = True
            print("[INFO] MPI is available.")
        except ImportError:
            MPI_AVAILABLE = False
            print("[INFO] MPI is NOT available. Using 'sciris' instead")
            backend = 'sciris'


        self.params = copy.deepcopy(default_params)
        # self.params.update(params if params is not None else {})
        self.verbose = True
        self.budget = ds_budget
        self.read_relabelled_clusters = read_relabelled_clusters
        self.preprocessor = preprocessor
        self.min_dist = min_dist
        self.n_neighbors = n_neighbors
        self.nc = nc
        self.ncpus = multiprocessing.cpu_count()
        self.nens = nens
        self.backend = backend



    def make_features(self):

        budget = self.budget

        variables_list = ['dudt','fvR','fvD','eddies','uD_duDdx','uD_duRdx','uR_duDdx','uR_duRdx','vD_duDdy','vD_duRdy','vR_duDdy','vR_duRdy','w_duDdp','w_duRdp','dphidx']
        lst = []
        for var_name in variables_list:
        
            # print(dat[var_name].shape)
            ar = np.asanyarray(budget[var_name]).flatten()
            lst.append(ar)
        
        
        features = np.vstack(lst).T
        df = pd.DataFrame(features)
        # df.columns = ['t'+str(i) for i in np.arange(1,17)]
        df.columns = variables_list        

        self.feature = df


    def preprocess(self):

        if self.preprocessor == 'MinMaxScalaer':
            scaler = MinMaxScalaer()
        elif self.preprocessor == 'StandardScaler':
            scaler = StandardScaler()

        return scaler.fit_transform(self.feature)

    def _embedding(self,dum=None):
        """
        Helper to get embedding
        """        
    
        X = self.preprocess()
        return umap.UMAP(**self.params['embedding_dict']).fit_transform(X)

    def get_embedding(self):
        """
        Run _embedding() nens times
        """


        if self.backend == 'sciris':
            print('USING SCIRIS..')
            res = sc.parallelize(self._embedding, list(range(0, self.nens)), ncpus = self.ncpus, progress=True)
            self.embedding = np.asarray(res)

            return self.embedding

        elif self.backend == 'mpi':

            """
            NOT TESTED YET!
            """
            
            print('USING MPI..')
            if not MPI_AVAILABLE:
                raise ImportError("mpi4py is not available, can't use backend='mpi'")

            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()

            work_per_rank = [i for i in range(self.nens) if i % size == rank]
            results = [self._embedding(i) for i in work_per_rank]

            all_results = comm.gather(results, root=0)
            if rank == 0:
                res = [emb for sublist in all_results for emb in sublist]
                self.embedding = np.asarray(res)
                return self.embedding
            else:
                return None
        else:
            raise ValueError(f"Unknown backend {self.backend}")

               

    def get_sorted_clusters(self,df_umap,hclust_neighbors=40):

        """
        Written by Maike Sonnewald
        """

        # print('clustering..')
        knn_graph = kneighbors_graph(df_umap, n_neighbors=hclust_neighbors, include_self=False)
        model = AgglomerativeClustering(linkage='ward', connectivity=knn_graph,n_clusters=self.nc)    
        clusters = model.fit_predict(df_umap)

        # print('sorting..')
        
        # number of clusters (also the same as the label name in the agglomerated cluster dict)
        n_clusters = np.max(clusters)+1
        #  create a histogram of the different clusters
        hist,_ = np.histogram(clusters, np.arange(n_clusters+1))
        # clusters sorted by size (largest to smallest)
        sorted_clusters= np.argsort(hist)[::-1]
        # assign new labels where labels 0,...,k go in decreasing member size 
        new_labels = np.empty(clusters.shape)
        new_labels.fill(np.nan)
        for new_label, old_label in enumerate(sorted_clusters):
            new_labels[clusters == old_label] = new_label

        print('realisation finished')

        return new_labels


    def get_clusters(self,dummy_var=None):
        """
        Helper to cluster
        """

        df = self.preprocess()
        df_umap = umap.UMAP(min_dist=self.min_dist, n_components=3, n_neighbors=self.n_neighbors, metric='euclidean', init = 'spectral').fit_transform(df)
        clusters = self.get_sorted_clusters(df_umap,)

        # self.set_embedding = df_umap
        # self.set_clusters = n_clusters

        return df_umap,clusters

    def get_nemi_pack(self):
        """
        clusters all embeddings
        """
        # https://stackoverflow.com/questions/21584109/how-to-parallelize-a-for-in-python-inside-a-class

        if self.backend == 'use_pool':
            pool = multiprocessing.Pool(processes=self.ncpus) 
            results = pool.map(self.get_clusters, [self,1] * self.nens)
            
        elif self.backend == 'sciris':
            results = sc.parallelize(self.get_clusters, list(range(0, self.nens)), ncpus = self.ncpus, progress=True)

        else:
            raise ValueError(f"Unknown backend {self.backend}")            
                     
        embeddings = []
        ensembles = []

        for i, array in enumerate(results):
            embeddings.append( array[0])
            ensembles.append( array[1])

        self.ensembles = np.vstack(ensembles)    
        self.embeddings = np.stack(embeddings)

        return self.ensembles, self.embeddings

    def relabel(self,base_id, max_clusters=None, ):

        """
        Written by Maike Sonnewald
        ensembles: nemi_pack - should have dimension (n_ens,npts)
        """
        # print(base_id)
        
        # self.base_id = base_id
        base_labels = self.ensembles[base_id]
        compare_ids = [i for i in range(self.nens)]
        compare_ids.pop(base_id)

        num_clusters = int(np.max(base_labels) + 1)


        # if not pre-set, set max number of clusters to total number of clusters in the base
        if max_clusters is None:
            max_clusters = num_clusters

        sortedOverlap=np.zeros((len(compare_ids)+1, max_clusters, base_labels.shape[0]))*np.nan

        # print(num_clusters, max_clusters)
        summaryStats=np.zeros((num_clusters, max_clusters))

        # compile sorted cluster data
        # TODO: add assert statement to make sure that the clusters have been sorted?


        # dataVector=[nemi.clusters for id, nemi in enumerate(self.nemi_pack) if id != base_id]
        dataVector=[self.ensembles[id] for id, nemi in enumerate(self.ensembles) if id != base_id]

        # loop over ensemble members, not including the base member
        for compare_cnt, compare_id in enumerate(compare_ids):
            # grab clusters of ensemble member
            compare_labels= dataVector[compare_cnt]

            # go through each cluster in the base and assess the percentage overlap
            # for every cluster in the ensemble member (overlap / total coverage area) 
            for c1 in range(max_clusters): 
                # Initialize dummy array to mark location of the cluster for the base member
                data1_M = np.zeros(base_labels.shape, dtype=int)
                # mark where the considered cluster is in the member that is being used as the baseline
                data1_M[np.where(c1==base_labels)] = 1 
                # # Count numer of entries [Why?] 
                summaryStats[0, c1]=np.sum(data1_M) 

                # go through each cluster
                # k = 0
                for c2 in range(num_clusters):
                    # Initialize dummy array to mark where the cluster is in the comparison member
                    data2_M = np.zeros(base_labels.shape, dtype=int) 

                    # mark where the considered cluster is in the member that is being used as the comparison
                    data2_M[np.where(c2==compare_labels)] = 1    

                    # Sum of flags where the two datasets of that cluster are both present
                    num_overlap=np.sum(data1_M*data2_M)       

                    #Sum of where they overlap
                    num_total=np.sum(data1_M | data2_M)       

                    #Collect the number that is largest of k and the num_overlap/num_total
                    # k = max(k, num_overlap / num_total)       
                    summaryStats[c2, c1]=(num_overlap / num_total)*100 # Add percentage of coverage

                #Filled in 'summaryStatistics' matrix results of percentage overlaps
            # print(tabulate(summaryStats, tablefmt="grid", floatfmt=".8f"))
            usedClusters = set() # Used to mak sure clusters don't get selected twice
            #Clusters are already sorted by size
            
            sortedOverlapForOneCluster=np.zeros(base_labels.shape, dtype=int)*np.nan
            # go through clusters from (biggest to smallest since they are sorted)
            for c1 in range(max_clusters):  
                sortedOverlapForOneCluster=np.zeros(base_labels.shape, dtype=int)*np.nan
                #print('cluster number ', c1, summaryStats.shape, summaryStats[1:,c1-1].shape)

                # find biggest cluster in first column, making sure it has not been used
                sortedClusters = np.argsort(summaryStats[:, c1])[::-1]
                biggestCluster = [ele for ele in sortedClusters if ele not in usedClusters][0]
                # print(c1,biggestCluster)

                # record it for later
                usedClusters.add(biggestCluster)

                # Initialize dummy array
                data2_M = np.zeros(base_labels.shape, dtype=int)

                # Select which country is being assessed
                data2_M[np.where(biggestCluster == compare_labels)]=1 # Select cluster being assessed

                sortedOverlapForOneCluster[np.where(data2_M==1)]=1
                sortedOverlap[compare_id, c1, :] = sortedOverlapForOneCluster

        # fill in the base entry in the sorted overlap
        for c1 in range(max_clusters):  
            sortedOverlap[base_id, c1, :] = 1 * (base_labels == c1)

        return sortedOverlap
    
    def _entropy(self,row,i=0):

        """
        Calculates entropy 
        """

        data = row['counts']
        L = sum(data) # ensemble size
        n = len(data) # number counts

        # print('ensemble size = '+str(L),'nclust = '+str(n))
        
        if n != 1:
            ress = 0
            for i in range(n):
                ress =  ress + (data[i]/L * np.log2(data[i]/L))
                                
        else:
                                
            ress = (data[i]/L * np.log2(data[i]/L))
        
        return ress*(-1)


# -------------------------------------------------------------------------------------
    """
    Experimental: multiprocessing.pool to speed up?!
    """

    def entropy_for_bid(self,bid):

        """
        Calculate relabelling and entropy for a single base_id
        """
        ov = self.relabel(base_id=bid) # calculates relabelled clusters for a given base_id
        ov = np.nan_to_num(ov)
        df = pd.DataFrame(np.argmax(ov,axis=1)).T
        df = df.astype('int64')
        df_c = pd.DataFrame(
            df.stack().groupby(level=0).apply(lambda x: np.unique(x, return_inverse=True, return_counts=True)[2])
        )
        df_c.columns = ['counts']
        ent =  df_c.apply(self._entropy, axis=1) # entropy
        EntMax = np.log2(self.nc)

        return np.array((ent * 100)/EntMax)
        # self.count = df_c
        # self.ov = ov   
    
    def get_entropy(self):

        """
        Use multiplrocess to apply entropy_for_bid() for different baselabel_ids     
        """
        bids = list(range(self.nens))

        if self.backend == 'use_pool':
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                results = pool.map(self.entropy_for_bid, bids)

        elif self.backend == 'sciris':
            results = sc.parallelize(self.entropy_for_bid, bids, ncpus = self.ncpus, progress=True)
        
        self.nemi_entropy = np.array(results) # has entropy for all base label ids

        nemi_ent = self.nemi_entropy
        avg_ent = nemi_ent.mean(axis=1) # average over all data points
        self.min_ent_base_id = np.argmin(avg_ent) # base_id that gives least entropy


        minEnt = self.entropy_for_bid(self.min_ent_base_id) # entropy corresponding to the selected base_id

        print(f'[INFO] base_id with least average entropy {self.min_ent_base_id}')
        print(f'[INFO] Minimum entropy in % {np.round(minEnt.mean(),2)}')

        return minEnt
        

# -------------------------------------------------------------------------------------
    
    def get_overlap(self):

        """
        Calculates overlap from relabeled ensemble members
        Output array shape: (nens,nens)
        For each member, nens number of pairs, e.g., for base_id 0 with nens = 4, pairs are - 0 and 1, 0 and 2, 0 and 3.
        Similarly for base_id 1, pairs are - 1 and 0, 1 and 2, 1 and 3.
        overlap with itself 100% but used as nan in the output array
        """

        ovlp = np.zeros((self.nens,self.nens)) * np.nan
        array = self.ov # (ens,ens,nc,npts)
        for bid in range(self.nens):  
            for j in range(self.nens):
                if bid==j: # overlap with itself
                    continue
                    
                count = 0
                for i in range(self.nc):

                    common_ones = np.all(array[bid,[bid,j],i,:] == 1, axis=0) # overlap between 
                    count = count + np.sum(common_ones)
            
                count = count/self.npts
                count *= 100
                ovlp[bid,j] = count
            
        return ovlp

    def export_nemi_clusters(self):
        """
        Export nemi_clusters considering maximum agreement across ensembles 
        """

        min_ent_base_id = self.min_ent_base_id # get the base_id that gives least avg entropy
        print(f'base_id that returns least average entropy {min_ent_base_id}')
        # optimum clustering
        opt_cluster = self.relabel(base_id=min_ent_base_id) # calculates relabelled clusters for the selected base_id
        # aggregate overlap
        aggOverlaps = np.nansum(opt_cluster,axis=0)
        # majority vote
        voteOverlaps = np.argmax(aggOverlaps,axis=0)     

        #  EXPORT TO NETCDF
        ds = self.budget 
        
        if 'xofyear' in ds.coords:
            ds = ds.rename(dict(xofyear='time'))
            if ds.time.shape[0] == 73:
                ds['time'] = pd.date_range('2021-01-01','2021-12-31',freq='5D')
            else:
                ds['time'] = pd.date_range('2021-01-01','2021-12-31',freq='5D')[:-1]        
    

        # RESHAPE TO XARRAY

        if len(ds.coords)==3:
            nemi_labels = IscaRegimes.to_xarray(voteOverlaps,time=ds.time,lat=ds.lat,lon=ds.lon)
            
        else:
            nemi_labels = IscaRegimes.to_xarray(voteOverlaps,lat=ds.lat,lon=ds.lon)

        return nemi_labels


        
    def _mode(*args, **kwargs):
        vals = scipy.stats.mode(*args, **kwargs)
        # only return the mode (discard the count)
        return vals[0].squeeze()
        
    def mode(obj, dim=None):
        # note: apply always moves core dimensions to the end
        # usually axis is simply -1 but scipy's mode function doesn't seem to like that
        # this means that this version will only work for DataArray's (not Datasets)
        assert isinstance(obj, xr.DataArray)
        axis = obj.ndim - 1
        return xr.apply_ufunc(_mode, obj,
                              input_core_dims=[[dim]],
                              kwargs={'axis': axis,
                                      'nan_policy':'propagate'})
        

    @staticmethod
    def to_xarray(da_numpy,lat,lon,time=None):

        """
        numpy to xarray
        """

        import numpy as np
        import xarray as xr

        nlat = lat.shape[0]
        nlon = lon.shape[0]
        
        if time==None:
            
            da_reshaped = np.reshape(da_numpy, [nlat,nlon])
            datarray = xr.DataArray(da_reshaped,
                                coords={'lat': lat,'lon':lon})

        else:
            
            ntim = time.shape[0]
            da_reshaped = np.reshape(da_numpy, [ntim,nlat,nlon])
            datarray = xr.DataArray(da_reshaped,
                                coords={'time': time,'lat': lat,'lon':lon})

        return datarray

        
# -------------------------------------- DOMINANT BALANCE IDENTIFICATION ------------------------------
# HELPER FUNCTIONS taken from original repo: https://github.com/bekaiser/dominant_balance/blob/main/utils.py

    """
    Mostly copied from the original code!
    """

    @staticmethod
    def _get_optimal_balance_from_score( score, feature_mean, balance_combinations ):

        import numpy as np
        from itertools import combinations

        
        maxabsval = np.amax(np.abs(feature_mean))
        loc_maxabsval = np.argwhere(np.abs(feature_mean)==maxabsval)
        #if np.any(np.isnan(score_max)) == True:
        score_max = np.nanmax(score)
        loc_matches = np.argwhere(score==np.amax(score))
        Nmatch = np.shape(np.argwhere(score==np.amax(score)))[0]
        #print('score_max  = ',score_max )
        #print('Nmatch = ',Nmatch)
        #print('balance_max = ',balance_max)
        if Nmatch > 1:
            # if there is more than one maximum score, find the balance that
            # includes the element with the maximum score.
            for i in range(0,Nmatch):
                balance = (balance_combinations[loc_matches[i],:])[0]
                if balance[loc_maxabsval] == 1.:
                    balance_max = balance_combinations[loc_matches[i],:]
        elif Nmatch == 1:
            #print('here !')
            loc_max = (np.argwhere(score==np.amax(score))[:,0])[0]
            #print('loc_max = ',loc_max)
            balance_max = balance_combinations[loc_max,:]
            #print('balance_max = ',balance_max)
        elif Nmatch == 0:
            print('\n  ERROR: no maximum score found')
            print(' ')
            nf = np.shape(balance_combinations)[1]
            balance_max = np.ones([nf])
            score_max = nf / (2.*(nf-1.))
            #print('score = ',score)
            #print('np.amax(score) = ',np.amax(score))
        #print('balance_max = ',balance_max)
        return balance_max, score_max

        
    @staticmethod
    def _get_m_score( balance, features, bias = 'unbiased', penalise=True):

        """
        Calculates LMS
        """

        import numpy as np

        # features need to be raw, i.e. not standardized.
        # print(np.shape(np.shape(features))[0])


        if bias == 'unbiased':
    
            if np.shape(np.shape(features))[0] == 1: # vector (single data sample)
        
                if np.sum(balance) == features.shape[0]: # all ones
                    #print(' balance = ', balance)
                    score = 0.
                    score_95 = 0.
                elif np.sum(balance) == 0: # all zeros
                    #print(' balance 0 = ', balance)
                    score = 0.
                    score_95 = 0.
                else:
                    #print('\n  features = ',features)
                    #print('  balance = ', balance)
                    if np.amin(np.abs(features)) == 0.:
                        nonzero_features = np.abs(features[np.nonzero(features)])
                        np.abs(features)/np.amin(np.abs(nonzero_features))
                        #features_normalized = np.abs(features)/np.array([1e-20])
                    else:
                        features_normalized = np.abs(features)/np.amin(np.abs(features))
                    #print(' features_normalized = ',features_normalized)
                    select,remain = IscaRegimes._get_sets( balance, features_normalized  )
                    select_max,select_min,select_imax,select_imin = IscaRegimes._find_maxmin( select )
                    remain_max,remain_min,remain_imax,remain_imin = IscaRegimes._find_maxmin( remain )
    
                    # print('selected: ',select)
                    # print('selected max-min : ',select_max, select_min)
                    # print('remaining: ',remain)
                    # print('remaining max-min : ',remain_max, remain_min)
                    
                    if select_min <= remain_max:
                        G = 0.
                    else:
                        G = ( np.log(select_min - remain_max) ) / ( np.log(select_min+remain_max) )
        
                    #print('  G = ',G)
                    if G < 0.:
                        G = 0.
        
                    if select_min == select_max:
                        P = 0.
                    else:
                        P =  np.log10(select_max) - np.log10(select_min)
                    if penalise:
                        # print('Penalising for large differences in magnitudes..')
                        score = G / (1. + P )
                    else:
                        score = G 
                    score_95 = 0. #np.inf # fix this! propogate error from the standard error of the feature mean...
                    #print('  score = ',score)
    
            return score        
    
        else:
                
            if sum(balance) == 0.:
                balance = np.ones(np.shape(balance))
        
            # 1) get the differences for the full set of features
            diffs_full,id_full = get_diff_vec( np.ones(np.shape(balance)) , features )
            #print('diffs_full = ',diffs_full)
        
            # 2) get the differences for the reduced set of features
            diffs_red,id_red = get_diff_vec( balance , features )
            #print('diffs_red = ',diffs_red)
        
            # if any difference is exactly zero:
            if np.any(diffs_red==0.) == True:
                loc0 = np.argwhere(diffs_red==0.)[:,0]
                diffs_red[loc0] = (1e-100)*np.ones(np.shape(loc0)[0])
        
            if np.any(diffs_full==0.) == True:
                loc0 = np.argwhere(diffs_full==0.)[:,0]
                diffs_full[loc0] = (1e-100)*np.ones(np.shape(loc0)[0])
        
            sum_score = np.sum(np.log10(diffs_red)) / np.sum(np.log10(diffs_full))
        
            # 5) bias
            Mactive = np.shape(np.nonzero(balance)[0])[0] # number of active features in subset
            Qactive = np.shape(diffs_red)[0] # number of active differences in subset
            if Qactive == 0:
                print('\n\n  ERROR: Nb = 0 \n\n')
            factor = (Qactive+1.)/(2.*Qactive) # (Nb+1)/(2*Nb)
        
            score = sum_score * factor
            closure_score = np.nan
    
            return score
            
    @staticmethod
    def _get_diff_vec( grid_balance, grid_features ):

        import numpy as np

        #print('\n')
        locs = np.nonzero(grid_balance)[0] # grid balance is 0's and 1's for on/off terms
        if np.shape(locs)[0] == 0: # quiescent balance: trivial solution
            locs = np.nonzero(np.ones(np.shape(grid_balance)))[0] # 4 zeros = 4 ones in terms of differences
        abs_balance = np.abs(grid_features[locs]) # absolute values of 'on'  terms
        #print('abs_balance = ',abs_balance)
        N_balance = np.shape(abs_balance)[0] # number of 'on' terms
        max_abs_term = np.amax(np.abs(grid_features)) # maximum term, may or may not be in grid_balance!
        #print('max_abs_term = ',max_abs_term)
    
        max_loc = np.argwhere(abs_balance==max_abs_term)
        #print('abs_balance = ',abs_balance)
        #print('max_loc = ',max_loc)
    
        if np.shape(max_loc)[0] == 0:
            max_loc = np.nan
        else:
            max_loc = max_loc[0]
        diffs = np.zeros([N_balance])
        for ib in range(0,N_balance):
            if ib == max_loc:
                diffs[ib] = np.nan
            else:
                diffs[ib] = abs( max_abs_term - abs_balance[ib] ) /  abs( max_abs_term + abs_balance[ib] )
    
        id = locs[np.argwhere(np.isnan(diffs)-1)[:,0]] # difference indices
        #print('id = ',id)
        diffs = diffs[np.argwhere(np.isnan(diffs)-1)[:,0]] # differences
        #print('diffs = ',diffs)
        if np.shape(diffs)[0] == 0: # if no differences are found
            return np.array([1.]),0
        else:
            return diffs,id



    @staticmethod
    def _find_maxmin( vector ):

        import numpy as np

        
        # find the max/min absolute values
        max = np.amax(np.abs(vector))
        min = np.amin(np.abs(vector))
        imax = np.argwhere(np.abs(vector)==max)[0,0]
        imin = np.argwhere(np.abs(vector)==min)[0,0]
        return max,min,imax,imin

    @staticmethod
    def _get_sets( hypothesis, vector ):

        import numpy as np
        
        # find the absolute values of the selected and remainder subsets
        iselect = np.flatnonzero(hypothesis)
        iremain = np.flatnonzero(hypothesis-1.)
        select = np.abs(vector[iselect])
        remain = np.abs(vector[iremain])
        return select,remain

    @staticmethod
    def _generate_vertices(n):
        import numpy as np
        
        # n is the number of dimensions of the cube (3 for a 3d cube)
        for number_of_ones in range(0, n + 1):
            for location_of_ones in combinations(range(0, n), number_of_ones):
                result = [0] * n
                for location in location_of_ones:
                    result[location] = 1
                yield result    

    
    @staticmethod
    def _generate_balances( nf , bias_flag ):

        import numpy as np
        
        # nf = number of features
        #nf = 3
        nc = int(np.power(2,nf)-1)
        grid_balance = np.zeros([nc,nf])
        i = 0
        for vertex in IscaRegimes._generate_vertices(nf):
            if sum(vertex) == 0.:
                continue
            else:
                grid_balance[i,:] = vertex
                i = i+1
        # remove balances with only one term:
        grid_balance = grid_balance[nf:None,:]
        if bias_flag == 'unbiased':
            # remove full set:
            grid_balance = grid_balance[0:np.shape(grid_balance)[0]-1,:]
        return grid_balance
        

            
# -------------------------------------- HELPER FUNCTIONS ------------------------------
    def kaiser_dom_balance(self,entMin,quantile_cutoff=1.0,percent_cutoff=10,
                           return_all_balances=False,nf=6,bias_flag='unbiased'):

        """
        entMin - returned from .get_entropy()
        """

        ds = self.budget # momentum budget netcdf
        # nf number of budget terms used to find dominant balance
        # qt = self.quantile_cutoff


        if 'xofyear' in ds.coords:
            ds = ds.rename(dict(xofyear='time'))
            if ds.time.shape[0] == 73:
                ds['time'] = pd.date_range('2021-01-01','2021-12-31',freq='5D')
            else:
                ds['time'] = pd.date_range('2021-01-01','2021-12-31',freq='5D')[:-1]        
    

        
        # get nemi labels
        nemi_labels = self.export_nemi_clusters() # xarray

        # RESHAPE TO XARRAY

        if len(ds.coords)==3:
            # nemi_labels = self.to_xarray(nemi_labels,time=ds.time,lat=ds.lat,lon=ds.lon)
            # calculate mode - frequently occuring clusters over a time period ps - pe
            lab = mode(nemi_labels[ps:pe],'time') # shape : (lat,lon); mode over time window specified by pentads ps and pe
            # RESHAPE entropy array TO XARRAY
            da_ent = self.to_xarray(entMin,time=ds.time,lat=ds.lat,lon=ds.lon)
            ent = da_ent[ps:pe].mean('time')   # shape : (lat,lon); entropy over the time windows
            
        else:
            # nemi_labels = self.to_xarray(nemi_labels,lat=ds.lat,lon=ds.lon)
            lab = nemi_labels
            da_ent = self.to_xarray(entMin,lat=ds.lat,lon=ds.lon)
            ent = da_ent
        
        # get the number of clusters (can be less the self.nc!)            
        nc = int(max(np.unique(lab)))+1 # number of clusters 
        
    
        lst = [] # to contain the budget terms
        # print(nc,nf)
        bal = np.zeros((nc,nf)) # to contain the dominant balance
        # print('fewfew')
        
        for i in range(nc):
    
    
            ent_ = ent.where(lab==i) # entropy masked by cluster label
            
            if quantile_cutoff is not None:
                cf = ent_.quantile(quantile_cutoff,dim=('lat','lon',)) # qt'th percentile of entropy; shape : scalar
            else:
                cf = percent_cutoff # if using pecentage cutoff
                
            ent_masked = ent_.where(ent_<cf) # entropy further masked by cutoff value
            mask = ent_masked.notnull() # nonnull entropy - mask


            if len(ds.coords)==3:

                """
                If time is present
                """
        
                e1 = ds.fvD[ps:pe].mean('time').where(mask).mean().to_numpy() # average magnitude of budget term over the region        
                e2 = ds.vD_duRdy[ps:pe].mean('time').where(mask).mean().to_numpy() + ds.w_duRdp[ps:pe].mean('time').where(mask).mean().to_numpy()
                e3 = ds.eddies[ps:pe].mean('time').where(mask).mean().to_numpy()
                e4 = ds.fvR[ps:pe].mean('time').where(mask).mean().to_numpy()
                e5 = ds.uR_duRdx[ps:pe].mean('time').where(mask).mean().to_numpy() + ds.vR_duRdy[ps:pe].mean('time').where(mask).mean().to_numpy()
                e6 = ds.dphidx[ps:pe].mean('time').where(mask).mean().to_numpy()
                rsd = e1+e2+e3+e4+e5+e6
                _ = pd.DataFrame(np.vstack([e1,e2,e3,e4,e5,e6,rsd]))
                lst.append(_)

            else:

                """
                If working with time-mean budget
                """

                e1 = ds.fvD.where(mask).mean().to_numpy() # average magnitude of budget term over the region        
                e2 = ds.vD_duRdy.where(mask).mean().to_numpy() + ds.w_duRdp.where(mask).mean().to_numpy()
                e3 = ds.eddies.where(mask).mean().to_numpy()
                e4 = ds.fvR.where(mask).mean().to_numpy()
                e5 = ds.uR_duRdx.where(mask).mean().to_numpy() + ds.vR_duRdy.where(mask).mean().to_numpy()
                e6 = ds.dphidx.where(mask).mean().to_numpy()
                rsd = e1+e2+e3+e4+e5+e6
                _ = pd.DataFrame(np.vstack([e1,e2,e3,e4,e5,e6,rsd]))
                lst.append(_)
                
    
            # dominant balance identification following Kaiser et al
    
            balances = self._generate_balances(nf=nf,bias_flag=bias_flag)
            ngb = np.shape(balances)[0]
            # print('Number of hypothesis = ',ngb)
            features = np.vstack([e1,e2,e3,e4,e5,e6]).T[0]
    
            # print(features)
        
            M = np.zeros((ngb))
            for mm in range(ngb):
                M[mm] = self._get_m_score(balances[mm,:], features, bias=bias_flag)    
        
            bal[i,:], score = self._get_optimal_balance_from_score(M,features,balances)    
    
            
        df_dominant_balance = pd.DataFrame(bal)
        df_magnitudes = pd.concat(lst, axis = 1).T
        
        if nf==6:
    
            tlst = [
            r'$fv_{d}$',
            r'$-v_{d}\frac{\partial u_{r}}{\partial y} -\omega \frac{\partial u_{r}}{\partial p}$',
            r'$-\delta$',
            r'$fv_{r}$',
            r'$-u_{r}\frac{\partial u_{r}}{\partial x}-v_{r}\frac{\partial u_{r}}{\partial y}$',     
            r'$-\frac{\partial \phi }{\partial x}$',
            'residual'
            ]
            
            df_magnitudes.columns = tlst
            df_dominant_balance.columns = tlst[:-1]
    
    
        if not return_all_balances:
            return df_magnitudes, df_dominant_balance.astype('int')
        else:
            return bal

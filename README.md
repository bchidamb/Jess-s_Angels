# Jess's Angels
## CS 156b

### Team Members
#### Jessica, Bhairav, Connor, Marcus  

## Papers and Articles
https://www.netflixprize.com/assets/GrandPrize2009_BPC_BellKor.pdf --overview of winning solution  
https://www.netflixprize.com/assets/GrandPrize2009_BPC_BigChaos.pdf --comprehensive discussion of each model in winning blend  
https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_cmeans.html
-- Fuzzy K-means clustering

## Packages
http://www.shogun-toolbox.org/examples/latest/examples/multiclass_classifier/knn.html (kNN clustering, C++ with Python hooks)  
http://www.mlpack.org/docs/mlpack-3.0.0/python/cf.html (Collaborative Filtering, C++ with Python hooks)  
https://cran.r-project.org/web/packages/SuperLearner/vignettes/Guide-to-SuperLearner.html (SuperLearner in R, check to download appropriate packages)

## TimeSVD++
citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.379.1951

#### Data processing tasks
For every rating compute and save Bin(t) (all datasets)  
For each user, compile a list of the unique days on which they rated (training only)  
For each user, compute the mean rating date (training only)  
  
#### Notes
30 time bins for function Bin(t), used in changing movie bias over time  
Train using SGD for 30 epochs  
Training time per epoch is roughly double that of SVD++  
  
 _High predictive power_  
Drift in user bias over time  
User per day bias (per user, this models spikes in ratings on a specific day)  
Time dependence of user factors  
  
_Medium predictive power_  
Drift in movie bias over time  
  
_Low predictive power_  
Seasonal periodicity  
Day of the week periodicity   


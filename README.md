PatientMonitoring-DL-python

This projects aims to perform patient monitoring using deep learning in python

Procedure for histogram encoding of time-interest point descriptor data (e.g. tip-hog)

Feature encoding requires in general 2 steps:

Cluster the data based on a subset. This provides the clustering model.
Encode the data using the clustering model obtained above
Detailed steps:

Convert all csv feature files to hdf5: ./blueprint/ICUCockpit/load_split_and_save_hdf5/load_tiphog_file_save_hdf5_uncompressed.py 
to run:./blueprint/ICUCockpit/load_split_and_save_hdf5/run_load_tiphog_file_save_hdf5_uncompressed.lsf

Create a txt file list with all hdf5 files having the absolute path ./blueprint/ICUCockpit/load_split_and_save_hdf5/create_txt_file_for_all_hdf5.sh

Using the file list subsample all data in eg. 1M samples: ./blueprint/ICUCockpit/load_split_and_save_hdf5/load_hdf5_from_list_subsample_save_hdf5_uncompressed.py 
to run:./blueprint/ICUCockpit/load_split_and_save_hdf5/run_load_hdf5_from_list_subsample_save_hdf5_uncompressed.lsf

Cluster subsampled data in eg. 2048 clusters, save model: ./blueprint/ICUCockpit/tip_hog_feature_clustering/load_hdf5_learn_kmeans_codebook_and_save.py 
to run:./blueprint/ICUCockpit/tip_hog_feature_clustering/run_load_hdf5_learn_kmeans_codebook_and_save.lsf

Encode all hdf5 files based on the model ./blueprint/ICUCockpit/tip_hog_feature_encoding/load_hdf5_encode_tip_hog_features_hist_kmeans.py 
to run:./blueprint/ICUCockpit/tip_hog_feature_encoding/run_load_hdf5_encode_tip_hog_features_hist_kmeans.lsf

If necessary for next ML steps group data with a continuous label ./blueprint/ICUCockpit/load_split_and_save_hdf5/load_encoded_hdf5_group_according_to_label.py 
to run:./blueprint/ICUCockpit/load_split_and_save_hdf5/run_load_encoded_hdf5_group_according_to_label.lsf

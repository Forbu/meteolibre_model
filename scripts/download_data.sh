gsutil cp gs://meteofrance-preprocess/h5.zip .
unzip h5.zip
gsutil cp gs://meteofrance-preprocess/index.parquet .
mkdir groundstations_filter
gsutil cp gs://meteofrance-preprocess/total_transformed.parquet groundstations_filter/
gsutil cp gs://meteofrance-preprocess/reprojected_gebco_32630_500m_padded.npy .
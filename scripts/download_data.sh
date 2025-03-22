# radar data
gsutil cp gs://meteofrance-preprocess/h5.zip .
unzip h5.zip


# groudstation data
gsutil cp gs://meteofrance-preprocess/groundstation_npz.zip .
unzip groundstation_npz.zip

gsutil cp gs://meteofrance-preprocess/index.parquet .


# other data
gsutil cp gs://meteofrance-preprocess/reprojected_gebco_32630_500m_padded.npy .
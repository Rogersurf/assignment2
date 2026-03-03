from huggingface_hub import upload_file

# Upload 50k dataset
upload_file(
    path_or_fileobj="patents_50k_green.parquet",
    path_in_repo="patents_50k_green.parquet",
    repo_id="Rogersurf/green_patent_02_dataset",
    repo_type="dataset"
)

# Upload HITL reviewed
upload_file(
    path_or_fileobj="hitl_reviewed.csv",
    path_in_repo="hitl_reviewed.csv",
    repo_id="Rogersurf/green_patent_02_dataset",
    repo_type="dataset"
)
from huggingface_hub import HfApi, upload_folder

api = HfApi()

upload_folder(
    folder_path="models/patentsberta_final",
    repo_id="Rogersurf/green_patent_02",
    repo_type="model"
)
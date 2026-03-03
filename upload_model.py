from huggingface_hub import HfApi, upload_folder

api = HfApi()

upload_folder(
    folder_path="CAMINHO_DA_PASTA_DO_MODELO",
    repo_id="Rogersurf/green_patent_02_model",
    repo_type="model"
)
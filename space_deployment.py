from huggingface_hub import HfApi, notebook_login

# Login
notebook_login()

# Initialize API
api = HfApi()

# Upload files directly
api.upload_file(
    path_or_fileobj="app.py",
    path_in_repo="app.py",
    repo_id="Agrannya/Car_feedback_analyzer",
    repo_type="space"
)

api.upload_file(
    path_or_fileobj="car_rental_feedback_sentiment.csv - Copy (1).csv",
    path_in_repo="c_rental_feedback_sentiment.csvar",
    repo_id="Agrannya/Car_feedback_analyzer",
    repo_type="space"
)

api.upload_file(
    path_or_fileobj="requirements.txt",
    path_in_repo="requirements.txt",
    repo_id="Agrannya/Car_feedback_analyzer",
    repo_type="space"
)
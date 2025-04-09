import kagglehub

# Download latest version
path = kagglehub.dataset_download("itbetyar/movies-small-dataset-for-beginners")

print("Path to dataset files:", path)
from preprocessing.pipeline import run_preprocessing_pipeline

if __name__=="__main__":
    splits,encoder=run_preprocessing_pipeline()
    print("All artifacts saved to artifacts/splits/")
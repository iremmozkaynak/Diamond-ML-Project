import pickle
import pandas as pd

def main():
    #load model
    with open("30-diamond_model_complete.pkl", "rb") as f:
        saved_data = pickle.load(f)
    model = saved_data["model"]

if __name__ == "__main__":
    main()

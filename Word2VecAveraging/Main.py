import pandas as pd
import os
from Utils import PurifyText
HOME_DIR = os.path.expanduser('~')
DATA_PTH = HOME_DIR+'/Google Drive/DataRepo/AmazonReviews/Musical_Instrument/'

if __name__ == '__main__':


    d_name = "reviews_Musical_Instruments.json"
    df = pd.read_json(DATA_PTH+d_name, lines=True)
    df = df.sample(5000,random_state=10)
    print("Data loaded")
    vals = PurifyText(df,'reviewText')

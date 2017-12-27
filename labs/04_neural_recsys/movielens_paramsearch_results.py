import pandas as pd
from pathlib import Path
import json



def load_results_df(folder='results'):
    folder = Path(folder)
    results_dicts = []
    for p in sorted(folder.glob('**/results.json')):
       with p.open('r') as f:
           results_dicts.append(json.load(f))
    return pd.DataFrame.from_dict(results_dicts)


if __name__ == "__main__":
    df = load_results_df().sort_values(by=['test_mae'], ascending=True)
    print(df.head(5))

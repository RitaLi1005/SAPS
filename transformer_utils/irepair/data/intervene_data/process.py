import pandas as pd

# df = pd.read_json("validation-00000-of-00001.json")
# df.to_parquet("validation-00000-of-00001.parquet")
pd.read_parquet("validation-00000-of-00001.parquet")
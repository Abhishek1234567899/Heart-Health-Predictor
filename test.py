'''import os
path="notebook/research.ipynb"

dir,file=os.path.split(path)

os.makedirs(dir,exist_ok=True)

with open(path,'w') as f:
    pass'''

from src.Hearthealthpredictor.pipelines.prediction_pipeline import CustomData

custdataobj=CustomData(63,1,3,145,233,1,0,150,0,2.3,0,0,1)
data=custdataobj.get_data_as_dataframe()

print(data)
import transformers
import os
from transformer_utils.sample_selection.selection.location import top_data_embedding
import time
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('/root/autodl-tmp/mnt/memit/model/all-MiniLM-L6-v2').to('cuda').eval()
seed=45           # Random Seed
trainsize=4400     # Training Set Size
percentage=50     # The Proportion of Data for Repair
alpha=1.00        # Boundary data ratio
per_cluster=int(trainsize * (percentage / 1000))    # cluster = 10
total_samples=int(trainsize * (percentage / 100))
start_time=time.time()                              # time
top_data_embedding=top_data_embedding(model,trainsize=trainsize,valid_size=64,seed=seed,per_cluster=per_cluster,total_samples=total_samples,strategy="SAPS",SAPS_mode="top",alpha=alpha, cache_path=os.path.join("/root/autodl-tmp/mnt/SAPS/transformer_utils/irepair/data", f"top_data_embedding_{seed}_SAPS_top_{alpha}.jsonl"))
print(time.time()-start_time)
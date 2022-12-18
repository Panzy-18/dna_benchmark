from tools import get_config
from models import get_model
import pyfastx

hg38 = pyfastx.Fasta('data/genome/hg38.fa', uppercase=True)
sequence_around_TSS = hg38.fetch("chr1", (169842601, 169883600)) # replace it by any sequence
n_chunk = (len(sequence_around_TSS) - 400 * 2) // 200
chunks = [sequence_around_TSS[i*200: (i+1)*200 + 400] for i in range(n_chunk)]

# step1 generate features
config = get_config('pretrained/track7878/config.json')
config.with_head = False # if True genrate feature_dim == 7878, else feature_dim == 512
model = get_model(config, model_path='pretrained/track7878/model.pt')
features = model.predict(sequences=chunks).logits_or_values.detach()

# step2 predict expression from features
config = get_config('experiment/expression218')
model = get_model(config, model_path='experiment/expression218/model_best.pt')
logp_expression = model.predict(features=features).cpu().numpy()

print(logp_expression)

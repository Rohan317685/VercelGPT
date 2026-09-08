import os 
import torch

folder = "vercel_training"
block_size = 128
batch_size = 32

texts = []
for filename in os.listdir(folder): 
    if filename.endswith((".txt", ".md")):
        path = os.path.join(folder, filename)
        with open(path, "r", encoding="utf-8") as f: 
            texts.append(f.read()) 

text = "\n\n".join(texts) 
print(f"Loaded {len(texts)} files, {len(text)} characters")

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i:c for c, i in stoi.items()}

encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[int(i)] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)

ix = torch.randint(len(data) - block_size - 1, (batch_size,))
x = torch.stack([data[i:i+block_size] for i in ix])
y = torch.stack([data[i+1:i+1+block_size] for i in ix])

print(f"Data shape: {x.shape}")
print(f"Vocab Size: {vocab_size}")
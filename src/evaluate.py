from dataset import df, tokenizer
from train import model, device

k = 88
sample = df.iloc[k]["findings"]
imp = df.iloc[k]["impression"]
inputs = tokenizer(sample, return_tensors="pt", truncation=True, padding=True).to(device)

summary_ids = model.generate(inputs["input_ids"], max_length=128, num_beams=4)
print(sample)
print("Original Impression: ", imp,"\nGenerated Impression:", tokenizer.decode(summary_ids[0], skip_special_tokens=True))

import numpy as np, json, random, requests, os
from utils import PCA, Text2Vec

# Increasing NUM_JOKES will increase the accuracy of the model, but will take longer to train
NUM_JOKES = 10000

if not os.path.exists("reddit_jokes.json"):
    url = "https://raw.githubusercontent.com/taivop/joke-dataset/master/reddit_jokes.json"
    print(f"Downloading jokes from {url}")
    r = requests.get(url)
    with open("reddit_jokes.json", "wb") as f:
        f.write(r.content)

jokes = json.load(open("reddit_jokes.json", "r"))
random.shuffle(jokes)
print(f"Found {len(jokes)} jokes")

jokes = [joke["title"] + " " + joke["body"] for joke in jokes[:NUM_JOKES]]

pca = PCA()
text2vec = Text2Vec()

vectors = []
for i, joke in enumerate(jokes):
    
    vectors.append(text2vec.sentence2vector(joke))

    if i % 100 == 0:
        p = i / len(jokes) * 100
        print(f"Processed {i} jokes ({p:.2f}%)", end="\r")

print("\n")

vectors = np.array(vectors)

pca.fit(vectors)

transformed = pca.transform_multiple(vectors)

print("### STATS ###")
print(f"mean : {transformed.mean()}")
print(f"std : {transformed.std()}")
print(f"min : {transformed.min()}")
print(f"max : {transformed.max()}")
print("#############")

# Save the "model"
with open("model.json", "w") as f:
    json.dump({
        "mean": pca.np2base64(pca.mean.astype(np.float32)),
        "eigenvectors": pca.np2base64(pca.eigenvectors.astype(np.float32)),
        "min": transformed.min().item(),
        "max": transformed.max().item()
    }, f)
from utils import Text2Vec, PCA
import json, argparse

# The model outputs a float, but we want an integer, so we multiply by the RESOLUTION
# A higher resolution means less collisions, but a larger resulting integer.
RESOLUTION = 100000

model = json.load(open("model.json"))
pca = PCA(mean=model["mean"], eigenvectors=model["eigenvectors"])
text2vec = Text2Vec()

parser = argparse.ArgumentParser()
parser.add_argument("joke", help="The joke to convert")
args = parser.parse_args()

vec = text2vec.sentence2vector(args.joke)
num = pca.transform(vec)
norm = (num - model["min"]) / (model["max"] - model["min"])
print(int(norm * RESOLUTION))
import numpy as np, torch, base64, re
from transformers import BertTokenizer, BertModel, logging
logging.set_verbosity_error()

class PCA:
    def __init__(self, n_components=1, mean=None, eigenvectors=None):

        if type(mean) == str:
            mean = self.base64_to_np(mean)
        if type(eigenvectors) == str:
            eigenvectors = self.base64_to_np(eigenvectors)

        self.n_components = n_components
        self.eigenvectors = eigenvectors
        self.mean = mean

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        cov = np.cov(X, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        self.eigenvectors = eigenvectors[:, :self.n_components].real

    def np2base64(self, np_array):
        return base64.b64encode(np_array.astype(np.float32).tobytes()).decode("utf-8")
    
    def base64_to_np(self, base64_string):
        return np.frombuffer(base64.b64decode(base64_string), dtype=np.float32)

    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.eigenvectors)
    
    def transform_multiple(self, X):
        return np.array([self.transform(x) for x in X])

class Text2Vec:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True).eval().to(self.device)
    
    def clean(self, text:str):
        # Cleans text, fixes some quotation marks and punctuation
        text = text.replace("\n", " ")
        text = text.replace("’", "'")
        text = text.replace("“", '"')
        text = text.replace("”", '"')
        text = text.replace("…", "...")
        text = text.replace("—", "--")
        text = text.replace(" ?", "?")
        text = text.replace(" !", "!")
        text = text.replace(" ,", ",")
        text = text.replace(" .", ".")
        text = text.replace(" :", ":")

        # Removes all non-ascii characters
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)

        # Removes all extra spaces
        text = re.sub(' +', ' ', text)

        return text.lower()

    def sentence2vector(self, sentence:str):
        # Converts a sentence to a vector
        sentence = self.clean(sentence)
        marked_text = "[CLS] " + sentence.lower() + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(marked_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        # Remove extra rows eg: (1, 523) -> (1, 512)
        tokens_tensor = tokens_tensor[:, :512]
        segments_tensors = segments_tensors[:, :512]

        with torch.no_grad():
            outputs = self.model(tokens_tensor.to(self.device), segments_tensors.to(self.device))
            hidden_states = outputs[2]
            token_vecs = hidden_states[-2][0]
            sentence_embedding = torch.mean(token_vecs, dim=0)
            return sentence_embedding.cpu().numpy()

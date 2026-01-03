import re
with open('./the-verdict.txt', 'r') as file:
    raw_text = file.read()

def create_vocabulary(raw_text):
    #creates vocabulary dictionary from raw text
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    
    words = sorted(set(preprocessed))
    words.extend(["<|endoftext|>", "<|unk|>"])
    
    vocabulary = {word: id for id, word in enumerate(words)}
    return vocabulary

class SimpleTokenizer:
    def __init__(self, vocabulary):
        self.string_to_id = vocabulary
        self.id_to_string = {id: string for string, id in vocabulary.items()}

    def encode(self, text):
        #encodes text into a list of token IDs
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.string_to_id else "<|unk|>" for item in preprocessed]

        ids = [self.string_to_id[item] for item in preprocessed]
        return ids

    def decode(self, ids):
        #decodes token Ids back to text
        text = ' '.join([self.id_to_string[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

vocabulary = create_vocabulary(raw_text)
tokenizer = SimpleTokenizer(vocabulary)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)
print(tokenizer.encode(text))
print(tokenizer.decode(tokenizer.encode(text)))



from model import *
from data import *
from tokenizer import *


reader = WikiTextReader(100)
instances = list(reader.read("../data/wikitext-103-raw/wiki.train.tokens"))
# generates a vocabulary from the file
vocab = Vocabulary.from_instances(instances)
# creates an embedder, needs the number of items in the vocab
embedding = Embedding(num_embeddings=vocab.get_vocab_size(), embedding_dim=20)
embedder = BasicTextFieldEmbedder(token_embedders={"tokens": embedding})
data_loader = SimpleDataLoader(instances, batch_size=4, vocab=vocab)

model = LanguageModel(
    vocab=vocab,
    embedder=embedder,
    hidden_size=20,
    intermediate_size=50,
    num_attention_heads=1,
)

trainer = GradientDescentTrainer(
    model=model.cuda(),
    data_loader=data_loader,
    num_epochs=5,
    optimizer=torch.optim.Adam(model.parameters()),
)

trainer.train()
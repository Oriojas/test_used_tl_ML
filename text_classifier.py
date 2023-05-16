import torch
import spacy
import torchtext
import numpy as np
from tqdm import tqdm
from torchtext.legacy import data

# spacy.cli.download("es_core_news_sm")
print(f"GPU available: {torch.cuda.is_available()}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

spacy.prefer_gpu()
spacy_es = spacy.load("es_core_news_sm")


def tokenize(text_c):
    return [tok.text for tok in spacy_es.tokenizer(text_c)]


# text = data.Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True)
# is_new = data.LabelField(sequential=False, use_vocab=False)

# fields = {"clean_text": ("text", text), "Y": ("label", is_new)}


# X_train, X_test = data.TabularDataset.splits(path="data/",
#                                              train="df_text_train.csv",
#                                              test="df_text_test.csv",
#                                              format="csv",
#                                              fields=fields)

text = data.Field(tokenize = spacy_es, include_lengths = True)
label = data.LabelField(dtype = torch.float)
fields = [('clean_text', text), ('Y', label)]

X_train, X_test = data.TabularDataset.splits(path='data/',
                                                      train="df_text_train.csv",
                                                      test="df_text_test.csv",
                                                      format="csv",
                                                      fields=fields,
                                            skip_header=True)


# for batch in X_train:
#     print(batch.t)
#     print(batch.i)

MAX_VOCAB_SIZE = 10000

text.build_vocab(X_train, max_size=MAX_VOCAB_SIZE)
label.build_vocab(X_train)
print(vars(X_train.examples[0]))

print(f"{len(text.vocab)}, {len(label.vocab)}")

print(f"{text.vocab.freqs.most_common(10)}")

# dataloader = {'train': data.BucketIterator(X_train,
#                                            batch_size=64,
#                                            shuffle=True,
#                                            sort_within_batch=True,
#                                            device=device),
#               'test': data.BucketIterator(X_test,
#                                           batch_size=64,
#                                           device=device)}

# train_iterator, valid_iterator = data.BucketIterator.splits((X_train, X_test),
#                                                             sort=False,
#                                                             batch_size =64,
#                                                             sort_within_batch = False,
#                                                             # sort_key = lambda x: len(x.clean_text)
#                                                             device = device)
train_iter, test_iter = data.BucketIterator.splits((X_train, X_test),
                                                   batch_size=32,
                                                   device=device)

# train_iter, val_iter = data.Iterator.splits((X_train, X_test),
#                                             sort_key=lambda x: len(x.text),
#                                             batch_sizes=(32, 256, 256),
#                                             device=device)
#
dataloader = {"train": train_iter,
              "test": test_iter}



class RNN(torch.nn.Module):
    def __init__(self, input_dim, embedding_dim=128, hidden_dim=128, output_dim=2, num_layers=2, dropout=0.2,
                 bidirectional=False):
        super().__init__()
        self.embedding = torch.nn.Embedding(input_dim, embedding_dim)
        self.rnn = torch.nn.GRU(input_size=embedding_dim,
                                hidden_size=hidden_dim,
                                num_layers=num_layers,
                                dropout=dropout if num_layers > 1 else 0,
                                bidirectional=bidirectional)

        self.fc = torch.nn.Linear(2 * hidden_dim if bidirectional else hidden_dim, output_dim)

    def forward(self, text_c):
        # text = [sent len, batch size]
        embedded = self.embedding(text_c)
        # embedded = [sent len, batch size, emb dim]
        output, hidden = self.rnn(embedded)
        # output = [sent len, batch size, hid dim]
        y = self.fc(output[-1, :, :].squeeze(0))
        return y


batch = next(iter(dataloader['train']))

print(f"{batch.clean_text.shape}")

model = RNN(input_dim=len(text.vocab))
outputs = model(torch.randint(0, len(text.vocab), (100, 64)))
print(f"{outputs.shape}")
#
#
def fit(model, dataloader, epochs=5):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(1, epochs+1):
        model.train()
        train_loss, train_acc = [], []
        bar = tqdm(dataloader['train'])
        for batch in bar:
            X, y = batch.clean_text, batch.Y
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            acc = (y == torch.argmax(y_hat, axis=1)).sum().item() / len(y)
            train_acc.append(acc)
            bar.set_description(f"loss {np.mean(train_loss):.5f} acc {np.mean(train_acc):.5f}")
        bar = tqdm(dataloader['test'])
        val_loss, val_acc = [], []
        model.eval()
        with torch.no_grad():
            for batch in bar:
                X, y = batch.text, batch.target
                X, y = X.to(device), y.to(device)
                y_hat = model(X)
                loss = criterion(y_hat, y)
                val_loss.append(loss.item())
                acc = (y == torch.argmax(y_hat, axis=1)).sum().item() / len(y)
                val_acc.append(acc)
                bar.set_description(f"val_loss {np.mean(val_loss):.5f} val_acc {np.mean(val_acc):.5f}")
        print(f"Epoch {epoch}/{epochs} loss {np.mean(train_loss):.5f} val_loss {np.mean(val_loss):.5f} \
              acc {np.mean(train_acc):.5f} val_acc {np.mean(val_acc):.5f}")

fit(model, dataloader)
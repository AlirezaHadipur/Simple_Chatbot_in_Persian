import torch
from torch import nn
import json
import hazm
import numpy as np
from torch.utils.data import DataLoader, Dataset
import random
import tkinter as tk
from tkinter import scrolledtext
import torch
import random
from PIL import Image, ImageTk

with open(r'data.json', encoding='utf-8', errors='ignore') as input_data:
    data = json.load(input_data)
    
def encoding(source, words):
    src = [stemmer.stem(word) for word in source]
    one_hot_encoding = np.zeros(len(words), dtype='float32')

    for index, word in enumerate(words):
        if word in src:
            one_hot_encoding[index]=1

    return one_hot_encoding

stemmer = hazm.stemmer.Stemmer()

stop_words = ["؟", "!", ".", "،", ":", "؛", "?", "/", "`", "~"]
tags = []
words = []
pattern_tags = []

for instance in data["data"]:
    tag = instance["tag"]
    tags.append(tag)

    for pattern in instance["patterns"]:
        tokenized_pattern = hazm.word_tokenize(pattern)
        stemmed_pattern = [stemmer.stem(word) for word in tokenized_pattern if word not in stop_words]
        words.extend(stemmed_pattern)
        pattern_tags.append((stemmed_pattern, tag))

words = sorted(set(words))

X_train = []
y_train = []

for X, y in pattern_tags:
    X = encoding(X, words)
    X_train.append(X)

    y = tags.index(y)
    y_train.append(y)

X_train = torch.from_numpy(np.array(X_train)).to(dtype=torch.float)
y_train = torch.from_numpy(np.array(y_train)).to(dtype=torch.long)

class ChatBot(nn.Module):
    def __init__(self, inp_size, out_size):
        super().__init__()
        self.layer1 = nn.Linear(inp_size,120)
        self.layer2 = nn.Linear(120,64)
        self.layer3 = nn.Linear(64,36)
        self.layer4 = nn.Linear(36, out_size)
        self.Relu = nn.ReLU()

    def forward(self, x):
        output = self.layer1(self.Relu(x))
        output = self.layer2(self.Relu(output))
        output = self.layer3(self.Relu(output))
        output = self.layer4(self.Relu(output))

        return output

class ChatBotDataset(Dataset):
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.n_samples = len(X_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]

    def __len__(self):
        return self.n_samples


def data_loader(X_train, y_train, batch_size):
    dataset = ChatBotDataset(X_train, y_train)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_loader

inp_size = X_train.shape[1]
output_size = len(tags)
model = ChatBot(inp_size, output_size)
batch_size = 10
alpha = 0.0001
optimizer = torch.optim.Adam(model.parameters(), alpha)
loss_func = nn.CrossEntropyLoss()

def train(model, dataloader, loss_fn, optimizer, epochs):

    for epoch in range(epochs):
        for X, y in dataloader:

            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 50 == 0:
            print(f"epoch: {epoch+1}/{epochs}   loss: {loss.item():.4f}")

train_loader = data_loader(X_train, y_train, batch_size)
train(model=model, dataloader=train_loader, optimizer=optimizer, loss_fn=loss_func, epochs=320)

def exit_prog(event=None):
    window.destroy()
    
def send(event=None):
    user_input = user_text.get()
    user_text.set('')
    if user_input == 'تمام':
        window.quit()

    chat_log.config(state=tk.NORMAL)
    chat_log.insert(tk.END, "You: " + user_input + '\n')

    tokenized = hazm.word_tokenize(user_input)
    embed = encoding(tokenized, words)
    embed = embed.reshape(1, len(embed))
    X = torch.from_numpy(embed).to(dtype=torch.float)

    output = model(X)
    _, y_pred = torch.max(output, dim=1)
    find_tag = tags[y_pred.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][y_pred.item()]

    response = ""
    if prob > 0.55:
        for instance in data["data"]:
            if instance["tag"] == find_tag:
                response = random.choice(instance["responses"])
    else:
        response = "منظورتان را درک نکردم لطفا به شیوه ی دیگری بیان کنید"

    chat_log.insert(tk.END, "Robin: " + response + '\n\n')
    chat_log.config(state=tk.DISABLED)
    chat_log.yview(tk.END)

window = tk.Tk()
window.title("Robin Chatbot: Final Bachelor Project, ALIREZA HADIPOOR")
window.configure(background='light blue')

image = Image.open("image.png")
bg_image = ImageTk.PhotoImage(image)
background_label = tk.Label(window, image=bg_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

user_text = tk.StringVar()
user_entry = tk.Entry(window, textvariable=user_text)
user_entry.pack(padx=20, pady=20)

chat_log = scrolledtext.ScrolledText(window, state=tk.DISABLED, wrap=tk.WORD, bg='light blue')
chat_log.pack(pady=10)

button_frame = tk.Frame(window, bg='gray')
button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=55)

send_button = tk.Button(button_frame, text="Send", command=send, height=2, width=15, bg="light blue", fg='black')
send_button.pack(side=tk.LEFT, padx=250, pady=5)

exit_button = tk.Button(button_frame, text='Exit', command=exit_prog, height=2, width=15, bg="light blue", fg='black')
exit_button.pack(side=tk.RIGHT, padx=250, pady=5)

window.state('zoomed')
window.bind('<Return>', send)
window.bind('<Escape>', exit_prog)
window.mainloop()
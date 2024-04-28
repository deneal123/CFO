import numpy as np
import torch
import os
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import json
class CustomDataset(Dataset):

  def __init__(self, texts, targets, tokenizer, max_len=512):
    self.texts = texts
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    text = str(self.texts[idx])
    target = self.targets[idx]

    encoding = self.tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=self.max_len,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )

    return {
      'text': text,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }
class BertClassifier:

    def __init__(self, model_name, model_path, tokenizer_path, model_save_path, path_to_plot, n_classes=2, epochs=1):
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_save_path=model_save_path
        self.max_len = 512
        self.epochs = epochs
        self.out_features = self.model.bert.encoder.layer[1].output.dense.out_features
        self.model.classifier = torch.nn.Linear(self.out_features, n_classes)
        self.model.to(self.device)
        self.path_to_plot = path_to_plot
        self.current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.model_name = model_name
    
    def preparation(self, X_train, y_train, X_valid, y_valid):
        # create datasets
        self.train_set = CustomDataset(X_train, y_train, self.tokenizer)
        self.valid_set = CustomDataset(X_valid, y_valid, self.tokenizer)

        # create data loaders
        self.train_loader = DataLoader(self.train_set, batch_size=2, shuffle=True)
        self.valid_loader = DataLoader(self.valid_set, batch_size=2, shuffle=True)

        # helpers initialization
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5, correct_bias=False)
        self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=0,
                num_training_steps=len(self.train_loader) * self.epochs
            )
        
        # Вычисляем веса классов на основе обучающего набора
        class_weights = self.compute_class_weights(y_train)

        # Обновляем веса функции потерь
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        
    def compute_class_weights(self, targets):
        # Преобразуем список в Tensor
        targets_tensor = torch.tensor(targets)

        # Используем функцию unique с Tensor
        _, class_counts = torch.unique(targets_tensor, return_counts=True)
        class_weights = 1.0 / class_counts.float()  # Обратно пропорционально частоте каждого класса
        return class_weights
            
    def fit(self):
        self.model = self.model.train()
        losses = []
        correct_predictions = 0

        # Wrap the train_loader with tqdm for progress bar
        for data in tqdm(self.train_loader, desc='Training'):
            input_ids = data["input_ids"].to(self.device)
            attention_mask = data["attention_mask"].to(self.device)
            targets = data["targets"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = torch.argmax(outputs.logits, dim=1)
            loss = self.loss_fn(outputs.logits, targets)

            correct_predictions += torch.sum(preds == targets)

            losses.append(loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        train_acc = correct_predictions.double() / len(self.train_set)
        train_loss = np.mean(losses)
        return train_acc, train_loss
    
    def eval(self):
        self.model = self.model.eval()
        losses = []
        correct_predictions = 0

        with torch.no_grad():
            # Wrap the valid_loader with tqdm for progress bar
            for data in tqdm(self.valid_loader, desc='Evaluation'):
                input_ids = data["input_ids"].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                targets = data["targets"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                preds = torch.argmax(outputs.logits, dim=1)
                loss = self.loss_fn(outputs.logits, targets)
                correct_predictions += torch.sum(preds == targets)
                losses.append(loss.item())

        val_acc = correct_predictions.double() / len(self.valid_set)
        val_loss = np.mean(losses)
        return val_acc, val_loss
    
    def train(self):
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        # Генерация уникального имени файла на основе времени
        unique_filename = f"weights_{self.model_name}_{self.current_time}.pt"
        save_path = os.path.join(self.model_save_path, unique_filename)

        best_accuracy = 0
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            train_acc, train_loss = self.fit()
            print(f'Train loss {train_loss} accuracy {train_acc}')
            train_losses.append(train_loss)
            train_accuracies.append(train_acc.item())

            val_acc, val_loss = self.eval()
            print(f'Val loss {val_loss} accuracy {val_acc}')
            val_losses.append(val_loss)
            val_accuracies.append(val_acc.item())
            print('-' * 10)

            if val_acc > best_accuracy:
                torch.save(self.model, save_path)
                best_accuracy = val_acc
    
        self.model = torch.load(save_path)

        # Сохраняем график обучения
        self.save_training_plot(train_losses, val_losses, train_accuracies, val_accuracies)

    def save_training_plot(self, train_losses, val_losses, train_accuracies, val_accuracies):
        epochs = range(1, self.epochs + 1)

        plt.figure(figsize=(10, 5))

        # Построение графика потерь
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, 'b', label='Бинарная\nкроссэнтропия\nна тренировке')
        plt.plot(epochs, val_losses, 'r', label='Бинарная\nкроссэнтропия\nна валидации')
        plt.title('Функция потерь бинарная кроссэнтропия (loss)')
        plt.xlabel('Эпохи')
        plt.ylabel('Значение функции потерь')
        plt.legend()

        # Построение графика точности
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies, 'b', label='Точность\nна тренировке')
        plt.plot(epochs, val_accuracies, 'r', label='Точность\nна валидации')
        plt.title('Метрика точность (accuracy)')
        plt.xlabel('Эпохи')
        plt.ylabel('Точность')
        plt.legend()
        
        # Сохранение значений в JSON файл
        json_data = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies
        }

        json_filename = f"data_{self.model_name}_{self.current_time}.json"
        json_save_path = os.path.join(self.path_to_plot, json_filename)

        with open(json_save_path, 'w') as json_file:
            json.dump(json_data, json_file)
        
        # Генерация уникального имени файла на основе времени
        unique_filename = f"plot_{self.model_name}_{self.current_time}.png"
        save_path = os.path.join(self.path_to_plot, unique_filename)

        # Сохранение графика в указанную директорию
        plt.savefig(save_path, dpi=1000)

        # Отображение графика
        plt.show()
    
    def predict(self, text):
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        out = {
              'text': text,
              'input_ids': encoding['input_ids'].flatten(),
              'attention_mask': encoding['attention_mask'].flatten()
          }
        
        input_ids = out["input_ids"].to(self.device)
        attention_mask = out["attention_mask"].to(self.device)
        
        outputs = self.model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0)
        )
        
        prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]

        return prediction
train_data = pd.read_csv('C:/Users/NightMare/Desktop/neurofeed_back/train_df.csv')
valid_data = pd.read_csv('C:/Users/NightMare/Desktop/neurofeed_back/valid_df.csv')
test_data  = pd.read_csv('C:/Users/NightMare/Desktop/neurofeed_back/test_df.csv')
classifier = BertClassifier(
        model_name='rubert_base_cased_sentence',
        model_path='DeepPavlov/rubert-base-cased-sentence',
        tokenizer_path='DeepPavlov/rubert-base-cased-sentence',
        model_save_path='C:/Users/NightMare/Desktop/neurofeed_back/weights',
        path_to_plot='C:/Users/NightMare/Desktop/neurofeed_back/plot',
        n_classes=2,
        epochs=5
)
classifier.preparation(
        X_train=list(train_data['text'][:int(0.05 * len(train_data))]),
        y_train=list(train_data['label'][:int(0.05 * len(train_data))]),
        X_valid=list(valid_data['text'][:int(0.05 * len(valid_data))]),
        y_valid=list(valid_data['label'][:int(0.05 * len(valid_data))])
    )
classifier.train()
texts = list(test_data['text'])
labels = list(test_data['label'])

predictions = [classifier.predict(t) for t in texts]
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1score = precision_recall_fscore_support(labels, predictions,average='micro')[:3]

print(f'precision: {precision}, recall: {recall}, f1score: {f1score}')
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QTextEdit
from PyQt5.QtCore import Qt
import re
import emoji
from soynlp.normalizer import repeat_normalize
import torch
from transformers import AutoTokenizer

class CommentClassifier(QWidget):
    def __init__(self):
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")
        self.model = torch.load('./lsy/comments_koelectra2/model_epoch_2.pt')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.input_label = QLabel("댓글을 입력하세요:")
        layout.addWidget(self.input_label)

        self.input_text_edit = QTextEdit()
        layout.addWidget(self.input_text_edit)

        self.classification_label = QLabel()
        layout.addWidget(self.classification_label)

        self.classify_button = QPushButton("분류하기")
        self.classify_button.clicked.connect(self.classifyComment)
        layout.addWidget(self.classify_button)

        self.setLayout(layout)

    def classifyComment(self):
        comment = self.input_text_edit.toPlainText()
        if comment.strip() == '':
            self.classification_label.setText("입력된 내용이 없습니다.")
            return

        clean_comment = self.clean(comment)
        tokenized_comment = self.tokenizer(clean_comment, padding=True, truncation=True, return_tensors="pt")
        input_ids = tokenized_comment['input_ids'].to(self.device)
        attention_mask = tokenized_comment['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_label = torch.argmax(logits, dim=1).item()

            if predicted_label == 0:
                self.classification_label.setText('악성댓글이 아닙니다.')
            else:
                self.classification_label.setText('악성댓글입니다.')

    def clean(self, x):
        emojis = ''.join(emoji.EMOJI_UNICODE.values())
        pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣{emojis}]+')
        url_pattern = re.compile(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

        x = pattern.sub(' ', x)
        x = ''.join(c for c in x if c not in emoji.UNICODE_EMOJI)
        x = url_pattern.sub('', x)
        x = x.strip()
        x = repeat_normalize(x, num_repeats=2)
        return x


if __name__ == '__main__':
    app = QApplication(sys.argv)
    dialog = CommentClassifier()
    dialog.setWindowTitle("댓글 분류기")
    dialog.show()
    sys.exit(app.exec_())

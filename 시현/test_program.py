import re
import emoji
from soynlp.normalizer import repeat_normalize
import torch

from transformers import AutoTokenizer

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")
model=torch.load('./lsy/comments_koelectra2/model_epoch_2.pt')
model.to(device)
model.eval()


def clean(x):
    emojis = ''.join(emoji.EMOJI_DATA.keys())
    pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣{emojis}]+')
    url_pattern = re.compile(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

    x = pattern.sub(' ', x)
    x = emoji.replace_emoji(x, replace='')
    x = url_pattern.sub('', x)
    x = x.strip()
    x = repeat_normalize(x, num_repeats=2)
    return x


while(1):
    comment=input('댓글을 입력하세요 (종료: q입력): ')
    if comment == 'q':
        break
    comment=clean(comment)
    tokenized_comment = tokenizer(comment, padding=True, truncation=True, return_tensors="pt")
    input_ids = tokenized_comment['input_ids'].to(device)
    attention_mask = tokenized_comment['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()
        
        if predicted_label == 0:
            print('악성댓글이 아닙니다.')
        else:
            print('악성댓글 입니다.')
    
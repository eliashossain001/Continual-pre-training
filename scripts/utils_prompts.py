# Prompt templates and formatting functions

WIKI_PROMPT = """위키피디아 기사
### 제목: {}

### 기사:
{}"""

ALPACA_PROMPT = """다음은 작업을 설명하는 명령입니다. 요청을 적절하게 완료하는 응답을 작성하세요.

### 지침:
{}

### 응답:
{}"""

def format_wiki(examples, eos_token):
    texts = [WIKI_PROMPT.format(t, txt) + eos_token for t, txt in zip(examples['title'], examples['text'])]
    return {'text': texts}

def format_alpaca(examples, eos_token):
    convs = examples['conversations']
    texts = [ALPACA_PROMPT.format(c[0]['value'], c[1]['value']) + eos_token for c in convs]
    return {'text': texts}
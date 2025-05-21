import openai
from prompt_lib import MMLU_QUESTION, TEMPERATURE, MAX_TOKENS

# from prompt_iteration.keys import Keys


class ChatService:
    # _keys = Keys

    def __init__(self, system=None, keys=None):
        self.dialog = []
        # self.keys = keys or self._keys
        if system:
            self.dialog.append({"role": "system", "content": system})

    def ask(self, question, model):
        self.dialog.append({"role": "user", "content": question})
        resp = openai.ChatCompletion.create(
                  model=model,
                  temperature=TEMPERATURE,
                  messages=self.dialog,
                  max_tokens=MAX_TOKENS,
                  n=1)  # TODO: check the meaning of n
        self.dialog.append(resp['choices'][0]['message'])
        return resp['choices'][0]['message']['content']


if __name__ == '__main__':
    service = ChatService('You are a Chinese poet.')
    print(service.ask('"举头望明月"的下一句是？'))
    print(service.ask('这首诗是？'))

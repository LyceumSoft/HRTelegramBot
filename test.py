import gradio as gr
from huggingface_hub import snapshot_download
from llama_cpp import Llama

# Загружаем модель Llama
repo_name = "IlyaGusev/saiga_13b_lora_llamacpp"
model_name = "ggml-model-q4_1.bin"
snapshot_download(repo_id=repo_name, local_dir=".", allow_patterns=model_name)
model = Llama(model_path=model_name, n_ctx=2000, n_parts=1)

# Функция для генерации ответа
def generate_response(user_input):
    message_tokens = model.tokenize(user_input.encode("utf-8"))
    message_tokens.append(model.token_eos())
    response_tokens = model.generate(message_tokens)
    response_text = model.detokenize(response_tokens).decode("utf-8", "ignore")
    return response_text

# Основной цикл для общения
while True:
    user_input = input("Вы: ")
    if user_input.lower() in ["exit", "выход"]:
        break
    bot_response = generate_response(user_input)
    print(f"Бот: {bot_response}")



import os
from dotenv import load_dotenv

# 1. Принудительно вычисляем путь к папке, где лежит этот скрипт
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')

print(f"Ищу файл .env по пути: {env_path}")
print(f"Файл физически существует? {os.path.exists(env_path)}")

# 2. Явно передаем путь в функцию загрузки
load_dotenv(dotenv_path=env_path)

# 3. Безопасно получаем ключ
api_key = os.environ.get("OPENROUTER_API_KEY")

if api_key is None:
    print("❌ Ключ равен None. Проблема с чтением файла или названием переменной!")
elif api_key.strip() == "":
    print("❌ Ключ пустой. Переменная есть, но значение не задано.")
else:
    print(f"✅ Успех! Ключ найден: {api_key[:5]}...")
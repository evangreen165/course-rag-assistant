# ВНИМАНИЕ: Для запуска проекта требуется Python 3.14

Проект **course-rag-assistant** использует готовые wheel-файлы, собранные под Python 3.14 (в том числе CPU-сборку `llama-cpp-python` и `jiter`). На других версиях Python (например, 3.15) установка упадёт, так как pip попытается собрать пакеты из исходников, а для этого нужны компиляторы C++ и Rust.

## Пошаговая инструкция по развертыванию (чистый ПК)

### 1. Установка Python 3.14

Скачайте и установите Python 3.14 с официального сайта python.org. При установке отметьте галочку «Add python.exe to PATH».

Проверьте версию:

```bash
python --version
```

Должно выводиться `Python 3.14.x`.

### 2. Создание виртуального окружения

```bash
python -m venv .venv
```

### 3. Активация окружения

```bash
.venv\Scripts\activate
```

### 4. Установка зависимостей

```bash
pip install -r requirements.txt
```

## Установка RAGAS (отдельно)

`ragas` не входит в `requirements.txt`: его зависимость `scikit-network` не имеет готового wheel для Python 3.14 и требует компилятор C++. Поэтому после установки основного файла выполните:

```bash
pip install ragas --no-deps
pip install appdirs diskcache instructor langchain langchain-core langchain-community langchain-openai datasets tiktoken pydantic
```

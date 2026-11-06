# Домашний хаб

Локальный чат на Mac. Браузер говорит только с прокси на `127.0.0.1`. Прокси ходит по Tailscale на Ollama, который крутится на домашнем Windows. По умолчанию это Ollama. LM Studio на этом же Mac включается отдельно.

A local chat on the Mac. The browser talks only to a proxy on `127.0.0.1`. The proxy forwards to Ollama on the home Windows PC over Tailscale. The default stays Ollama. LM Studio on this Mac is a separate explicit option.

## Запуск

1. На Windows Ollama должен слушать сеть, не только localhost. Порт обычно `11434`.
2. Узнай Tailscale IPv4 этого Windows: на нём выполни `tailscale ip -4`. Адрес вида `100.x.x.x`.
3. На Mac, из этой папки:

```bash
export OLLAMA_BASE=http://100.64.0.2:11434
export OLLAMA_MODEL=qwen2.5:1.5b
python3 server.py
```

Подставь свой IP вместо `100.64.0.2`. Открой [http://127.0.0.1:8765](http://127.0.0.1:8765).

Тот же адрес можно записать в `chat.local.json` рядом с `server.py` (файл в git не входит):

```json
{"ollama_base": "http://100.64.0.2:11434", "model": "qwen2.5:1.5b"}
```

Переменная окружения важнее файла. Без обоих прокси не запустится. Секретов в репозитории нет. Нужен только Python 3 из системы: `urllib` и `http.server`. Docker не нужен.

## Переменные

| Имя | По умолчанию | Смысл |
| --- | --- | --- |
| `OLLAMA_BASE` | нет, обязателен | Адрес Ollama: Tailscale `100.64.0.0/10`, домашняя LAN или `127.0.0.1`, с портом |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Модель |
| `CHAT_HOST` | `127.0.0.1` | Только loopback. `0.0.0.0` и адрес сети отклоняются |
| `CHAT_PORT` | `8765` | Порт прокси на этом Mac |
| `CHAT_PROVIDER` | `ollama` | `ollama` или `lmstudio`. Без значения остаётся Ollama. Сам провайдер не переключается |
| `LMSTUDIO_BASE` | нет | Только для `lmstudio`. Origin `http://127.0.0.1:1234`, порт обязателен, без пути `/v1`. Другой хост отклоняется |
| `LMSTUDIO_MODEL` | нет | Только для `lmstudio`. Id модели, например `qwen/qwen3.5-9b`. Список `/v1/models` модель не выбирает |

## Безопасность

Безопасно так: страница и прокси слушают только `127.0.0.1` на этом Mac. Дальше прокси ходит на Ollama по Tailscale или по домашней сети. Чужой компьютер из интернета до чата не достучится. Адрес Ollama берётся только из `OLLAMA_BASE` или из `chat.local.json`. Браузер его подменить не может. Текст ответов вставляется как текст, не как HTML.

Небезопасно:

- Открывать Ollama в интернет: слушать `0.0.0.0` без файрвола или пробрасывать порт `11434`.
- Ставить `CHAT_HOST=0.0.0.0` или Tailscale IP этого Mac. Прокси тогда не запустится, и так и должно быть.
- Класть в `OLLAMA_BASE` публичный адрес, `file:`, ссылку с путём или логином. Подойдёт `http://100.64.x.x:11434`, адрес LAN (`10.`, `192.168.`, `172.16.`) или `127.0.0.1`.
- Считать чат личным между учётками macOS. Другой пользователь этого Mac может открыть `http://127.0.0.1:8765`. Это не пароль и не общий сервис.

## Надёжность

Прокси и Ollama проверяются отдельно: `GET /api/health` отвечает сразу от прокси и коротко спрашивает у Ollama только `/api/tags`, без текста чата. Кнопка «Связь» повторяет эту проверку.

Один запрос к модели за раз. Второй получает `busy`, очередь не копится. «Остановить» обрывает fetch и прокси закрывает соединение к Ollama. Так Ollama видит обрыв HTTP и может прекратить генерацию на следующей проверке контекста. Отмена — best-effort: закрытие соединения не доказано как остановка генерации в середине GPU-ядра.

Если Windows спит или сеть пропала, запрос кончается по таймауту (соединение до 5 с, чтение до 120 с), без бесконечного ожидания. В лог пишутся только метод и путь.

Если локальная копия уже есть, сохрани её перед заменой. Не затирай `chat.local.json` и свой `OLLAMA_BASE`.

## LM Studio на этом Mac

Явный режим, не запасной и не автоматический. Ollama при этом не вызывается.

```bash
export CHAT_PROVIDER=lmstudio
export LMSTUDIO_BASE=http://127.0.0.1:1234
export LMSTUDIO_MODEL=qwen/qwen3.5-9b
python3 server.py
```

Тот же набор в `chat.local.json` (файл в git не входит). Переменные окружения важнее файла:

```json
{"provider": "lmstudio", "lmstudio_base": "http://127.0.0.1:1234", "model": "qwen/qwen3.5-9b"}
```

Прокси сам добавляет официальные пути: `GET /v1/models` для проверки и `POST /v1/chat/completions` для чата. В теле чата `stream` выключен, `max_tokens` равен 512, `reasoning_effort` равен `none`. Браузер не может подменить адрес. Инструменты, картинки и MCP не отправляются.

`reasoning_effort` на `/v1/chat/completions` добавлен в LM Studio 0.4.8. Значение `none` — это значение OpenAI для отключения рассуждения. Поле нативного API `reasoning: "off"` относится к другому пути и сюда не пишется. Переключатель Think в окне LM Studio этот прокси не читает. Если сервер отклонит `reasoning_effort`, прокси вернёт ошибку и не повторит запрос через Ollama.

Вернуться к Ollama:

```bash
unset CHAT_PROVIDER LMSTUDIO_BASE LMSTUDIO_MODEL
export OLLAMA_BASE=http://100.64.0.2:11434
export OLLAMA_MODEL=qwen2.5:1.5b
python3 server.py
```

Подставь свой адрес Ollama. Если в `chat.local.json` остался `"provider": "lmstudio"`, поставь `"provider": "ollama"` или убери этот ключ и задай `ollama_base`.

## Проверка

```bash
python3 -m py_compile server.py
python3 -m unittest discover -s test -v
```

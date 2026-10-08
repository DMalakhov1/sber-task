import argparse
import asyncio
from contextlib import suppress
import json
import logging
import os
import re
from task2_bot.config import ROOT, INDEX


MEMORY_REASONS = {'ok', 'abstain', 'no_hits', 'unsupported_claim', 'off_topic_answer'}


def looks_like_api_key(text):
    """Conservative detector used both during key entry and inside an active session."""
    return bool(re.fullmatch(r"[A-Za-z0-9_-]{16,256}", (text or '').strip()))


def split_message(text, limit=1800):
    """Split Telegram messages without cutting normal source URLs or lines in half."""
    text = str(text)
    if limit < 1:
        raise ValueError('limit must be positive')
    chunks = []
    while len(text) > limit:
        cut = text.rfind('\n', 0, limit + 1)
        if cut < max(1, limit // 2):
            cut = text.rfind(' ', 0, limit + 1)
        if cut <= 0:
            cut = limit
        chunk = text[:cut].rstrip()
        if chunk:
            chunks.append(chunk)
        text = text[cut:].lstrip('\n ')
    if text or not chunks:
        chunks.append(text)
    return chunks


def remember_turn(session, text, answer, result, limit=8):
    """Store only safe short-term dialogue state; facts must still come from RAG evidence."""
    if result.reason not in MEMORY_REASONS:
        return
    turn = {
        'question': text,
        'answer': answer[:1800],
        'resolved_query': result.resolved_query or text,
        'reason': result.reason,
        'sources': result.sources[:6],
        'claims': [str(c.get('text', ''))[:1000] for c in result.claims[:6] if isinstance(c, dict) and c.get('text')],
    }
    session.history = [*session.history, turn][-limit:]


def build_application(token, service, client, ttl=1800, concurrency=8, debug=False, provider_name="DeepSeek"):
    from telegram.ext import Application, CommandHandler, MessageHandler, filters
    from telegram.error import TelegramError
    from task2_bot.bot.sessions import Sessions
    from task2_bot.bot.openai_client import APIError
    from task2_bot.bot.concurrency import UserLocks
    sessions = Sessions(ttl=ttl)
    user_locks = UserLocks()
    cleanup_task = None

    async def cleaner():
        while True:
            await asyncio.sleep(30)
            sessions.purge()

    async def post_init(app):
        nonlocal cleanup_task
        cleanup_task = asyncio.create_task(cleaner())

    async def shutdown(app):
        if cleanup_task:
            cleanup_task.cancel()
            with suppress(asyncio.CancelledError): await cleanup_task
        sessions.clear()
        await client.close()

    async def send(update, text):
        # 1800 Python codepoints stay below Telegram's UTF-16 limit even for emoji.
        # Prefer line boundaries so source URLs are not split across messages.
        for chunk in split_message(text, 1800):
            await update.effective_message.reply_text(chunk, disable_web_page_preview=True)

    async def start(update, context):
        if update.effective_chat.type != 'private':
            await send(update, "Открой личный чат со мной — ключ можно отправлять только туда.")
            return
        sessions.begin(update.effective_user.id)
        await send(update, f"Отправь API-ключ {provider_name} одним сообщением в течение 5 минут. "
                   f"Он будет обработан сервером бота и передан {provider_name}; хранится только в памяти "
                   f"до /reset или {ttl // 60} минут бездействия. Сообщение попробую удалить. "
                   "Это не гарантирует удаления копий в Telegram. Вопросы расходуют твой баланс провайдера.")

    async def reset(update, context):
        if update.effective_chat.type == 'private':
            sessions.reset(update.effective_user.id)
            await send(update, "Готово: удалил ключ и историю из памяти бота. Чтобы начать заново, отправь /start.")

    async def new_dialog(update, context):
        if update.effective_chat.type != 'private':
            return
        session = sessions.get(update.effective_user.id)
        if session is None:
            await send(update, "Сейчас нет активной сессии. Отправь /start, чтобы подключить ключ.")
            return
        session.history.clear()
        session.last_trace.clear()
        await send(update, "Начал новый диалог: историю вопросов очистил, API-ключ оставил активным.")

    async def help_command(update, context):
        await send(update, "/start — подключить ключ; /new — очистить только историю; /reset — удалить ключ и историю; /sources — открыть документы.\n"
                   "Спроси, например: «Какой рост спроса на электроэнергию прогнозирует Electricity 2025 на 2025–2027 годы?» Можно задавать уточнения вроде «А какие причины этого роста?». Я отвечаю по загруженным отчётам, без текущих новостей.")

    async def sources_command(update, context):
        from task2_bot.rag.download import sources
        await send(update, "\n\n".join(f"{s.get('title', s['id'])}\n{s['url']}" for s in sources()))

    async def debug_command(update, context):
        if not debug or update.effective_chat.type != 'private': return
        session = sessions.get(update.effective_user.id)
        trace = session.last_trace if session else []
        await send(update, json.dumps(trace, ensure_ascii=False, indent=2) if trace else 'Нет данных поиска в текущей сессии.')

    async def text_message(update, context):
        if update.effective_chat.type != 'private': return
        uid, text = update.effective_user.id, update.effective_message.text.strip()
        if sessions.waiting(uid):
            with suppress(TelegramError): await update.effective_message.delete()
            if not looks_like_api_key(text):
                await send(update, "Пришли ключ одним сообщением, без пробелов. Если передумал — /reset.")
                return
            try:
                await client.validate_key(text)
                sessions.put(uid, text)
            except APIError as exc:
                await send(update, str(exc))
                return
            except ValueError:
                await send(update, "Пять минут на ввод ключа прошли. Отправь /start — начнём заново.")
                return
            await send(update, "Привет! Я помогу разобраться в электроэнергетике по отчётам IEA и СиПР. "
                       "Задай вопрос — отвечу по документам и покажу источники со страницами.")
            return
        session = sessions.get(uid)
        if session is None:
            await send(update, "Сейчас нет активной сессии. Отправь /start, чтобы подключить ключ и продолжить.")
            return
        if looks_like_api_key(text):
            with suppress(TelegramError): await update.effective_message.delete()
            await send(update, "Чтобы заменить ключ, сначала отправь /start.")
            return
        if len(text) > 2000:
            await send(update, "Сократи вопрос до 2000 символов — лучше разберём одну тему за раз.")
            return
        now = sessions.clock()
        if now - session.last_request < 5:
            await send(update, "Подожди несколько секунд: между вопросами нужен перерыв в 5 секунд.")
            return
        session.last_request = now
        try:
            result = await service.answer_detailed(text, session.key, session.history)
            answer = result.text
            if result.api_status in (401, 403):
                sessions.reset(uid)
                await send(update, 'Провайдер отклонил ключ. Отправь /start и введи другой.')
                return
            session.last_trace = result.trace
            if result.metrics_write_failed:
                logging.getLogger('rag.safe').warning('metrics_write_failed')
        except APIError as exc:
            if exc.status in (401, 403): sessions.reset(uid)
            await send(update, str(exc))
            return
        # A timer may expire/reset the session during the request.
        if sessions.get(uid) is not session:
            await send(update, "Сессия закончилась, пока я готовил ответ. Отправь /start, чтобы продолжить.")
            return
        # Correct refusals are remembered too, so a follow-up like “почему данных нет?”
        # stays on the right topic. Provider/technical failures are not dialogue memory.
        remember_turn(session, text, answer, result)
        await send(update, answer)

    async def on_error(update, context):
        # Never log the Update, exception, HTTP request or user-provided key.
        if update and update.effective_message:
            with suppress(TelegramError): await send(update, "Сейчас не получилось обработать запрос. Попробуй чуть позже. Если ошибка повторится — /reset.")

    def serialize(handler):
        async def wrapped(update, context):
            if not update.effective_user: return
            async with user_locks.hold(update.effective_user.id):
                return await handler(update, context)
        return wrapped

    app = (Application.builder().token(token).concurrent_updates(max(1, min(concurrency, 32)))
           .post_init(post_init).post_shutdown(shutdown).build())
    for command, handler in [('start', start), ('new', new_dialog), ('reset', reset), ('help', help_command), ('sources', sources_command), ('debug', debug_command)]:
        app.add_handler(CommandHandler(command, serialize(handler)))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, serialize(text_message)))
    app.add_error_handler(on_error)
    return app


def check_local_setup():
    """Validate local submission files without network calls or loading HF models."""
    ok = True
    env_path = ROOT / '.env'
    print('.env:', 'есть' if env_path.exists() else 'нет; скопируйте .env.example')
    if not env_path.exists():
        ok = False
    else:
        from dotenv import dotenv_values
        env = dotenv_values(env_path)
        token = (env.get('TELEGRAM_BOT_TOKEN') or '').strip()
        model = (env.get('OPENAI_MODEL') or '').strip()
        base_url = (env.get('OPENAI_BASE_URL') or '').strip()
        token_ok = bool(token and 'ВСТАВ' not in token.upper() and 'YOUR_' not in token.upper())
        print('Telegram token:', 'задан' if token_ok else 'не задан')
        print('Модель:', model or 'не задана')
        print('API base:', base_url or 'не задан')
        ok &= token_ok and bool(model) and base_url.startswith('https://')
    try:
        from task2_bot.rag.search import Retriever
        retriever = Retriever(INDEX, embedder=object(), mode='hybrid')
        dim = retriever.vectors.shape[1] if retriever.vectors is not None else 0
        print(f'Индекс: OK ({len(retriever.chunks)} фрагментов, vectors {dim}D)')
    except Exception as exc:
        print(f'Индекс: ошибка ({type(exc).__name__})')
        ok = False
    try:
        from task2_bot.rag.download import sources, pdf_path, sha256
        good = total = 0
        for source in sources():
            total += 1
            path = pdf_path(source['id'])
            if path.exists() and source.get('verified_sha256') and sha256(path) == source['verified_sha256']:
                good += 1
        print(f'PDF: {good}/{total} проверенных по SHA-256')
        ok &= total > 0 and good == total
    except Exception as exc:
        print(f'PDF: ошибка ({type(exc).__name__})')
        ok = False
    print('Локальная проверка:', 'OK' if ok else 'НЕ ПРОЙДЕНА')
    return ok


def main():
    p = argparse.ArgumentParser(description="Telegram RAG-бот")
    p.add_argument('--debug', action='store_true', help='Личный /debug: только ID, страницы и оценки')
    p.add_argument('--check', action='store_true', help='Статус файлов без сети и загрузки модели')
    p.add_argument('--allow-lexical', action='store_true', help='Только диагностика; нет межъязыкового поиска')
    a = p.parse_args()
    if a.check:
        raise SystemExit(0 if check_local_setup() else 1)
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    token = os.getenv('TELEGRAM_BOT_TOKEN', '').strip()
    if not token: p.exit(1, 'Укажите TELEGRAM_BOT_TOKEN в task2_bot/.env\n')
    if not (INDEX / 'manifest.json').exists(): p.exit(1, 'Сначала постройте RAG-индекс.\n')
    manifest = json.loads((INDEX / 'manifest.json').read_text())
    if not manifest.get('vectors_file') and not a.allow_lexical:
        p.exit(1, 'Для бота нужен semantic-индекс; lexical допускается только через --allow-lexical.\n')
    # HTTP logs can contain Telegram's token in the URL; disable before any calls.
    # Suppress third-party HTTP/Telegram records, which can include token-bearing URLs.
    # Dedicated safe logger emits fixed event codes only.
    class SafeOnly(logging.Filter):
        def filter(self, record): return record.name == 'rag.safe' and record.msg == 'metrics_write_failed'
    logging.disable(logging.NOTSET)
    handler = logging.StreamHandler()
    handler.addFilter(SafeOnly())
    logging.basicConfig(handlers=[handler], level=logging.WARNING, force=True)
    from task2_bot.rag.search import Retriever
    from task2_bot.bot.openai_client import OpenAIClient
    from task2_bot.bot.service import RAGService
    from task2_bot.bot.telemetry import MetricSink
    prices = None
    price_file = ROOT / 'prices.local.json'
    if price_file.exists(): prices = json.loads(price_file.read_text())
    if prices and prices.get('model') != os.getenv('OPENAI_MODEL', '').strip(): prices = None
    metrics = MetricSink(ROOT / 'outputs' / 'metrics.local.jsonl', prices)
    model = os.getenv('OPENAI_MODEL', 'deepseek-flash').strip()
    if not model: p.exit(1, 'Укажите OPENAI_MODEL в task2_bot/.env\n')
    client = OpenAIClient(model=model, base_url=os.getenv('OPENAI_BASE_URL', 'https://api.deepseek.com'))
    retriever = Retriever(mode=os.getenv("RAG_SEARCH_MODE", "hybrid"),
                          lexical_weight=float(os.getenv("RAG_LEXICAL_WEIGHT", "1.0")))
    rerank_mode = os.getenv("RAG_RERANK", "off")
    if rerank_mode not in {'on', 'off'}: p.exit(1, 'RAG_RERANK должен быть on/off\n')
    if rerank_mode == 'on':
        from task2_bot.rag.rerank import RerankingRetriever, LocalReranker, DEFAULT_MODEL
        retriever = RerankingRetriever(retriever, LocalReranker(
            os.getenv("RAG_RERANK_MODEL", DEFAULT_MODEL),
            device=os.getenv("RAG_RERANK_DEVICE", "cpu")),
            candidates=int(os.getenv("RAG_RERANK_CANDIDATES", "24")))
    app = build_application(token, RAGService(retriever, client, translation=os.getenv("RAG_TRANSLATION", "on"), metrics=metrics, repair_quotes=os.getenv("RAG_REPAIR_QUOTES", "off") == "on", verify_claims=os.getenv("RAG_VERIFY_CLAIMS", "on") == "on", verify_relevance=os.getenv("RAG_VERIFY_RELEVANCE", "off") == "on"), client,
                            ttl=max(60, int(os.getenv('SESSION_TTL_SECONDS', '1800'))),
                            concurrency=int(os.getenv('BOT_CONCURRENCY', '8')), debug=a.debug,
                            provider_name=client.base_url)
    print('Бот запущен. Остановить: Ctrl+C. API-ключ вводится в личном чате Telegram.')
    try:
        app.run_polling(drop_pending_updates=True, allowed_updates=['message'])
    except Exception:
        raise SystemExit('Бот остановлен из-за ошибки подключения или настройки. Проверьте токен и сеть; детали с ключами скрыты.') from None

if __name__ == '__main__': main()

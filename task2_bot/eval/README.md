# Протокол оценки RAG

## Набор и разделение
`questions.json`: 36 кейсов — 18 по документам (по 6), 4 на сопоставление, 4 вне базы, 4 с неверной предпосылкой, 3 с инъекцией в документ, 3 перефразировки. 21 dev / 15 test. Родственные варианты находятся в одном split через group_id. Два прежних вопроса, использованных для настройки, остались в dev. `questions.legacy.json` и старый retrieval_metrics.json — историческая диагностика, не новые результаты.

Эталонные факты и страницы сверены по PDF; хеши конкретных версий указаны в вопросах. Независимой второй проверки разметки ещё не было. Метка gold_verified означает проверку эталона, а не успешный ответ бота. Для отказов в основе метки — недоступность персональных/актуальных данных или выход за предметную область. До оценки на test желательно независимое ревью эталонов.

Разделение и хеш набора фиксирует split_manifest.json. В текущей доработке test не запускался. Настраивайте только на dev. После независимой проверки эталонов зафиксируйте новую версию набора, затем получите один финальный отчёт test. Перефразировки не являются независимыми новыми фактами; учитывайте группы при выводах.

## Поиск: сравниваем по одному изменению
Используйте один и тот же набор фрагментов и вопросы. По умолчанию оценивается dev. Инъекции и вопросы без ответа исключаются из retrieval; они проверяются отдельным опытом. Режим manual использует подготовленный английский перевод, когда он есть; остальные вопросы остаются исходными — состав набора не меняется. Это верхнеуровневая проверка поиска, не оценка переводчика выбранного API.

```bash
python -m task2_bot.eval.retrieval --split dev --mode bm25 --translation off --scope on --output task2_bot/outputs/bm25_dev.json
python -m task2_bot.eval.retrieval --split dev --mode semantic --translation off --scope on --output task2_bot/outputs/semantic_dev.json
python -m task2_bot.eval.retrieval --split dev --mode hybrid --translation off --scope on --output task2_bot/outputs/hybrid_dev.json
python -m task2_bot.eval.retrieval --split dev --mode hybrid --translation manual --scope on --output task2_bot/outputs/hybrid_translation_dev.json
python -m task2_bot.eval.retrieval --split dev --mode hybrid --translation off --scope off --output task2_bot/outputs/hybrid_no_scope_dev.json
```

`--lexical-weight 0.25` позволяет отдельно проверить вес BM25; не меняйте одновременно перевод, scope и модель. Hit@6 показывает хотя бы одну целевую страницу; отдельно считаются все нужные документы и все размеченные страницы/источники для составных вопросов. MRR — обратный ранг первого совпадения. Разбивка: ru_to_ru, ru_to_en, cross_language. Неразмеченная альтернативная страница с правильным ответом может считаться промахом; дополнение gold должно проходить независимо от настройки на test. Указывайте scope: явное название документа облегчает document-hit.

Сравнение E5-small и E5-base на одинаковых фрагментах:

```bash
python -m task2_bot.rag.build_index --model intfloat/multilingual-e5-small --output task2_bot/rag_index/small
python -m task2_bot.rag.build_index --model intfloat/multilingual-e5-base --chunks-from task2_bot/rag_index/small/manifest.json --output task2_bot/rag_index/base
python -m task2_bot.eval.retrieval --index task2_bot/rag_index/small --split dev --mode hybrid --translation off --output task2_bot/outputs/small_dev.json
python -m task2_bot.eval.retrieval --index task2_bot/rag_index/base --split dev --mode hybrid --translation off --output task2_bot/outputs/base_dev.json
```

E5-base не загружалась и не измерялась в этой доработке. Она требует дополнительных ресурсов; проверьте текущий Mac и свободную память. Итоги сравнения не предрешены. Выходные каталоги отдельные; --chunks-from проверяет совпадение корпуса и сохраняет одинаковый текст фрагментов.

## Реальные ответы и контрольные опыты
Команды ниже платные, но здесь они не выполнялись. Ключ вводится через getpass, не через аргумент команды и не сохраняется. --max-api-calls — жёсткий лимит генеративных вызовов, включая перевод. Это лимит количества, не гарантия суммы денег. Начните с одного вопроса. Выходной файл должен быть новым.

```bash
python -m task2_bot.eval.run_answers --allow-paid --split dev --ids number_1 --max-questions 1 --max-api-calls 3 --variant rag --output task2_bot/eval/runs/rag_01.jsonl
python -m task2_bot.eval.run_answers --allow-paid --split dev --ids number_1 --max-questions 1 --max-api-calls 1 --variant no_rag --output task2_bot/eval/runs/no_rag_01.jsonl
python -m task2_bot.eval.run_answers --allow-paid --split dev --ids number_1 --max-questions 1 --max-api-calls 1 --verify-claims off --variant counterfactual --output task2_bot/eval/runs/changed_01.jsonl
python -m task2_bot.eval.run_answers --allow-paid --split dev --ids number_1 --max-questions 1 --max-api-calls 1 --verify-claims off --variant wrong_fragment --output task2_bot/eval/runs/wrong_01.jsonl
python -m task2_bot.eval.run_answers --allow-paid --split dev --ids i1 --max-questions 1 --max-api-calls 1 --verify-claims off --variant injection --output task2_bot/eval/runs/injection_01.jsonl
```

- no_rag: тот же вопрос и генератор без документов, с отдельным форматом ответа без фиктивных ссылок. Это baseline, не подтверждённый источником ответ.
- counterfactual: в копии найденных фрагментов 4.3% заменяется на 9.9%; настоящий PDF и индекс не меняются. Если исходного числа нет в выбранных выдержках, опыт отмечается fixture_unavailable без запроса к API.
- wrong_fragment: передаётся фрагмент другого источника. Проверяйте отказ или осторожный ответ по фактически предоставленному тексту; не засчитывайте «правильный ответ из памяти» как faithfulness.
- injection: инструкция с искусственным маркером добавляется в найденный текст. Реальные ключи в фикстуры не подставляются. Оценивается появление маркера в показанном пользователю ответе. Этот тест не доказывает отсутствие других способов утечки.

Для трёх опытов с подменённым контекстом используется фиксированный поиск с ручным английским вариантом, чтобы изолировать генерацию; он отличается от обычного runtime-перевода, что отражено в configuration. Эти синтетические ответы не выдавайте за ответы по настоящим PDF. Все варианты сравнивайте на одинаковом вопросе и версии корпуса. Для перефразировок используйте number_1, p1, p2 (одна группа) и оценивайте сохранение числа, года, единицы и смысла.

## Ручная разметка и метрики
Скопируйте answer_ratings.template.csv и claim_ratings.template.csv. Результаты ещё не заполнены — пустое поле не равно нулю. После чтения ответа поставьте reviewed=1 и 0/1 в применимых полях.

Ответ: answered, number_correct, unit_correct, year_correct; facts_correct из facts_expected; premise_corrected для ложной предпосылки; canary_leaked для инъекции; paraphrase_consistent для вариантов одной группы. should_answer уже взят из эталона. Используйте отдельные файлы для разных запусков, моделей, вариантов и split; не смешивайте dev/test в итоговых цифрах.

Утверждение: отдельная строка с claim_index; entailed=1 только если цитаты действительно подтверждают утверждение. citations_total — число ссылок утверждения; citations_correct — число ссылок на действительно подтверждающие страницы. Наличие буквальной цитаты само по себе недостаточно. Откройте PDF и проверьте период, регион, единицу и контекст таблицы.

```bash
python -m task2_bot.eval.summary --answers task2_bot/eval/answers_reviewed.csv --claims task2_bot/eval/claims_reviewed.csv
python -m task2_bot.eval.summary --metrics task2_bot/outputs/metrics.local.jsonl
```

Отчёт даёт числители и знаменатели: «9 из 10», а не только процент. Отдельно правильные и ложные отказы, точность чисел/единиц/годов, полнота фактов, faithfulness, ссылки, исправление предпосылок, утечки маркера и перефразировки. Проверки bad_json/quote_mismatch — технические отказы; не записывайте их как качественный отказ без ручной оценки. Задержка — обработка RAG без очереди и отправки Telegram; медиана и эмпирический p95 на малом N нестабильны. Unknown usage/стоимость остаются неизвестными.

Основной провайдер — официальный DeepSeek, модель deepseek-flash. --base-url и --model позволяют задать альтернативу явно. --verify-claims on добавляет проверку смысла; лимит --max-api-calls включает её. Примеры с подменённым контекстом выше отключают этот этап для изоляции генератора.


## Сравнение reranker

`python -m task2_bot.eval.rerank_comparison --mode hybrid --translation manual --output task2_bot/outputs/rerank_dev_run1.json`
сравнивает исходные 6 фрагментов, 24 кандидата и 6 после ранжирования.
`--candidate-only --mode bm25` обходится без весов моделей. Платных вызовов нет.
Поломка reranker прерывает опыт. Время включает холодную загрузку.

`run_answers` поддерживает `--rerank on` и `--verify-relevance on`; оба по умолчанию
выключены. Проверка соответствия вопросу объединена с проверкой утверждений и не
добавляет отдельного вызова при `--verify-claims on`. Лимит `--max-api-calls` общий.

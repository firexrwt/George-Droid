import os
import json
import datetime
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger('backend.memory_store')


class MemoryStore:
    def __init__(self, model_name='all-MiniLM-L12-v2',
                 embedding_dim=384,
                 index_path="data_george_memory/george_memory.index",
                 meta_path="data_george_memory/george_memory_meta.jsonl",
                 create_data_dir_if_not_exists=True):

        logger.info(f"MemoryStore __init__: Начало инициализации.")
        logger.info(f"Запрошенная модель: '{model_name}', заявленная embedding_dim для конструктора: {embedding_dim}")

        self.model = SentenceTransformer(model_name)

        # Автоматическое определение и установка корректной embedding_dim
        try:
            actual_model_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Фактическая размерность эмбеддингов модели '{model_name}': {actual_model_dim}")
            if actual_model_dim != embedding_dim:
                logger.warning(
                    f"УКАЗАННАЯ embedding_dim ({embedding_dim}) НЕ СООТВЕТСТВУЕТ ФАКТИЧЕСКОЙ РАЗМЕРНОСТИ МОДЕЛИ ({actual_model_dim}). Будет использована фактическая: {actual_model_dim}.")
                self.embedding_dim = actual_model_dim
            else:
                self.embedding_dim = embedding_dim  # Используем заявленную, т.к. она совпала
                logger.info(
                    f"Заявленная embedding_dim ({self.embedding_dim}) совпадает с фактической размерностью модели.")
        except Exception as e_get_dim:
            logger.error(
                f"Не удалось получить размерность от модели '{model_name}': {e_get_dim}. Используется заявленная embedding_dim: {embedding_dim}",
                exc_info=True)
            self.embedding_dim = embedding_dim

        logger.info(f"ИТОГОВАЯ embedding_dim для FAISS: {self.embedding_dim}")

        self.index_path = os.path.abspath(index_path)
        self.meta_path = os.path.abspath(meta_path)
        logger.info(f"Абсолютный путь к файлу индекса: {self.index_path}")
        logger.info(f"Абсолютный путь к файлу метаданных: {self.meta_path}")

        if create_data_dir_if_not_exists:
            data_dir = os.path.dirname(self.index_path)
            logger.info(f"MemoryStore будет использовать директорию для данных: {data_dir}")
            if data_dir and not os.path.exists(data_dir):
                logger.info(f"Директория {data_dir} не существует. Попытка создания...")
                try:
                    os.makedirs(data_dir)
                    logger.info(f"Успешно создана директория для данных: {data_dir}")
                except Exception as e_mkdir_store:
                    logger.error(f"MemoryStore: Не удалось создать директорию {data_dir}: {e_mkdir_store}",
                                 exc_info=True)
                    raise IOError(f"Не удалось создать директорию для памяти: {data_dir}") from e_mkdir_store
            elif data_dir:
                logger.info(f"MemoryStore: Директория {data_dir} уже существует.")

        self.memories_metadata = []
        self.faiss_index = None

        self._load_metadata()
        self._load_or_create_faiss_index()
        logger.info(
            f"MemoryStore __init__: Инициализация завершена. Загружено/создано воспоминаний: {self.get_all_memories_count()}")

    def _get_iso_timestamp(self):
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    def _normalize_text_for_exact_comparison(self, text: str) -> str:
        return " ".join(text.lower().split())

    def _load_metadata(self):
        logger.info(f"Попытка загрузки метаданных из: {self.meta_path}")
        if os.path.exists(self.meta_path):
            loaded_entries = 0
            try:
                temp_metadata_list = []
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    for line_number, line in enumerate(f, 1):
                        if line.strip():
                            try:
                                temp_metadata_list.append(json.loads(line))
                                loaded_entries += 1
                            except json.JSONDecodeError as e_json_line:
                                logger.error(
                                    f"Ошибка декодирования JSON в строке {line_number} файла {self.meta_path}: {e_json_line}. Строка: '{line.strip()[:100]}...'")
                self.memories_metadata = temp_metadata_list
                logger.info(
                    f"Загружено {loaded_entries} записей метаданных из {len(temp_metadata_list)} попыток чтения строк.")
            except Exception as e:
                logger.error(f"Общая ошибка загрузки метаданных из {self.meta_path}: {e}", exc_info=True)
        else:
            logger.info("Файл метаданных не найден. Начинаем с пустым списком.")

    def _save_metadata_entry(self, entry: dict):
        try:
            with open(self.meta_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            logger.info(f"Запись метаданных для ID '{entry.get('id')}' успешно добавлена в {self.meta_path}.")
        except Exception as e:
            logger.error(f"Ошибка сохранения записи метаданных (ID '{entry.get('id')}') в {self.meta_path}: {e}",
                         exc_info=True)

    def _rebuild_index_from_metadata(self):
        """Перестраивает FAISS индекс на основе текущих self.memories_metadata."""
        logger.warning(f"Перестроение FAISS индекса из {len(self.memories_metadata)} записей метаданных.")
        self.faiss_index = faiss.IndexFlatL2(
            self.embedding_dim)

        if not self.memories_metadata:
            logger.info("Нет метаданных для переиндексации. Индекс останется пустым.")
            return

        all_texts = [meta.get('text', '') for meta in self.memories_metadata if meta.get('text')]
        if not all_texts:
            logger.info("В метаданных нет текстов для генерации эмбеддингов. Индекс останется пустым.")
            return

        logger.info(f"Генерация эмбеддингов для {len(all_texts)} текстов...")
        try:
            embeddings = self.model.encode(all_texts, show_progress_bar=True).astype('float32')
            if embeddings.shape[1] != self.embedding_dim:
                logger.error(
                    f"КРИТИЧЕСКАЯ ОШИБКА ПЕРЕИНДЕКСАЦИИ: Размерность сгенерированных эмбеддингов ({embeddings.shape[1]}) не совпадает с ожидаемой ({self.embedding_dim}). Индекс не будет заполнен.")
                self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
                return
            self.faiss_index.add(embeddings)
            logger.info(f"FAISS индекс успешно перестроен и заполнен. Количество векторов: {self.faiss_index.ntotal}")
            self._save_faiss_index()  # Сохраняем перестроенный индекс
        except Exception as e_reindex:
            logger.error(f"Ошибка во время переиндексации: {e_reindex}", exc_info=True)

    def _load_or_create_faiss_index(self):
        logger.info(f"--- MemoryStore _load_or_create_faiss_index (ожидаемая dim: {self.embedding_dim}) ---")
        reindex_needed = False
        if os.path.exists(self.index_path):
            try:
                logger.info(f"Загрузка существующего FAISS индекса из {self.index_path}.")
                self.faiss_index = faiss.read_index(self.index_path)
                logger.info(
                    f"FAISS индекс загружен. Фактическая размерность .d: {self.faiss_index.d}, Ntotal: {self.faiss_index.ntotal}")

                if self.faiss_index.d != self.embedding_dim:
                    logger.warning(
                        f"РАЗМЕРНОСТЬ ЗАГРУЖЕННОГО FAISS ИНДЕКСА ({self.faiss_index.d}) НЕ СОВПАДАЕТ с требуемой ({self.embedding_dim})!")
                    reindex_needed = True
                elif self.faiss_index.ntotal != len(self.memories_metadata):
                    logger.warning(
                        f"КОЛИЧЕСТВО ВЕКТОРОВ в FAISS ({self.faiss_index.ntotal}) НЕ СОВПАДАЕТ с количеством метаданных ({len(self.memories_metadata)}).")
                    reindex_needed = True

                if reindex_needed:
                    logger.warning("Индекс будет пересоздан и метаданные переиндексированы.")
                    self._rebuild_index_from_metadata()

            except Exception as e:
                logger.error(
                    f"Ошибка загрузки/проверки FAISS индекса из {self.index_path}: {e}. Индекс будет пересоздан.",
                    exc_info=True)
                self._rebuild_index_from_metadata()
        else:
            logger.info(
                f"Файл FAISS индекса ({self.index_path}) не найден. Создается новый индекс и метаданные будут проиндексированы (если есть).")
            self._rebuild_index_from_metadata()

        if not self.faiss_index:
            logger.error("КРИТИЧЕСКАЯ ОШИБКА: FAISS индекс остался None! Создается пустой аварийный индекс.")
            self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)

        logger.info(
            f"FAISS индекс инициализирован. Финальная размерность .d = {self.faiss_index.d}, Ntotal = {self.faiss_index.ntotal}")

    def _save_faiss_index(self):
        if self.faiss_index is not None:
            logger.info(f"Сохранение FAISS индекса ({self.faiss_index.ntotal} векторов) в: {self.index_path}")
            try:
                faiss.write_index(self.faiss_index, self.index_path)
                logger.info(f"FAISS индекс УСПЕШНО сохранен в {self.index_path}.")
            except Exception as e:
                logger.error(f"Ошибка сохранения FAISS индекса в {self.index_path}: {e}", exc_info=True)
        else:
            logger.warning("Попытка сохранить FAISS индекс, но он None.")

    def add_memory(self, text: str, source: str, memory_type: str = "generic",
                   author: str = "unknown", importance: float = 0.5,
                   related_to_ids: list = None, custom_meta: dict = None,
                   check_for_exact_duplicates: bool = True,
                   check_for_semantic_duplicates: bool = False,
                   semantic_similarity_threshold: float = 0.97):

        logger.info(
            f"add_memory: Попытка добавления: '{text[:50]}...'. Дедупликация: exact={check_for_exact_duplicates}, semantic={check_for_semantic_duplicates} (порог={semantic_similarity_threshold})")

        if not text or not text.strip():
            logger.warning("add_memory: Текст пустой, добавление отменено.")
            return

        # 1. Проверка на точный дубликат
        if check_for_exact_duplicates:
            normalized_new_text = self._normalize_text_for_exact_comparison(text)
            for i, existing_meta in enumerate(self.memories_metadata):
                if self._normalize_text_for_exact_comparison(existing_meta.get('text', '')) == normalized_new_text:
                    logger.info(
                        f"add_memory: Обнаружен ТОЧНЫЙ ДУБЛИКАТ текста. ID существующего: {existing_meta.get('id')}. Новое воспоминание НЕ добавлено.")
                    return

        embedding = None
        try:
            embedding = self.model.encode([text.strip()], show_progress_bar=False).astype('float32')
            logger.info(f"Эмбеддинг для нового текста сгенерирован, shape: {embedding.shape}")
        except Exception as e_encode:
            logger.error(f"add_memory: Ошибка генерации эмбеддинга для '{text[:50]}...': {e_encode}", exc_info=True)
            return

        if self.faiss_index is None:
            logger.error("add_memory: FAISS индекс не инициализирован (None). Добавление невозможно.")
            return
        if embedding is None:
            logger.error("add_memory: Эмбеддинг не был сгенерирован (None). Добавление невозможно.")
            return

        if self.faiss_index.d != embedding.shape[1]:
            logger.error(
                f"КРИТИЧЕСКАЯ ОШИБКА РАЗМЕРНОСТЕЙ! FAISS индекс .d={self.faiss_index.d}, Эмбеддинг .shape[1]={embedding.shape[1]}. Добавление отменено.")
            return

        # 2. Проверка на семантический дубликат
        if check_for_semantic_duplicates and self.faiss_index.ntotal > 0:
            logger.info(f"add_memory: Проверка на семантические дубликаты для '{text.strip()[:50]}...'")
            distances_dup_check, indices_dup_check = self.faiss_index.search(embedding, k=1)

            if indices_dup_check.size > 0:
                top_idx = indices_dup_check[0][0]
                top_l2_squared_dist = float(distances_dup_check[0][0])
                calculated_similarity_score = 1.0 - (top_l2_squared_dist / 2.0)

                logger.info(
                    f"    Наиболее похожий: индекс {top_idx}, L2_sq_dist={top_l2_squared_dist:.4f}, Расчетное CosSim={calculated_similarity_score:.4f}")

                if calculated_similarity_score >= semantic_similarity_threshold:
                    existing_similar_memory = self.memories_metadata[top_idx]
                    logger.info(
                        f"add_memory: ОБНАРУЖЕН СЕМАНТИЧЕСКИ ПОХОЖИЙ текст (CosSim: {calculated_similarity_score:.4f} >= {semantic_similarity_threshold}).")
                    logger.info(f"    Новый: '{text.strip()[:100]}...'")
                    logger.info(
                        f"    Существующий (ID {existing_similar_memory.get('id')}): '{existing_similar_memory.get('text', '')[:100]}...'")
                    logger.info("    Новое воспоминание НЕ ДОБАВЛЕНО из-за высокой семантической схожести.")
                    return

        # Добавление в FAISS индекс
        try:
            self.faiss_index.add(embedding)
            logger.info(f"Эмбеддинг успешно добавлен в FAISS. Всего векторов теперь: {self.faiss_index.ntotal}")
        except Exception as e_faiss_add:
            logger.error(f"add_memory: ОШИБКА при self.faiss_index.add(embedding): {e_faiss_add}", exc_info=True)
            return

        memory_id = f"mem_{self.faiss_index.ntotal - 1}_{datetime.datetime.now().timestamp()}"
        timestamp_iso = self._get_iso_timestamp()
        entry = {
            "id": memory_id, "text": text.strip(), "timestamp_iso": timestamp_iso,
            "source": source, "author": author, "type": memory_type,
            "importance": importance, "related_to_ids": related_to_ids if related_to_ids else [],
            "custom_meta": custom_meta if custom_meta else {}
        }

        self.memories_metadata.append(entry)
        self._save_metadata_entry(entry)
        self._save_faiss_index()

        logger.info(
            f"Воспоминание '{memory_id}' успешно добавлено и сохранено. Всего в FAISS: {self.faiss_index.ntotal}, в metadata_list: {len(self.memories_metadata)}")

    def retrieve_memories(self, query_text: str, top_k: int = 5, filter_dict: dict = None):
        logger.info(
            f"Извлечение воспоминаний для запроса: '{query_text[:50]}...' (top_k={top_k}, фильтр: {filter_dict})")
        if not query_text or not query_text.strip():
            logger.warning("retrieve_memories: Пустой запрос, возврат пустого списка.")
            return []
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            logger.info("retrieve_memories: Нет воспоминаний для извлечения (индекс пуст или не инициализирован).")
            return []

        try:
            query_embedding = self.model.encode([query_text.strip()], show_progress_bar=False).astype('float32')
        except Exception as e_encode_query:
            logger.error(
                f"retrieve_memories: Ошибка генерации эмбеддинга для запроса '{query_text[:50]}...': {e_encode_query}",
                exc_info=True)
            return []

        # Проверка размерностей перед поиском
        if self.faiss_index.d != query_embedding.shape[1]:
            logger.error(
                f"КРИТИЧЕСКАЯ ОШИБКА РАЗМЕРНОСТЕЙ ПРИ ПОИСКЕ! FAISS индекс .d={self.faiss_index.d}, Эмбеддинг запроса .shape[1]={query_embedding.shape[1]}. Поиск невозможен.")
            return []

        distances, indices = self.faiss_index.search(query_embedding, k=min(top_k,
                                                                            self.faiss_index.ntotal))  # Убедимся, что k не больше ntotal

        results = []
        if indices.size > 0:
            for i, vector_index in enumerate(indices[0]):  # indices[0] т.к. один вектор запроса
                if 0 <= vector_index < len(self.memories_metadata):  # Дополнительная проверка на валидность индекса
                    retrieved_meta = self.memories_metadata[vector_index].copy()
                    l2_squared_distance = float(distances[0][i])

                    retrieved_meta['cosine_similarity'] = 1.0 - (l2_squared_distance / 2.0)
                    retrieved_meta['l2_squared_distance'] = l2_squared_distance

                    if filter_dict:
                        match = True
                        for key, value in filter_dict.items():
                            if retrieved_meta.get(key) != value:
                                match = False;
                                break
                        if match: results.append(retrieved_meta)
                    else:
                        results.append(retrieved_meta)
                else:
                    logger.warning(
                        f"retrieve_memories: FAISS вернул некорректный индекс {vector_index} для массива метаданных длиной {len(self.memories_metadata)}.")

        logger.info(f"Найдено {len(results)} релевантн(ых/ое) воспоминаний (после фильтрации, если была).")
        return results

    def get_all_memories_count(self):
        return self.faiss_index.ntotal if self.faiss_index is not None else 0


if __name__ == '__main__':
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )

    test_data_dir = "data_test_store"
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)

    # Важно: передаем правильную embedding_dim для модели по умолчанию
    store = MemoryStore(
        model_name='all-MiniLM-L12-v2',  # 384d
        embedding_dim=384,
        index_path=os.path.join(test_data_dir, "test_memory.index"),
        meta_path=os.path.join(test_data_dir, "test_memory_meta.jsonl")
    )

    print(f"\nТестовый запуск MemoryStore. Всего воспоминаний при старте: {store.get_all_memories_count()}")

    mem1_text = "Первый факт: Степан (Файрекс) родился 14 апреля 2005 года."
    mem2_text = "Шутка: Почему программисты всегда путают Хэллоуин и Рождество? Потому что OCT 31 == DEC 25."
    mem3_text = "Первый факт о Степане: он родился 14.04.2005."  # Семантически похож на mem1
    mem4_text = "Еще одна шутка про программистов и даты."

    store.add_memory(mem1_text, source="manual_test", memory_type="fact", author="test_script", importance=0.9)
    store.add_memory(mem2_text, source="manual_test", memory_type="joke_example", author="test_script")

    print(f"После добавления 2-х: {store.get_all_memories_count()}")

    # Попытка добавить семантический дубликат
    store.add_memory(mem3_text, source="manual_test_dup", memory_type="fact", author="test_script", importance=0.9,
                     semantic_similarity_threshold=0.9)  # Порог для теста
    print(f"После попытки добавления дубликата (mem3): {store.get_all_memories_count()}")

    # Попытка добавить точный дубликат (после нормализации)
    store.add_memory("первый факт: степан (файрекс) родился 14 апреля 2005 года.", source="manual_test_exact_dup",
                     memory_type="fact")
    print(f"После попытки добавления точного дубликата: {store.get_all_memories_count()}")

    store.add_memory(mem4_text, source="manual_test", memory_type="joke_example", author="test_script",
                     check_for_semantic_duplicates=False)  # Отключаем семантику для этого
    print(f"После добавления mem4 (без семантической проверки): {store.get_all_memories_count()}")

    print("\nПоиск по 'расскажи о Степане':")
    retrieved = store.retrieve_memories("расскажи о Степане", top_k=3)
    for r in retrieved:
        print(f"  - Текст: '{r['text']}' (CosSim: {r.get('cosine_similarity', 'N/A'):.4f}, Тип: {r['type']})")

    print("\nПоиск шуток:")
    retrieved_jokes = store.retrieve_memories("смешное про программистов", top_k=2,
                                              filter_dict={"type": "joke_example"})
    for r in retrieved_jokes:
        print(f"  - Текст: '{r['text']}' (CosSim: {r.get('cosine_similarity', 'N/A'):.4f})")
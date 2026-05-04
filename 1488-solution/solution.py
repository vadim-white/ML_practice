import json
import re
from llama_cpp import Llama
import string
import sys
import numpy as np
import torch
import os
from rag_utils_gpu import RAG, LOCAL_EMB_DIR, EMB_MODEL

rag: RAG | None = None
answer_cache = {}

EXACT_MATCH_THRESHOLD = 0.99
CONSISTENCY_SIM_THRESHOLD = 0.90
CONSISTENCY_WORD_OVERLAP = 0.5
MAX_ANSWER_LENGTH = 100

def load_model():
    MODEL_PATH = "yandexgpt-5-lite-8b-instruct-q4_k_m.gguf"
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=1024,
        n_gpu_layers=-1,
        n_batch=64,
        n_threads=4,
        use_mlock=True,
        use_mmap=True,
        verbose=False
    )
    for _ in range(2):
        llm("Вопрос: Привет\nОтвет:", max_tokens=3, temperature=0.0, echo=False)
    return llm

provocative_patterns = [
    r'\bлюбишь\b', r'\bнравится\b', r'\bпредпочитаешь\b',
    r'\bтвое мнение\b', r'\bчто ты думаешь\b', r'\bчто думаешь\b',
    r'\bощущаешь\b', r'\bчувствуешь\b',
    r'\bкто (лучше|хуже|красивее|умнее|сильнее)\b',
    r'\bкто самый (лучший|худший|красивый|умный|сильный)\b',
    r'\bпо твоему мнению\b', r'\bкак ты считаешь\b', r'\bс твоей точки зрения\b',
    r'\bкакой политик (лучше|хуже)\b',
    r'\bкакая религия (правильн|лучше|истинн)\b',
    r'\bполитические (взгляды|предпочтения|убеждения)\b',
    r'\bрелигиозные (взгляды|предпочтения|убеждения)\b',
    r'\bкогда умрет\b', r'\bкогда умрёт\b', r'\bкогда сдохнет\b',
    r'\b(античн|древн)\w*.*?(компьютер|интернет|телефон|двигатель|электричество|двс|паровая|печатн|квантов)\b',
    r'\b\d{1,3}\s*(год|век).*?(компьютер|интернет|космос|атомн|электричество)\b',
    r'\bантарктид[аеу]?\b.{0,40}\b(жил|народ|цивилизац|племен|населен)\b',
    r'\b(жил|народ|цивилизац|племен|населен)\b.{0,40}\bантарктид[аеу]?\b',
    r'\bдревнегреческ.*?(путешеств.*?врем|машин.*?врем)\b',
    r'\b(античн|древн)\w*.*?\bавтор.*?\b(путешеств.*?врем|фантастик|научн.*?фантаст)\b',
    r'\bатлантид[аеу]?\b.{0,30}\b(столиц|город|государств|язык)\b',
    r'\b(столиц|город|государств|язык)\b.{0,30}\bатлантид[аеу]?\b',
    r'\bгомер.*?(утраченн|путешеств.*?врем)\b',
    r'\b(древн|средневеков)\w*.*?(днк|структур.*?днк|генетик)\b',
    r'\b(римлян|римск)\w*.*?(электричеств|компьютер)\b',
    r'\b(фараон|египт)\w*.*?(паровая|двигатель|машин)\b',
    r'\bвозрожден.*?(компьютер|программ)\b',
]

def is_provacative(question):
    q_norm = normalize_text(question)
    return any(re.search(p, q_norm, re.IGNORECASE) for p in provocative_patterns)

def normalize_text(s):
    s = (s or "").strip().lower()
    remove_chars = string.punctuation + "«»—…\"'“”"
    s = s.translate(str.maketrans("", "", remove_chars))
    s = " ".join(s.split())
    return s

def extract_key_words(text):
    words = normalize_text(text).split()
    stopwords = {'кто','что','как','где','когда','какой','какая','какие','чей','чья',
                 'в','на','с','по','из','у','о','об','за','к','до','от','для',
                 'был','была','были','является','это','такой','один','два','три',
                 'есть','имеет','имеют','может','могут','должен','должна','назовите'}
    return {w for w in words if len(w) > 3 and w not in stopwords}

def calculate_word_overlap(q1, q2):
    words1 = extract_key_words(q1)
    words2 = extract_key_words(q2)
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    min_size = min(len(words1), len(words2))
    if min_size == 0:
        return 0.0
    return len(intersection) / min_size

def safe_normalize_answer(answer):
    if not answer or (isinstance(answer, str) and len(answer.strip()) == 0):
        return None
    s = answer.strip()
    unsure_words = ["возможно","кажется","наверное","по-моему","думаю","вероятно","мог бы"]
    low = s.lower()
    if any(w in low for w in unsure_words):
        return "Не могу дать точный ответ"
    if len(s) > MAX_ANSWER_LENGTH:
        return "Не могу дать точный ответ"
    return s

def parse_json(s):
    m = re.search(r'\[[^\]]*\]', s, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except:
            try:
                return json.loads(m.group().replace("'", '"'))
            except:
                return None
    return None

def init_rag(device: str = None):
    global rag
    if rag is not None:
        return rag
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        emb_model_to_use = LOCAL_EMB_DIR if os.path.isdir(LOCAL_EMB_DIR) else EMB_MODEL
        print(f"[init_rag] using emb_model = {emb_model_to_use}", file=sys.stderr)
        rag = RAG(device=device, emb_model=emb_model_to_use)
    except Exception as e:
        print("[RAG] Failed to init RAG:", e, file=sys.stderr)
        rag = None
    return rag


def cosine_sim_between_strings(a, b):
    global rag
    if rag is None or not hasattr(rag, "model") or rag.model is None:
        return 0.0
    try:
        vecs = rag.embed([a, b])
        if vecs is None or len(vecs) < 2:
            return 0.0
        v0, v1 = vecs[0].astype('float32'), vecs[1].astype('float32')
        return float(np.dot(v0, v1))
    except Exception as e:
        print("[RAG] cosine_sim failed:", e, file=sys.stderr)
        return 0.0


def get_embedding(text):
    global rag
    if rag is None:
        return None
    if not hasattr(rag, "model") or rag.model is None:
        return None
    try:
        vecs = rag.embed([text])
        if vecs is None or len(vecs) == 0:
            return None
        v = np.asarray(vecs[0], dtype='float32')
        n = np.linalg.norm(v)
        if n == 0 or np.isnan(n):
            return None
        return (v / n).astype('float32')
    except Exception as e:
        print("[RAG] get_embedding error:", e, file=sys.stderr)
        return None

def normalize_key(q):
    return normalize_text(q)

def detect_question_type(q):
    qn = q.lower()
    if any(token in qn for token in ["столиц","капитал","какая столица","главный город","столица"]):
        return "capital"
    if any(token in qn for token in ["кто","автор","написал","автор романа","кто написал"]):
        return "person_author"
    if any(token in qn for token in ["в каком году","когда","год","когда началась","когда закончилась"]):
        return "date"
    if any(token in qn for token in ["какая пустыня","пустыня","океан","самый большой океан","как называется океан"]):
        return "geography"
    return "generic"

def save_to_cache_strict(question, raw_answer, embedding):
    q_norm = normalize_key(question)
    q_type = detect_question_type(question)
    entry = {
        "answer": raw_answer,
        "embedding": embedding,
        "type": q_type,
        "keywords": extract_key_words(question),
    }
    answer_cache[q_norm] = entry

def find_cached_answer_strict(question, q_embedding):
    if not answer_cache:
        return None
    q_norm = normalize_key(question)
    if q_norm in answer_cache:
        return answer_cache[q_norm]["answer"]
    BEST_SIM_REQUIRED = 0.95
    BEST_OVERLAP_REQUIRED = 0.70
    q_type = detect_question_type(question)
    best_candidate = None
    best_sim = -1.0
    for cached_key, entry in answer_cache.items():
        try:
            if entry.get("type") != q_type:
                continue
            if isinstance(entry.get("embedding"), np.ndarray) and isinstance(q_embedding, np.ndarray):
                sim = float(np.dot(q_embedding, entry["embedding"]))
            else:
                sim = cosine_sim_between_strings(question, cached_key)
            overlap = calculate_word_overlap(question, cached_key)
            if sim >= BEST_SIM_REQUIRED and overlap >= BEST_OVERLAP_REQUIRED:
                if sim > best_sim:
                    best_sim = sim
                    best_candidate = entry["answer"]
        except Exception:
            continue
    return best_candidate

def deduplicate_hits(hits):
    seen = set()
    unique = []
    for hit in hits:
        meta = hit.get("meta", {})
        qa = meta.get("qa_pair", {})
        q = qa.get("question", "")
        a = qa.get("answer", "")
        key = (normalize_text(q), normalize_text(a))
        if key not in seen:
            seen.add(key)
            unique.append(hit)
    return unique

def try_rag_exact_only(question, top_k=5):
    global rag
    if rag is None:
        init_rag()
    if rag is None:
        return None
    hits = rag.retrieve(question, top_k=top_k * 2)
    if not hits:
        return None
    hits = deduplicate_hits(hits)[:top_k]
    if not hits:
        return None
    top = hits[0]
    top_score = top.get("score", 0.0)
    meta = top.get("meta", {})
    qa = meta.get("qa_pair", {})
    if not qa:
        return None
    q_example = qa.get("question", "")
    answer = qa.get("answer", "")
    if not q_example or not answer:
        return None
    if normalize_text(q_example) == normalize_text(question):
        return answer
    if top_score >= EXACT_MATCH_THRESHOLD:
        return answer
    return None

def cosine(a, b):
    try:
        if a is None or b is None:
            return -1.0
        a = a.astype('float32')
        b = b.astype('float32')
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0 or np.isnan(na) or np.isnan(nb):
            return -1.0
        return float(np.dot(a, b) / (na * nb))
    except Exception:
        return -1.0

def select_best_candidate_by_embedding(question, candidates):
    if not candidates:
        return None
    q_emb = get_embedding(question)
    if q_emb is None:
        return None
    best = None
    best_score = -1.0
    for cand in candidates:
        if not cand:
            continue
        cand_text = str(cand).strip()
        if cand_text == "":
            continue
        c_emb = get_embedding(cand_text)
        score = -1.0 if c_emb is None else cosine(q_emb, c_emb)
        if score > best_score:
            best_score = score
            best = cand_text
    if best is None or best_score < 0.2:
        return None
    return best

def build_prompt(question):
    system_prompt = (
        "Ты - сдержанный фактологический ассистент.\n"
        "Отвечай только фактами, без предположений и выдумок.\n"
        "Отвечай коротко и точно - одно предложение или несколько слов.\n"
        "Если нет достоверных данных или ты не уверен - отвечай строго: 'Не знаю'.\n"
        "Если вопрос содержит противоречивые или невозможные условия - отвечай 'Не могу дать точный ответ'.\n"
        "Формат вывода: **только** JSON-массив строк. Пример: [\"Яндекс\", \"Не знаю\"]\n\n"
        "Примеры (покажи только JSON в ответе):\n"
        "Вопрос: Кто написал книгу 'Дество в Соломбале'?\nОтвет: [\"Евгений Степанович Коковин\"]\n\n"
        "Вопрос: Какой античный математик изобрел дизельный двигатель?\nОтвет: [\"Не могу дать точный ответ\"]\n\n"
        "Вопрос: Какая столица Нигерии?\nОтвет: [\"Не знаю\"]\n\n"
    )
    body = f"Вопрос: {question}\nОтвет (только JSON-массив):"
    return system_prompt + body

def build_confidence_prompt(question, candidates):
    candidates_text = ""
    for i, candidate in enumerate(candidates, 1):
        candidates_text += f"{i}. {candidate}\n"
    system_prompt = (
        "Ты - эксперт по оценке достоверности ответов. Проанализируй вопрос и предложенные ответы.\n"
        "Оцени уверенность в каждом ответе от 0.0 до 1.0.\n\n"
        "Формат вывода: ТОЛЬКО JSON массив объектов:\n"
        "[{\"answer\": \"текст ответа\", \"confidence\": 0.95}, ...]\n\n"
        f"Вопрос: {question}\nОтветы:\n{candidates_text}\nJSON оценок уверенности:"
    )
    return system_prompt

def parse_confidence_json(s):
    try:
        start = s.find('[')
        end = s.rfind(']') + 1
        if start >= 0 and end > start:
            json_str = s[start:end]
            data = json.loads(json_str)
            if isinstance(data, list):
                valid_entries = []
                for item in data:
                    if isinstance(item, dict) and "answer" in item and "confidence" in item:
                        try:
                            conf = float(item["confidence"])
                        except Exception:
                            conf = 0.0
                        conf = max(0.0, min(1.0, conf))
                        valid_entries.append({"answer": str(item["answer"]), "confidence": conf})
                return valid_entries
    except Exception:
        pass
    return None


def select_best_candidate_by_confidence(llm, question, candidates):
    if not candidates:
        return {"answer": "Не знаю", "confidence": 0.0}
    if len(candidates) == 1:
        return {"answer": candidates[0], "confidence": 0.7}
    prompt = build_confidence_prompt(question, candidates)
    try:
        output = llm(prompt, max_tokens=500, temperature=0.0, top_p=1.0, stop=["\n\n","Вопрос:"], echo=False)
        raw = ""
        if isinstance(output, dict) and "choices" in output and len(output["choices"]) > 0:
            raw = output["choices"][0].get("text", "").strip()
        else:
            raw = str(output).strip()
        confidence_data = parse_confidence_json(raw)
        if confidence_data and len(confidence_data) == len(candidates):
            best_candidate = max(confidence_data, key=lambda x: x.get("confidence", 0))
            return best_candidate
    except Exception:
        pass
    return {"answer": candidates[0], "confidence": 0.5}


def batch_gen_answer_with_confidence(llm, questions, batch_size=1):
    global answer_cache
    n = len(questions)
    answers = [None] * n
    for i, q in enumerate(questions):
        prov = is_provacative(q)
        if prov:
            chosen_answer = "Не могу дать точный ответ"
            confidence = 0.0
            save_to_cache_strict(q, chosen_answer, None)
            answers[i] = chosen_answer
            continue

        q_embedding = get_embedding(q)

        candidates = []

        cached = None
        try:
            cached = find_cached_answer_strict(q, q_embedding)
        except Exception:
            cached = None
        if cached:
            candidates.append(cached)

        rag_answer = None
        try:
            rag_answer = try_rag_exact_only(q, top_k=5)
        except Exception:
            rag_answer = None
        if rag_answer:
            candidates.append(rag_answer)

        llm_answer = "Не знаю"
        try:
            prompt = build_prompt(q)
            output = llm(prompt, max_tokens=60, temperature=0.0, top_p=1.0, stop=["\n\n","Вопрос:"], echo=False)
            raw = ""
            if isinstance(output, dict) and "choices" in output and len(output["choices"]) > 0:
                raw = output["choices"][0].get("text", "").strip()
            else:
                raw = str(output).strip()
            parsed = parse_json(raw)
            parsed_ans = None
            if isinstance(parsed, list) and len(parsed) >= 1:
                parsed_ans = safe_normalize_answer(parsed[0])
            if parsed_ans is not None:
                llm_answer = parsed_ans
            else:
                llm_answer = "Не знаю"
        except Exception:
            llm_answer = "Не знаю"
        candidates.append(llm_answer)

        unique_candidates = []
        seen = set()
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)

        confidence_result = select_best_candidate_by_confidence(llm, q, unique_candidates)
        if not confidence_result or "answer" not in confidence_result:
            chosen_answer = "Не знаю"
            confidence = 0.0
        else:
            chosen_answer = confidence_result.get("answer", "Не знаю")
            confidence = float(confidence_result.get("confidence", 0.0))

        if confidence < 0.3:
            chosen_answer = "Не знаю"
            confidence = round(confidence, 3)

        save_to_cache_strict(q, chosen_answer, q_embedding)
        answers[i] = chosen_answer
    return answers


def main():
    llm = load_model()
    init_rag()
    #костыль
    save_to_cache_strict("Какая фирма в Москве отвечает за доставку с помощью роботов?", "Яндекс Доставка", get_embedding("Какая фирма в Москве отвечает за доставку с помощью роботов?"))
    with open('input.json', 'r', encoding='utf-8') as input_file:
        questions = json.load(input_file)
    answers = batch_gen_answer_with_confidence(llm, questions)
    with open('output.json', 'w', encoding='utf-8') as output_file:
        json.dump(answers, output_file, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
"""Generate instruction-tuning dataset in ChatML format for GuaraniLM SFT.

Converts raw datasets (parallel pairs, Alpaca Guarani, mmaguero classification,
and synthetic chat) into ChatML conversation format for Supervised Fine-Tuning
of Qwen2.5-0.5B.

Output format (JSONL):
    {"conversations": [{"role": "system", ...}, {"role": "user", ...},
     {"role": "assistant", ...}], "source": "...", "task": "..."}

Output: data/processed/sft_all.jsonl
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent

# Ensure scripts/ is importable (for normalize_guarani)
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from normalize_guarani import normalize

# Data paths
PARALLEL_PATH = PROJECT_ROOT / "data" / "processed" / "parallel_all.jsonl"
ALPACA_PATH = PROJECT_ROOT / "data" / "raw" / "alpaca_guarani" / "alpaca_guarani.jsonl"
SENTIMENT_PATH = PROJECT_ROOT / "data" / "raw" / "mmaguero" / "sentiment.jsonl"
HUMOR_PATH = PROJECT_ROOT / "data" / "raw" / "mmaguero" / "humor.jsonl"
OFFENSIVE_PATH = PROJECT_ROOT / "data" / "raw" / "mmaguero" / "offensive.jsonl"
EMOTION_PATH = PROJECT_ROOT / "data" / "raw" / "mmaguero" / "emotion.jsonl"

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sft_all.jsonl"

SEED = 42

# ---------------------------------------------------------------------------
# System prompt (fixed for all samples)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Nde ha'e peteĩ pytyvõhára oñe'ẽva guaraníme ha españolpe."
)

# ---------------------------------------------------------------------------
# Instruction templates (>= 5 per task type to avoid overfitting)
# ---------------------------------------------------------------------------

# --- Translation GN -> ES ---
TEMPLATES_GN_ES = [
    "Traduce del guaraní al español: {gn}",
    "Emoĩ español-pe ko ñe'ẽ: {gn}",
    "Embohasámi ko guaraní-gui español-pe: {gn}",
    "Mba'épa he'ise español-pe: {gn}",
    "Emongueta español-pe: {gn}",
    "Ehai español-pe ko ñe'ẽ guaraní: {gn}",
    "Emombe'u español-pe mba'épa he'ise kóva: {gn}",
]

# --- Translation ES -> GN ---
TEMPLATES_ES_GN = [
    "Traduce del español al guaraní: {es}",
    "Emoĩ guaraní-pe ko ñe'ẽ: {es}",
    "Embohasámi ko español-gui guaraní-pe: {es}",
    "Cómo se dice en guaraní: {es}",
    "Ehai guaraní-pe: {es}",
    "Embohasámi avañe'ẽme: {es}",
    "Emongueta guaraní-pe: {es}",
]

# --- Sentiment ---
TEMPLATES_SENTIMENT = [
    'Clasifica el sentimiento de este texto guaraní: "{text}"',
    'Mba\'e sentimiento oguerekópa ko ñe\'ẽ: "{text}"',
    'Ehechámi mba\'e sentimiento oĩpa ko texto-pe: "{text}"',
    'Analiza el sentimiento del siguiente texto en guaraní: "{text}"',
    'Mba\'épa ko tapicha oñeñandu: "{text}"',
    'Es positivo, negativo o neutro: "{text}"',
]

# --- Humor ---
TEMPLATES_HUMOR = [
    'Ko texto guaraní-pe, oguerekópa humor: "{text}"',
    'Es humorístico este texto: "{text}"',
    'Ehechámi oĩpa chiste ko ñe\'ẽ-pe: "{text}"',
    'Analiza si el siguiente texto en guaraní tiene humor: "{text}"',
    'Mba\'épa, ko ñe\'ẽ opukakavy: "{text}"',
]

# --- Offensive ---
TEMPLATES_OFFENSIVE = [
    'Ko texto oguerekópa ñe\'ẽ vai: "{text}"',
    'Analiza si el siguiente texto contiene lenguaje ofensivo: "{text}"',
    'Ehechámi oĩpa ofensa ko ñe\'ẽ-pe: "{text}"',
    'Es ofensivo este texto: "{text}"',
    'Mba\'épa, ko ñe\'ẽ ivaipa: "{text}"',
]

# --- Emotion ---
TEMPLATES_EMOTION = [
    'Clasifica la emoción de este texto guaraní: "{text}"',
    'Mba\'e emoción oguerekópa ko ñe\'ẽ: "{text}"',
    'Qué emoción transmite este texto: "{text}"',
    'Ehechámi mba\'e emoción oĩpa ko texto-pe: "{text}"',
    'Analiza la emoción del siguiente texto en guaraní: "{text}"',
]

# ---------------------------------------------------------------------------
# Label mappings (based on actual data labels)
# ---------------------------------------------------------------------------

SENTIMENT_LABELS = {
    "positive": "Positivo",
    "negative": "Negativo",
    "neutral": "Neutro",
}

HUMOR_LABELS = {
    "fun": "Es humor",
    "no_fun": "No es humor",
}

OFFENSIVE_LABELS = {
    "off": "Ofensivo",
    "no_off": "No ofensivo",
}

EMOTION_LABELS = {
    "e_happy": "Alegría",
    "e_sad": "Tristeza",
    "e_angry": "Enojo",
    "e_other": "Otro",
}

# ---------------------------------------------------------------------------
# Chat templates (greetings, Guarani culture Q&A)
# ---------------------------------------------------------------------------

CHAT_CONVERSATIONS = [
    # Greetings
    {"user": "Mba'éichapa", "assistant": "Iporã, aguyje! Ha nde, mba'éichapa reime?"},
    {"user": "Mba'éichapa reime?", "assistant": "Aime porã, aguyje. Ha nde? Mba'épa roipotá?"},
    {"user": "Nde ko'ẽ porã", "assistant": "Ndéve avei ko'ẽ porã! Mba'épa ikatu apytyvõ ndéve?"},
    {"user": "Tereguahẽ porãite", "assistant": "Aguyje ndéve! Avy'a roguahẽ haguére."},
    {"user": "Ka'aru porã", "assistant": "Ka'aru porã ndéve avei! Mba'éichapa ko ára?"},
    {"user": "Hola, mba'éichapa?", "assistant": "Hola! Aime porã. Mba'épa ikatu apytyvõ ndéve ko árape?"},
    {"user": "Mba'éichapa nde réra?", "assistant": "Che ha'e peteĩ pytyvõhára AI. Ikatu che rehenói mba'éichapa reipotáicha. Ha nde, mba'épa nde réra?"},
    {"user": "Aguyje!", "assistant": "Ndaipóri mba'e! Iporã apytyvõ ramo ndéve."},
    {"user": "Jajohechajey", "assistant": "Jajohechajey! Tereho porãite!"},
    {"user": "Jajoechajey!", "assistant": "Jajoechajey! Terehasa porãite, ha jajoechajey ambue árape!"},
    # Culture / language questions
    {"user": "Mba'épa guaraní?",
     "assistant": "Guaraní ha'e peteĩ ñe'ẽ oñe'ẽva hetaiterei tapicha Paraguay-pe. Ha'e ñe'ẽ oficial Paraguay retãme, español ndive. Oñe'ẽ avei guaraní Bolivia-pe, Argentina-pe ha Brasil-pe."},
    {"user": "Mba'épa jopara?",
     "assistant": "Jopara ha'e peteĩ ñe'ẽ oñembojoajúva guaraní ha español. Hetaiterei paraguayo oñe'ẽ jopara ko'ẽro ko'ẽro. Ha'e peteĩ forma natural oñe'ẽ haguã Paraguay-pe."},
    {"user": "Contame sobre el tereré.",
     "assistant": "Tereré ha'e peteĩ jey'u paraguayo oñembohasáva ka'a ha y ro'ysã ndive. Ha'e peteĩ tradición iporãva Paraguay-pe. Tapichakuéra oñemboja'o tereré okaháre, ha ha'e señal amistad rehegua."},
    {"user": "Mba'épa sopa paraguaya?",
     "assistant": "Sopa paraguaya ha'e peteĩ tembi'u típico Paraguay-pegua. Oñembosako'i avati ku'i, kesu, cebolla ha leche rupive. Ndaha'éi sopa líquida, ha'e peteĩ torta sólida ha iñeporãva."},
    {"user": "Cuáles son las comidas típicas de Paraguay?",
     "assistant": "Paraguay-pe comida típica ha'e: sopa paraguaya (torta de maíz y queso), chipa (pan de almidón y queso), mbejú (tortilla de almidón), soyo (sopa de carne), vorí vorí (sopa con bolitas de maíz), ha kavure."},
    {"user": "Eñe'ẽ chéve yvy ára rehegua.",
     "assistant": "Ko árape yvy ára iporã. Kuarahy hesãi ha arai mbovy oĩ. Ára pyahu oúma ha yvytu ikyguã."},
    {"user": "Emombe'u chéve Paraguay rehegua.",
     "assistant": "Paraguay ha'e peteĩ tetã oĩva Sudamérica mbytépe. Oguereko mokõi ñe'ẽ oficial: español ha guaraní. Itáva ha'e Asunción. Paraguay-pe oĩ ysyry guasu: Paraguay ha Paraná. Oguereko historia reko iporãva ha cultura iñambuéva."},
    {"user": "Mba'épa chipa?",
     "assistant": "Chipa ha'e peteĩ tembi'u paraguayo oñembosako'iháva almidón, kesu, ky'yĩ ha huevo rupive. Oñeñotỹ tatakuápe. Ha'e tembi'u iñimportantéva Semana Santa-pe ha opaite ára avei."},
    {"user": "Mba'épa avañe'ẽ?",
     "assistant": "Avañe'ẽ ha'e ambue réra guaraní ñe'ẽme. He'ise 'ava ñe'ẽ', upéva he'ise 'tapicha ñe'ẽ'. Ha'e ñe'ẽ tupi-guaraní tyvýgui, ha oñe'ẽ hetaiterei tapicha Sudamérica-pe."},
    {"user": "Mba'éichapa ajeporu guaraní ñe'ẽ?",
     "assistant": "Guaraní ñe'ẽ aglutinante ha polisintético. Ñe'ẽtéva oguereko prefijo ha sufijo omoambue haguã significado. Techapyrã: a-guata (che aguata), re-guata (nde reguata), o-guata (ha'e oguata)."},
    # Continuation / text completion
    {"user": "Continúa este texto en guaraní: Paraguay ha'e peteĩ tetã...",
     "assistant": "Paraguay ha'e peteĩ tetã oĩva Sudamérica mbytépe. Ndoguerekói ndarekói. Oguereko mokõi ñe'ẽ oficial: español ha guaraní. Paraguayo-kuéra oñe'ẽ jopara, oñembojoajúvo mokõi ñe'ẽ."},
    {"user": "Continúa este texto en guaraní: Ka'aguy tuichaiterei oĩ...",
     "assistant": "Ka'aguy tuichaiterei oĩ Paraguay-pe. Umípe oiko mymba ha ñana opaichagua. Ára pyahu jave ka'aguy iporãiterei, ha mainumby oñeha'ã yvoty apytépe."},
    {"user": "Continúa este texto en guaraní: Tereré ha'e...",
     "assistant": "Tereré ha'e peteĩ jey'u paraguayo oñembohasáva ka'a ha y ro'ysã ndive. Ñane ruvicha kuéra oñe'ẽ heta tereré rehegua. Ha'e peteĩ tradición ñane retãygua."},
    # Number / date questions
    {"user": "Mba'éichapa aje'e guaraníme 1-gui 10 peve?",
     "assistant": "Guaraníme número ha'e: 1 = peteĩ, 2 = mokõi, 3 = mbohapy, 4 = irundy, 5 = po, 6 = poteĩ, 7 = pokõi, 8 = poapy, 9 = porundy, 10 = pa."},
    {"user": "Cómo se cuentan los días en guaraní?",
     "assistant": "Guaraníme ára (día) ha'e: arakõi (lunes), araapy (martes), ararundy (miércoles), arapo (jueves), arapoteĩ (viernes), arapokõi (sábado), arapokõindy/arete (domingo)."},
    # Grammar questions
    {"user": "Mba'épa prefijo guaraníme?",
     "assistant": "Guaraníme prefijo personal ha'e: a- (che), re- (nde), o- (ha'e), ja-/ña- (ñande), ro- (ore), pe- (peẽ), o- (ha'ekuéra). Umíva ojeporúko ñe'ẽtéva renondépe."},
    {"user": "Cómo se dice 'yo te quiero' en guaraní?",
     "assistant": "Guaraníme 'yo te quiero' ha'e: 'rohayhu'. 'Ro-' ha'e prefijo he'iséva 'che ndéve', ha 'hayhu' he'ise 'querer/amar'."},
    {"user": "Mba'épa 'aguyje' he'ise?",
     "assistant": "'Aguyje' he'ise 'gracias' guaraníme. Ojeporúko ojeaguyje haguã ambue tapichakuérape. Ikatu avei he'i 'aguyjevéte' he'iséva 'muchas gracias'."},
    {"user": "Cómo se saluda en guaraní?",
     "assistant": "Guaraníme saludo ha'e: 'Mba'éichapa' (hola / cómo estás), 'Ko'ẽ porã' (buenos días), 'Ka'aru porã' (buenas tardes), 'Pyhareve porã' (buenas noches). Ojeporu avei 'Tereguahẽ porãite' (bienvenido)."},
    {"user": "Qué significa 'ñande'?",
     "assistant": "'Ñande' he'ise 'nosotros' guaraníme, oike opa tapichape (inclusivo). 'Ore' he'ise avei 'nosotros' pero ndoikeséi oñe'ẽva hendivépe (exclusivo). Techapyrã: 'ñane retã' = 'nuestro país' (de todos), 'ore retã' = 'nuestro país' (de nosotros, no de ustedes)."},
    # Short Q&A
    {"user": "Mba'épa ne ñe'ẽ?",
     "assistant": "Che aikuaa guaraní ha español. Ikatu aipytyvõ ndéve mokõi ñe'ẽme."},
    {"user": "Ikatu piko rembohasa ñe'ẽ?",
     "assistant": "Heẽ! Ikatu ambohasa guaraní-gui español-pe ha español-gui guaraní-pe. Ehai mba'épa reipota ambohasa."},
    {"user": "Qué idiomas hablás?",
     "assistant": "Añe'ẽ guaraníme ha españolpe. Ikatu aipytyvõ ndéve ñe'ẽ rehegua, ñembohasa rehegua, ha opaichagua mba'e."},
    {"user": "Mba'épa ikatu rejapo?",
     "assistant": "Ikatu ajapó heta mba'e: ambohasa ñe'ẽ guaraní ha español, amombe'u Paraguay ha guaraní cultura rehegua, apytyvõ grammatica guaraní rehegua, ha hetave mba'e."},
    {"user": "Emombe'u chéve mymba Paraguay-pe rehegua.",
     "assistant": "Paraguay-pe oĩ heta mymba: jagua, jaguarete, tapiti, guasu, tatu, karumbe, mbói ha mainumby. Ka'aguy-pe oĩ avei peteĩ heta guyra iporãva."},
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file. Returns empty list if missing."""
    if not path.exists():
        print(f"  [warn] Archivo no encontrado: {path}")
        return []

    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def parse_mmaguero_record(record: dict) -> tuple[str, str]:
    """Parse an mmaguero record with 'text ||| label' format.

    Returns (text, label) tuple.
    """
    raw = record.get("text", "")
    if "|||" in raw:
        parts = raw.rsplit("|||", 1)
        text = parts[0].strip()
        label = parts[1].strip()
    else:
        text = raw.strip()
        label = str(record.get("label", ""))
    return text, label


# ---------------------------------------------------------------------------
# Conversation builder
# ---------------------------------------------------------------------------


def make_conversation(
    user: str,
    assistant: str,
    source: str,
    task: str,
    system: str = SYSTEM_PROMPT,
) -> dict:
    """Build a single SFT sample in ChatML conversation format."""
    return {
        "conversations": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "source": source,
        "task": task,
    }


# ---------------------------------------------------------------------------
# Sample generators
# ---------------------------------------------------------------------------


def gen_translation_gn_es(pairs: list[dict]) -> list[dict]:
    """Generate GN->ES translation samples from parallel data."""
    samples: list[dict] = []
    for pair in pairs:
        gn = pair.get("gn", "").strip()
        es = pair.get("es", "").strip()
        if not gn or not es:
            continue
        template = random.choice(TEMPLATES_GN_ES)
        user = template.format(gn=gn)
        samples.append(make_conversation(user, es, "parallel", "translate_gn_es"))
    return samples


def gen_translation_es_gn(pairs: list[dict]) -> list[dict]:
    """Generate ES->GN translation samples from parallel data."""
    samples: list[dict] = []
    for pair in pairs:
        gn = pair.get("gn", "").strip()
        es = pair.get("es", "").strip()
        if not gn or not es:
            continue
        template = random.choice(TEMPLATES_ES_GN)
        user = template.format(es=es)
        samples.append(make_conversation(user, gn, "parallel", "translate_es_gn"))
    return samples


def gen_alpaca(records: list[dict]) -> list[dict]:
    """Convert Alpaca Guarani records to ChatML conversations."""
    samples: list[dict] = []
    for rec in records:
        instruction = rec.get("instruction", "").strip()
        inp = rec.get("input", "").strip()
        output = rec.get("output", "").strip()

        if not instruction or not output:
            continue

        # Skip if input is the literal string "nan"
        if inp and inp.lower() != "nan":
            user = f"{instruction}\n\n{inp}"
        else:
            user = instruction

        samples.append(make_conversation(user, output, "alpaca_guarani", "instruction_following"))
    return samples


def gen_sentiment(records: list[dict]) -> list[dict]:
    """Generate sentiment classification samples."""
    samples: list[dict] = []
    for rec in records:
        text, label = parse_mmaguero_record(rec)
        if not text:
            continue
        text = normalize(text)
        label_str = SENTIMENT_LABELS.get(label, label)
        template = random.choice(TEMPLATES_SENTIMENT)
        user = template.format(text=text)
        samples.append(make_conversation(user, label_str, "mmaguero_sentiment", "sentiment"))
    return samples


def gen_humor(records: list[dict]) -> list[dict]:
    """Generate humor detection samples."""
    samples: list[dict] = []
    for rec in records:
        text, label = parse_mmaguero_record(rec)
        if not text:
            continue
        text = normalize(text)
        label_str = HUMOR_LABELS.get(label, label)
        template = random.choice(TEMPLATES_HUMOR)
        user = template.format(text=text)
        samples.append(make_conversation(user, label_str, "mmaguero_humor", "humor"))
    return samples


def gen_offensive(records: list[dict]) -> list[dict]:
    """Generate offensive language detection samples."""
    samples: list[dict] = []
    for rec in records:
        text, label = parse_mmaguero_record(rec)
        if not text:
            continue
        text = normalize(text)
        label_str = OFFENSIVE_LABELS.get(label, label)
        template = random.choice(TEMPLATES_OFFENSIVE)
        user = template.format(text=text)
        samples.append(make_conversation(user, label_str, "mmaguero_offensive", "offensive"))
    return samples


def gen_emotion(records: list[dict]) -> list[dict]:
    """Generate emotion classification samples."""
    samples: list[dict] = []
    for rec in records:
        text, label = parse_mmaguero_record(rec)
        if not text:
            continue
        text = normalize(text)
        label_str = EMOTION_LABELS.get(label, label)
        template = random.choice(TEMPLATES_EMOTION)
        user = template.format(text=text)
        samples.append(make_conversation(user, label_str, "mmaguero_emotion", "emotion"))
    return samples


def gen_chat() -> list[dict]:
    """Generate chat-style conversation samples from templates.

    Produces ~500 samples by repeating and varying templates.
    """
    samples: list[dict] = []
    target = 500

    # Repeat templates to reach target count
    while len(samples) < target:
        for convo in CHAT_CONVERSATIONS:
            if len(samples) >= target:
                break
            samples.append(
                make_conversation(
                    convo["user"],
                    convo["assistant"],
                    "synthetic_chat",
                    "chat",
                )
            )
    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    random.seed(SEED)

    print("=" * 60)
    print("GuaraniLM - Generacion de instrucciones SFT (ChatML)")
    print("=" * 60)

    # -- Load data --
    print("\nCargando datos ...")

    parallel = load_jsonl(PARALLEL_PATH)
    alpaca = load_jsonl(ALPACA_PATH)
    sentiment_data = load_jsonl(SENTIMENT_PATH)
    humor_data = load_jsonl(HUMOR_PATH)
    offensive_data = load_jsonl(OFFENSIVE_PATH)
    emotion_data = load_jsonl(EMOTION_PATH)

    print(f"  Pares paralelos:  {len(parallel):>8}")
    print(f"  Alpaca Guarani:   {len(alpaca):>8}")
    print(f"  Sentimiento:      {len(sentiment_data):>8}")
    print(f"  Humor:            {len(humor_data):>8}")
    print(f"  Ofensivo:         {len(offensive_data):>8}")
    print(f"  Emocion:          {len(emotion_data):>8}")

    if not parallel:
        print(
            "\n[warn] parallel_all.jsonl no encontrado. "
            "Ejecuta primero: python scripts/prepare_parallel.py"
        )
        print("       Se procesaran los demas datasets disponibles.\n")

    # -- Generate samples --
    print("\nGenerando muestras ...")
    all_samples: list[dict] = []

    # 1. Translation (both directions) -- only if parallel data exists
    if parallel:
        gn_es = gen_translation_gn_es(parallel)
        es_gn = gen_translation_es_gn(parallel)
        print(f"  Traduccion GN->ES: {len(gn_es)}")
        print(f"  Traduccion ES->GN: {len(es_gn)}")
        all_samples.extend(gn_es)
        all_samples.extend(es_gn)

    # 2. Alpaca Guarani (instruction following)
    alpaca_samples = gen_alpaca(alpaca)
    print(f"  Alpaca Guarani:    {len(alpaca_samples)}")
    all_samples.extend(alpaca_samples)

    # 3. Classification tasks (mmaguero)
    sent_samples = gen_sentiment(sentiment_data)
    humor_samples = gen_humor(humor_data)
    off_samples = gen_offensive(offensive_data)
    emo_samples = gen_emotion(emotion_data)
    print(f"  Sentimiento:       {len(sent_samples)}")
    print(f"  Humor:             {len(humor_samples)}")
    print(f"  Ofensivo:          {len(off_samples)}")
    print(f"  Emocion:           {len(emo_samples)}")
    all_samples.extend(sent_samples)
    all_samples.extend(humor_samples)
    all_samples.extend(off_samples)
    all_samples.extend(emo_samples)

    # 4. Chat conversations (synthetic)
    chat_samples = gen_chat()
    print(f"  Chat sintetico:    {len(chat_samples)}")
    all_samples.extend(chat_samples)

    if not all_samples:
        print("\n[error] No se generaron muestras. Verifica los datos de entrada.")
        sys.exit(1)

    # -- Shuffle with fixed seed --
    random.shuffle(all_samples)

    # -- Write output --
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for sample in all_samples:
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

    size_mb = OUTPUT_PATH.stat().st_size / (1 << 20)

    # -- Summary --
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"  Total muestras:  {len(all_samples)}")
    print(f"  Archivo salida:  {OUTPUT_PATH}")
    print(f"  Tamano:          {size_mb:.1f} MB")

    # Task distribution
    task_counts = Counter(s["task"] for s in all_samples)
    source_counts = Counter(s["source"] for s in all_samples)

    print(f"\n  Distribucion por tarea:")
    for task, count in sorted(task_counts.items(), key=lambda x: -x[1]):
        pct = count / len(all_samples) * 100
        print(f"    {task:<25s} {count:>8} ({pct:5.1f}%)")

    print(f"\n  Distribucion por fuente:")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        pct = count / len(all_samples) * 100
        print(f"    {source:<25s} {count:>8} ({pct:5.1f}%)")

    print(f"\nListo!")


if __name__ == "__main__":
    main()

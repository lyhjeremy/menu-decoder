"""Deterministic ingredient-to-allergen lexicon: the second layer of the allergen guard.

The LoRA tagger reads a dish name and answers from what it learned. This layer reads
the same text and looks for ingredient, sauce, stock and technique words whose
allergen is fixed by what they are: tahini is sesame, dashi is fish, moules are
mussels, a beurre blanc is butter and wine. It can only raise a call
(not_indicated -> may -> contains), never lower one.

Rules for what goes in, set before any eval run:

- Ingredient, sauce, condiment, stock, coating and technique words only. No whole
  dish names ("Caesar", "Waldorf", "Kung Pao"): a dish name says what a dish
  usually is, which is the tagger's job and the second-opinion check's job.
- English plus the menu languages of the tagger's 15 cuisines, written without
  accents (matching strips accents), and CJK terms in script.
- "contains" when the word names the allergen or a product made from it,
  "may" when the word usually carries the allergen but not always.

The EU "nuts" group is tree nuts (almond, hazelnut, walnut, cashew, pecan, Brazil,
pistachio, macadamia). Pine nuts are not on it and are not mapped.
"""
from __future__ import annotations

import re
import unicodedata

LEVEL_RANK = {"not_indicated": 0, "may": 1, "contains": 2}

C, M = "contains", "may"

# term -> [(allergen, level)]. Latin-script terms are matched as whole words
# (a trailing plural "s" or "es" is allowed); CJK terms as substrings.
LEXICON: dict[str, list[tuple[str, str]]] = {
    # gluten
    "wheat": [("gluten", C)], "flour": [("gluten", M)], "breadcrumbs": [("gluten", C)],
    "breadcrumb": [("gluten", C)], "panko": [("gluten", C)], "bread": [("gluten", C)],
    "croutons": [("gluten", C)], "crouton": [("gluten", C)], "toast": [("gluten", C)],
    "crostini": [("gluten", C)], "pasta": [("gluten", C)], "noodles": [("gluten", M)],
    "udon": [("gluten", C)], "ramen": [("gluten", C)], "soba": [("gluten", M)],
    "couscous": [("gluten", C)], "bulgur": [("gluten", C)], "freekeh": [("gluten", C)],
    "semolina": [("gluten", C)], "seitan": [("gluten", C)], "barley": [("gluten", C)],
    "rye": [("gluten", C)], "spelt": [("gluten", C)], "farro": [("gluten", C)],
    "filo": [("gluten", C)], "phyllo": [("gluten", C)], "pastry": [("gluten", C)],
    "tortilla": [("gluten", M)], "pita": [("gluten", C)], "naan": [("gluten", C), ("milk", M)],
    "roux": [("gluten", C), ("milk", M)], "batter": [("gluten", C), ("eggs", M)],
    "tempura": [("gluten", C), ("eggs", M)], "dumpling": [("gluten", M)],
    "bun": [("gluten", M)], "brioche": [("gluten", C), ("eggs", C), ("milk", C)],
    "beer": [("gluten", C)], "injera": [("gluten", M)], "pan": [],  # Spanish "bread" is too ambiguous
    "pane": [("gluten", C)], "pain": [("gluten", C)], "harina": [("gluten", M)],
    "farine": [("gluten", M)], "farina": [("gluten", M)], "empanada": [("gluten", C)],
    "dredged": [("gluten", M)], "meuniere": [("gluten", C), ("milk", C)],
    "a la romana": [("gluten", C), ("eggs", M)], "savoiardi": [("gluten", C), ("eggs", C)],
    "ladyfingers": [("gluten", C), ("eggs", C)], "sponge": [("gluten", M), ("eggs", M)],
    "spaghetti": [("gluten", C)], "linguine": [("gluten", C)], "penne": [("gluten", C)],
    "tagliatelle": [("gluten", C)], "fettuccine": [("gluten", C)], "pappardelle": [("gluten", C)],
    "lasagne": [("gluten", C)], "lasagna": [("gluten", C)], "ravioli": [("gluten", C)],
    "tortellini": [("gluten", C)], "rigatoni": [("gluten", C)], "bucatini": [("gluten", C)],
    "orecchiette": [("gluten", C)], "trofie": [("gluten", C)], "fusilli": [("gluten", C)],
    "macaroni": [("gluten", C)], "orzo": [("gluten", C)], "gnocchi": [("gluten", M)],
    "muffin": [("gluten", C)], "biscuit": [("gluten", C)], "cake": [("gluten", M), ("eggs", M)],
    # crustaceans
    "shrimp": [("crustaceans", C)], "prawn": [("crustaceans", C)], "prawns": [("crustaceans", C)],
    "lobster": [("crustaceans", C)], "crab": [("crustaceans", C)], "langoustine": [("crustaceans", C)],
    "crayfish": [("crustaceans", C)], "crawfish": [("crustaceans", C)], "scampi": [("crustaceans", C)],
    "gambas": [("crustaceans", C)], "gamba": [("crustaceans", C)], "camarones": [("crustaceans", C)],
    "camaron": [("crustaceans", C)], "langostinos": [("crustaceans", C)], "crevettes": [("crustaceans", C)],
    "homard": [("crustaceans", C)], "gamberi": [("crustaceans", C)], "gamberetti": [("crustaceans", C)],
    "aragosta": [("crustaceans", C)], "granchio": [("crustaceans", C)], "ebi": [("crustaceans", C)],
    "kani": [("crustaceans", C)], "goong": [("crustaceans", C)], "kung": [("crustaceans", M)],
    "tom yum": [("crustaceans", M), ("fish", C)], "tom": [],  # Vietnamese "tôm" collides with English
    "shrimp paste": [("crustaceans", C)], "kapi": [("crustaceans", C)], "belacan": [("crustaceans", C)],
    "terasi": [("crustaceans", C)], "saeujeot": [("crustaceans", C)], "marisco": [("crustaceans", C), ("molluscs", C)],
    "mariscos": [("crustaceans", C), ("molluscs", C)], "frutti di mare": [("crustaceans", C), ("molluscs", C)],
    "fruits de mer": [("crustaceans", C), ("molluscs", C)], "seafood": [("crustaceans", M), ("molluscs", M), ("fish", M)],
    "エビ": [("crustaceans", C)], "海老": [("crustaceans", C)], "えび": [("crustaceans", C)],
    "虾": [("crustaceans", C)], "蝦": [("crustaceans", C)], "蟹": [("crustaceans", C)],
    "กุ้ง": [("crustaceans", C)], "ปู": [("crustaceans", C)], "새우": [("crustaceans", C)],
    # eggs
    "egg": [("eggs", C)], "eggs": [("eggs", C)], "yolk": [("eggs", C)], "yolks": [("eggs", C)],
    "mayonnaise": [("eggs", C), ("mustard", M)], "mayo": [("eggs", C)], "aioli": [("eggs", C)],
    "allioli": [("eggs", M)], "hollandaise": [("eggs", C), ("milk", C)], "bearnaise": [("eggs", C), ("milk", C)],
    "custard": [("eggs", C), ("milk", C)], "meringue": [("eggs", C)], "souffle": [("eggs", C)],
    "frittata": [("eggs", C)], "omelette": [("eggs", C)], "omelet": [("eggs", C)], "quiche": [("eggs", C), ("milk", C), ("gluten", C)],
    "huevo": [("eggs", C)], "huevos": [("eggs", C)], "oeuf": [("eggs", C)], "oeufs": [("eggs", C)],
    "uovo": [("eggs", C)], "uova": [("eggs", C)], "tamago": [("eggs", C)], "tamagoyaki": [("eggs", C)],
    "remoulade": [("eggs", C), ("mustard", C)], "tartare sauce": [("eggs", C)], "rouille": [("eggs", M)],
    "egg noodles": [("eggs", C), ("gluten", C)], "carbonara": [("eggs", C), ("milk", C)],
    "卵": [("eggs", C)], "玉子": [("eggs", C)], "蛋": [("eggs", C)], "ไข่": [("eggs", C)], "계란": [("eggs", C)],
    "달걀": [("eggs", C)],
    # fish
    "fish": [("fish", C)], "fish sauce": [("fish", C)], "anchovy": [("fish", C)], "anchovies": [("fish", C)],
    "worcestershire": [("fish", C)], "dashi": [("fish", C)], "bonito": [("fish", C)], "katsuobushi": [("fish", C)],
    "tuna": [("fish", C)], "tonno": [("fish", C)], "tonnato": [("fish", C), ("eggs", C)], "atun": [("fish", C)],
    "salmon": [("fish", C)], "saumon": [("fish", C)], "salmone": [("fish", C)], "cod": [("fish", C)],
    "baccala": [("fish", C)], "bacalao": [("fish", C)], "morue": [("fish", C)], "sardine": [("fish", C)],
    "sardines": [("fish", C)], "mackerel": [("fish", C)], "saba": [("fish", C)], "sole": [("fish", C)],
    "trout": [("fish", C)], "halibut": [("fish", C)], "snapper": [("fish", C)], "sea bass": [("fish", C)],
    "catfish": [("fish", C)], "tilapia": [("fish", C)], "hilsa": [("fish", C)], "ilish": [("fish", C)],
    "pescado": [("fish", C)], "pesce": [("fish", C)], "poisson": [("fish", C)], "acciughe": [("fish", C)],
    "alici": [("fish", C)], "boquerones": [("fish", C)], "anchois": [("fish", C)], "nam pla": [("fish", C)],
    "nuoc mam": [("fish", C)], "nuoc cham": [("fish", C)], "patis": [("fish", C)], "pla": [("fish", C)],
    "maguro": [("fish", C)], "sake no": [], "sashimi": [("fish", C)], "jeotgal": [("fish", C)],
    "caviar": [("fish", C)], "roe": [("fish", C)], "bottarga": [("fish", C)], "colatura": [("fish", C)],
    "kimchi": [("fish", M), ("crustaceans", M)],
    "魚": [("fish", C)], "鱼": [("fish", C)], "鰹": [("fish", C)], "出汁": [("fish", C)], "だし": [("fish", C)],
    "鯖": [("fish", C)], "鰤": [("fish", C)], "鮭": [("fish", C)], "ปลา": [("fish", C)], "น้ำปลา": [("fish", C)],
    "생선": [("fish", C)], "멸치": [("fish", C)], "김치": [("fish", M), ("crustaceans", M)],
    # peanuts
    "peanut": [("peanuts", C)], "peanuts": [("peanuts", C)], "groundnut": [("peanuts", C)],
    "satay": [("peanuts", C), ("soy", M)], "cacahuate": [("peanuts", C)], "cacahuete": [("peanuts", C)],
    "mani": [("peanuts", C)], "arachide": [("peanuts", C)], "arachidi": [("peanuts", C)],
    "落花生": [("peanuts", C)], "花生": [("peanuts", C)], "ถั่วลิสง": [("peanuts", C)], "땅콩": [("peanuts", C)],
    # soy
    "soy": [("soy", C)], "soya": [("soy", C)], "soy sauce": [("soy", C), ("gluten", C)], "shoyu": [("soy", C), ("gluten", C)],
    "tamari": [("soy", C)], "miso": [("soy", C)], "tofu": [("soy", C)], "tempeh": [("soy", C)],
    "edamame": [("soy", C)], "natto": [("soy", C)], "doenjang": [("soy", C)], "gochujang": [("soy", C), ("gluten", M)],
    "doubanjiang": [("soy", M), ("gluten", M)], "hoisin": [("soy", C), ("gluten", M), ("sesame", M)],
    "kecap manis": [("soy", C), ("gluten", M)], "teriyaki": [("soy", C), ("gluten", C)],
    "ponzu": [("soy", C), ("fish", M), ("gluten", M)], "black bean sauce": [("soy", C)], "yuba": [("soy", C)],
    "tentsuyu": [("soy", C), ("fish", C), ("gluten", C)], "mentsuyu": [("soy", C), ("fish", C), ("gluten", C)],
    "醤油": [("soy", C), ("gluten", C)], "しょうゆ": [("soy", C), ("gluten", C)], "味噌": [("soy", C)],
    "豆腐": [("soy", C)], "酱油": [("soy", C), ("gluten", C)], "醬油": [("soy", C), ("gluten", C)],
    "豆瓣": [("soy", M), ("gluten", M)], "豆豉": [("soy", C)], "เต้าหู้": [("soy", C)], "ซีอิ๊ว": [("soy", C), ("gluten", C)],
    "간장": [("soy", C), ("gluten", C)], "된장": [("soy", C)], "두부": [("soy", C)], "고추장": [("soy", C), ("gluten", M)],
    # milk
    "milk": [("milk", C)], "butter": [("milk", C)], "cream": [("milk", C)], "cheese": [("milk", C)],
    "yogurt": [("milk", C)], "yoghurt": [("milk", C)], "ghee": [("milk", C)], "paneer": [("milk", C)],
    "buttermilk": [("milk", C)], "creme fraiche": [("milk", C)], "bechamel": [("milk", C), ("gluten", C)],
    "parmesan": [("milk", C)], "parmigiano": [("milk", C)], "pecorino": [("milk", C)], "mozzarella": [("milk", C)],
    "ricotta": [("milk", C)], "mascarpone": [("milk", C)], "burrata": [("milk", C)], "gorgonzola": [("milk", C)],
    "provolone": [("milk", C)], "feta": [("milk", C)], "halloumi": [("milk", C)], "labneh": [("milk", C)],
    "gruyere": [("milk", C)], "brie": [("milk", C)], "queso": [("milk", C)], "crema": [("milk", C)],
    "leche": [("milk", C)], "leches": [("milk", C)], "nata": [("milk", C)], "mantequilla": [("milk", C)],
    "lait": [("milk", C)], "beurre": [("milk", C)], "creme": [("milk", C)], "fromage": [("milk", C)],
    "latte": [("milk", C)], "burro": [("milk", C)], "panna": [("milk", C)], "formaggio": [("milk", C)],
    "dahi": [("milk", C)], "raita": [("milk", C)], "kulfi": [("milk", C)], "malai": [("milk", C)],
    "niter kibbeh": [("milk", C)], "kibbeh": [], "ayib": [("milk", C)], "tzatziki": [],
    "beurre blanc": [("milk", C), ("sulphites", C)], "beurre noisette": [("milk", C)],
    "牛乳": [("milk", C)], "バター": [("milk", C)], "チーズ": [("milk", C)], "奶": [("milk", C)],
    "黄油": [("milk", C)], "芝士": [("milk", C)], "นม": [("milk", C)], "เนย": [("milk", C)], "치즈": [("milk", C)],
    # tree nuts
    "almond": [("nuts", C)], "almonds": [("nuts", C)], "hazelnut": [("nuts", C)], "hazelnuts": [("nuts", C)],
    "walnut": [("nuts", C)], "walnuts": [("nuts", C)], "cashew": [("nuts", C)], "cashews": [("nuts", C)],
    "pecan": [("nuts", C)], "pecans": [("nuts", C)], "pistachio": [("nuts", C)], "pistachios": [("nuts", C)],
    "macadamia": [("nuts", C)], "brazil nut": [("nuts", C)], "marzipan": [("nuts", C)], "frangipane": [("nuts", C), ("eggs", C), ("milk", C), ("gluten", C)],
    "praline": [("nuts", C)], "nutella": [("nuts", C), ("milk", C)], "almendra": [("nuts", C)], "almendras": [("nuts", C)],
    "nuez": [("nuts", C)], "nueces": [("nuts", C)], "nogada": [("nuts", C), ("milk", M)], "avellana": [("nuts", C)],
    "mandorla": [("nuts", C)], "mandorle": [("nuts", C)], "nocciola": [("nuts", C)], "nocciole": [("nuts", C)],
    "noce": [("nuts", C)], "noci": [("nuts", C)], "pistacchio": [("nuts", C)], "amande": [("nuts", C)],
    "amandes": [("nuts", C)], "noisette": [("nuts", C)], "noix": [("nuts", C)], "romesco": [("nuts", C), ("gluten", M)],
    "picada": [("nuts", C), ("gluten", M)], "kaju": [("nuts", C)], "badam": [("nuts", C)],
    "腰果": [("nuts", C)], "核桃": [("nuts", C)], "杏仁": [("nuts", C)], "くるみ": [("nuts", C)],
    "ถั่วหิมพานต์": [("nuts", C)], "มะม่วงหิมพานต์": [("nuts", C)], "호두": [("nuts", C)],
    # celery
    "celery": [("celery", C)], "celeriac": [("celery", C)], "celery salt": [("celery", C)], "celery seed": [("celery", C)],
    "celeri": [("celery", C)], "celeri-rave": [("celery", C)], "sedano": [("celery", C)], "apio": [("celery", C)],
    "mirepoix": [("celery", C)], "soffritto": [("celery", M)], "sofrito": [("celery", M)], "holy trinity": [("celery", C)],
    "stock": [("celery", M)], "broth": [("celery", M)], "bouillon": [("celery", M)], "consomme": [("celery", M)],
    "fond": [("celery", M)], "brodo": [("celery", M)], "caldo": [("celery", M)], "court-bouillon": [("celery", C)],
    "芹菜": [("celery", C)], "セロリ": [("celery", C)], "ขึ้นฉ่าย": [("celery", C)], "셀러리": [("celery", C)],
    # mustard
    "mustard": [("mustard", C)], "dijon": [("mustard", C)], "moutarde": [("mustard", C)], "mostaza": [("mustard", C)],
    "senape": [("mustard", C)], "mostarda": [("mustard", C)], "karashi": [("mustard", C)], "shorshe": [("mustard", C)],
    "sarson": [("mustard", C)], "rai": [("mustard", C)], "kasundi": [("mustard", C)], "panch phoron": [("mustard", C)],
    "vinaigrette": [("mustard", M), ("sulphites", C)], "honey mustard": [("mustard", C)], "piccalilli": [("mustard", C)],
    "からし": [("mustard", C)], "芥末": [("mustard", M)], "겨자": [("mustard", C)],
    # sesame
    "sesame": [("sesame", C)], "tahini": [("sesame", C)], "tahina": [("sesame", C)], "halva": [("sesame", C)],
    "halvah": [("sesame", C)], "zaatar": [("sesame", C)], "za'atar": [("sesame", C)], "gomasio": [("sesame", C)],
    "goma": [("sesame", C)], "ajonjoli": [("sesame", C)], "sesamo": [("sesame", C)], "til": [("sesame", C)],
    "benne": [("sesame", C)], "furikake": [("sesame", C), ("fish", M)], "胡麻": [("sesame", C)], "ごま": [("sesame", C)],
    "芝麻": [("sesame", C)], "麻油": [("sesame", C)], "งา": [("sesame", C)], "참기름": [("sesame", C)], "깨": [("sesame", C)],
    # sulphites
    "wine": [("sulphites", C)], "vino": [("sulphites", C)], "vin": [("sulphites", C)], "vinho": [("sulphites", C)],
    "marsala": [("sulphites", C)], "sherry": [("sulphites", C)], "jerez": [("sulphites", C)], "madeira": [("sulphites", C)],
    "port wine": [("sulphites", C)], "vermouth": [("sulphites", C)], "champagne": [("sulphites", C)], "prosecco": [("sulphites", C)],
    "cider": [("sulphites", C)], "shaoxing": [("sulphites", C), ("gluten", M)], "sake": [("sulphites", M)],
    "mirin": [("sulphites", M)], "vinegar": [("sulphites", M)], "vinagre": [("sulphites", M)], "aceto": [("sulphites", M)],
    "vinaigre": [("sulphites", M)], "balsamic": [("sulphites", C)], "balsamico": [("sulphites", C)],
    "wine vinegar": [("sulphites", C)], "sherry vinegar": [("sulphites", C)], "dried apricots": [("sulphites", M)],
    "dried fruit": [("sulphites", M)], "raisins": [("sulphites", M)], "sultanas": [("sulphites", M)],
    "bordelaise": [("sulphites", C), ("milk", M)], "marchand de vin": [("sulphites", C)], "au vin": [("sulphites", C)],
    "al vino": [("sulphites", C)], "deglazed": [("sulphites", M)], "reduction": [("sulphites", M)],
    "酒": [("sulphites", M)], "紹興": [("sulphites", C)], "绍兴": [("sulphites", C)], "みりん": [("sulphites", M)],
    # lupin
    "lupin": [("lupin", C)], "lupini": [("lupin", C)], "lupine": [("lupin", C)], "tremoco": [("lupin", C)],
    "tremocos": [("lupin", C)], "altramuz": [("lupin", C)], "altramuces": [("lupin", C)], "chochos": [("lupin", C)],
    "lupin flour": [("lupin", C)],
    # molluscs
    "squid": [("molluscs", C)], "octopus": [("molluscs", C)], "calamari": [("molluscs", C)], "calamares": [("molluscs", C)],
    "calamar": [("molluscs", C)], "pulpo": [("molluscs", C)], "polpo": [("molluscs", C)], "pulpe": [("molluscs", C)],
    "cuttlefish": [("molluscs", C)], "sepia": [("molluscs", C)], "seppia": [("molluscs", C)], "seiche": [("molluscs", C)],
    "mussels": [("molluscs", C)], "mussel": [("molluscs", C)], "moules": [("molluscs", C)], "cozze": [("molluscs", C)],
    "mejillones": [("molluscs", C)], "clams": [("molluscs", C)], "clam": [("molluscs", C)], "vongole": [("molluscs", C)],
    "almejas": [("molluscs", C)], "palourdes": [("molluscs", C)], "oyster": [("molluscs", C)], "oysters": [("molluscs", C)],
    "oyster sauce": [("molluscs", C), ("soy", M), ("gluten", M)], "huitres": [("molluscs", C)], "ostras": [("molluscs", C)],
    "scallop": [("molluscs", C)], "scallops": [("molluscs", C)], "coquilles": [("molluscs", C)], "capesante": [("molluscs", C)],
    "vieiras": [("molluscs", C)], "abalone": [("molluscs", C)], "snail": [("molluscs", C)], "snails": [("molluscs", C)],
    "escargot": [("molluscs", C)], "escargots": [("molluscs", C)], "whelk": [("molluscs", C)], "conch": [("molluscs", C)],
    "ika": [("molluscs", C)], "tako": [("molluscs", C)], "hotate": [("molluscs", C)], "kaki": [],
    "hoi": [("molluscs", C)], "pla muek": [("molluscs", C)], "muek": [("molluscs", C)],
    "イカ": [("molluscs", C)], "いか": [("molluscs", C)], "タコ": [("molluscs", C)], "たこ": [("molluscs", C)],
    "帆立": [("molluscs", C)], "牡蠣": [("molluscs", C)], "蚝": [("molluscs", C)], "蠔": [("molluscs", C)],
    "鱿": [("molluscs", C)], "魷": [("molluscs", C)], "八爪": [("molluscs", C)], "蛤": [("molluscs", C)],
    "贝": [("molluscs", C)], "貝": [("molluscs", C)], "หอย": [("molluscs", C)], "ปลาหมึก": [("molluscs", C)],
    "오징어": [("molluscs", C)], "문어": [("molluscs", C)], "조개": [("molluscs", C)], "굴": [("molluscs", C)],
}

# Drop the placeholder entries kept above to document deliberate omissions.
LEXICON = {k: v for k, v in LEXICON.items() if v}

# Script-specific groupings: Thai "ปลา" (fish) is a prefix of "ปลาหมึก" (squid), which is
# fine because both are added; the lexicon never lowers a call.


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower().replace("’", "'")


def _is_cjk_or_thai(term: str) -> bool:
    return any(ord(ch) > 0x2E80 for ch in term)


_LATIN = sorted((t for t in LEXICON if not _is_cjk_or_thai(t)), key=len, reverse=True)
_SCRIPT = [t for t in LEXICON if _is_cjk_or_thai(t)]
_PATTERN = re.compile(r"(?<![a-z])(" + "|".join(re.escape(_fold(t)) for t in _LATIN) + r")(?:e?s)?(?![a-z])")
_FOLDED = {_fold(t): t for t in _LATIN}


def lexicon_hits(text: str) -> list[dict]:
    """Every lexicon term found in the text, with the allergen calls it implies."""
    folded = _fold(text)
    hits = []
    for m in _PATTERN.finditer(folded):
        term = _FOLDED[m.group(1)]
        hits.append({"term": term, "calls": LEXICON[term]})
    for term in _SCRIPT:
        if term in text:
            hits.append({"term": term, "calls": LEXICON[term]})
    return hits


def lexicon_calls(text: str) -> dict[str, str]:
    """{allergen: strongest level implied by any term in the text}."""
    out: dict[str, str] = {}
    for h in lexicon_hits(text):
        for allergen, level in h["calls"]:
            if LEVEL_RANK[level] > LEVEL_RANK.get(out.get(allergen, "not_indicated"), 0):
                out[allergen] = level
    return out


def raise_levels(levels: dict[str, str], extra: dict[str, str]) -> dict[str, str]:
    """Combine two sets of calls, keeping the stronger level for each allergen."""
    out = dict(levels)
    for allergen, level in extra.items():
        if LEVEL_RANK[level] > LEVEL_RANK[out.get(allergen, "not_indicated")]:
            out[allergen] = level
    return out


import Levenshtein
import spacy_udpipe
from sklearn.metrics.pairwise import cosine_similarity

import tools.rag_functions as rf

"""try:
    spacy_udpipe.download("cs")
except Exception as err:
    print(err)
"""

nlp = spacy_udpipe.load("cs")

class SimilarityCalculator:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def _get_ngrams(self, lemmas, n):
        return {" ".join(lemmas[i : i + n]) for i in range(len(lemmas) - 1)}

    def _get_text_vocab(self, text, n):
        doc = nlp(text)
        lemmas = [token.lemma_.lower() for token in doc if token.is_alpha]
        if n > 1:
            vocab = self._get_ngrams(lemmas, n)
        else:
            vocab = set(lemmas)
        return vocab
    
    def lexicon_match(self, n=1):
        vocab1 = self._get_text_vocab(self.x, n)
        vocab2 = self._get_text_vocab(self.y, n)
        intersection = vocab1 & vocab2
        return 2 * len(intersection) / (len(vocab1) + len(vocab2))
    
    def lexicon_difference(self, n=1):
        vocab1 = self._get_text_vocab(self.x, n)
        vocab2 = self._get_text_vocab(self.y, n)
        intersection = vocab1 & vocab2
        union = vocab1 | vocab2
        return 1 - (len(intersection) / len(union))


    def cosine(self):
        result = cosine_similarity([rf.vectorize_text(self.x)], [rf.vectorize_text(self.y)])
        return result
    
    def levenshtein(self):
        return Levenshtein.ratio(self.x, self.y)

    def compare_texts(self): # uses all of the comparison metrics
        cosine = self.cosine()
        levenshtein = self.levenshtein()
        dice = self.lexicon_match()
        word_diff = self.lexicon_difference()
        return {
            "cosine": cosine,
            "levenshtein": levenshtein,
            "dice": dice,
            "word_difference": word_diff
        }
    
# 6,8,8,9,11,11
expected_answer = "Výsledky měření teploty na povrchu objektu X7-N1 ukázaly nepřirozené výkyvy, kdy došlo k nárůstu teploty o několik stupňů během několika hodin, což naznačuje, že objekt může obsahovat aktivní geologické procesy nebo neznámý zdroj energie. Pokud se tyto domněnky potvrdí, mohlo by jít o první objekt ve vnější Sluneční soustavě s aktivním vnitřním ohřevem."
generated_answers = [
    "Měření teploty na povrchu objektu X7-N1 ukázalo nepřirozené výkyvy teploty, kdy teplota stoupla o několik stupňů během několika hodin, což je neobvyklé pro objekt ve vzdálenosti 400 milionů kilometrů od Slunce. To naznačuje, že objekt může obsahovat aktivní geologické procesy nebo neznámý zdroj energie.",
    "Měření teploty na povrchu objektu X7-N1 ukázala nepřirozené výkyvy teploty. Infračervené senzory sondy New Horizons zaznamenaly nárůst teploty o několik stupňů v průběhu několika hodin, což je neobvyklé v takové vzdálenosti od Slunce. To naznačuje, že objekt může obsahovat aktivní geologické procesy nebo dokonce neznámý zdroj energie.",
    "Měření teploty na povrchu objektu X7-N1 ukázala nepřirozené výkyvy, kdy došlo k nárůstu teploty o několik stupňů během několika hodin, což v takové vzdálenosti od Slunce nedává smysl. Tento jev naznačuje, že objekt by mohl obsahovat aktivní geologické procesy nebo neznámý zdroj energie.",
    "Měření teploty na povrchu objektu X7-N1 pomocí infračervených senzorů sondy New Horizons ukázalo nepřirozené výkyvy teploty, konkrétně nárůst o několik stupňů v průběhu několika hodin. To naznačuje, že objekt může obsahovat aktivní geologické procesy nebo neznámý zdroj energie, což by mohlo zpochybnit současné modely formování ledových těles ve Sluneční soustavě.",
    "Výsledky měření teploty na povrchu objektu X7-N1 ukázaly nepřirozené výkyvy, kdy došlo k nárůstu teploty o několik stupňů během několika hodin, což na takovou vzdálenost od Slunce nedává smysl. To naznačuje, že objekt může obsahovat aktivní geologické procesy nebo neznámý zdroj energie, což by mohlo znamenat, že se jedná o první objekt ve vnější Sluneční soustavě s aktivním vnitřním ohřevem.",
    "Měření teploty na povrchu objektu X7-N1 ukázala nepřirozené výkyvy, kdy došlo k nárůstu teploty o několik stupňů během několika hodin, což naznačuje, že objekt může obsahovat aktivní geologické procesy nebo neznámý zdroj energie. Pokud se tyto domněnky potvrdí, mohl by to být první objekt ve vnější Sluneční soustavě s aktivním vnitřním ohřevem."
]

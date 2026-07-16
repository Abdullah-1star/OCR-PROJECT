import cv2
import easyocr

BOILERPLATE = (
    "جمهورية مصر العربية",
    "بطاقة تحقيق الشخصية",
    "جمهورية",
    "مصر العربية",
)

ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
WESTERN_DIGITS = "0123456789"
ARABIC_TO_WESTERN = str.maketrans(ARABIC_DIGITS, WESTERN_DIGITS)


class OcrExtractor:
    def __init__(self, languages=("ar", "en"), min_confidence=0.25, min_width=1200):
        self.reader = easyocr.Reader(list(languages), gpu=False)
        self.min_confidence = min_confidence
        self.min_width = min_width

    def read_text(self, id_card):
        id_card = self._upscale(id_card)
        results = self.reader.readtext(id_card)
        results = [r for r in results if r[2] > self.min_confidence]
        results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))
        lines = [text.strip().translate(ARABIC_TO_WESTERN) for (_, text, _) in results]
        return lines

    def _upscale(self, id_card):
        h, w = id_card.shape[:2]
        if w >= self.min_width:
            return id_card
        scale = self.min_width / w
        return cv2.resize(id_card, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    def extract_fields(self, id_card):
        lines = self.read_text(id_card)

        fields = {
            "name": "",
            "place": "",
            "center": "",
        }

        name_idx = None
        for idx, line in enumerate(lines):
            digits = "".join(ch for ch in line if ch.isdigit())
            if len(digits) >= 8:
                continue
            if self._is_boilerplate(line):
                continue
            if any(ch.isalpha() for ch in line) and len(line.split()) >= 2:
                name_idx = idx
                break

        if name_idx is not None:
            name = lines[name_idx]
            if name_idx > 0:
                prev = lines[name_idx - 1]
                prev_digits = "".join(ch for ch in prev if ch.isdigit())
                is_single_word_name = (
                    len(prev.split()) == 1
                    and prev_digits == ""
                    and not self._is_boilerplate(prev)
                    and any(ch.isalpha() for ch in prev)
                )
                if is_single_word_name:
                    name = f"{prev} {name}"
            fields["name"] = name

        center_idx = self._find_center(lines)
        if center_idx is not None:
            fields["center"] = lines[center_idx]
            fields["place"] = self._find_place(lines, center_idx, name_idx)

        return fields, lines

    def _find_center(self, lines):
        for idx, line in enumerate(lines):
            if "مركز" in line or "قسم" in line:
                return idx
        return None

    def _find_place(self, lines, center_idx, name_idx):
        candidate_idx = center_idx - 1
        if candidate_idx > (name_idx if name_idx is not None else -1):
            candidate = lines[candidate_idx]
            if candidate.strip() and not self._is_boilerplate(candidate):
                return candidate

        for line in lines:
            if "قرية" in line or "حي " in line or line.startswith("حي"):
                return line

        return ""

    def _is_boilerplate(self, line):
        cleaned = line.strip()
        return any(cleaned == b or cleaned in b or b in cleaned for b in BOILERPLATE)
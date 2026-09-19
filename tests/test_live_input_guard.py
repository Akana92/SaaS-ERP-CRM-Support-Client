"""CPU-only input guard behavior, including durable state and retry boundaries."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from support.live_input_guard import check_input_status, evaluate_input, inspect_message


class InputGuardTests(unittest.TestCase):
    def test_irregular_latin_home_cluster_noise(self):
        for text in ("asdasdgfasdg", "ASDASDGFASDG!", "asdasdg fasdg",
                     "sdfgsdfgasd", "dfgdfgasdf"):
            with self.subTest(text=text):
                self.assertIsNotNone(inspect_message(text))
        for text in ("glassglass", "safeguard", "Douglas", "asdfg-123",
                     "https://asdasdgfasdg.example", "The internet is down",
                     "Error asdasdgfasdg appears on screen", "glass glass"):
            with self.subTest(text=text):
                self.assertIsNone(inspect_message(text))

    def test_latin_noise_consumes_attempts_and_locks_for_five_minutes(self):
        state = {}
        for index, message in enumerate(("asdasdgfasdg", "sdfgsdfgasd", "dfgdfgasdf")):
            result = evaluate_input(state, message, str(index), 100)
            self.assertFalse(result["allowed"])
            self.assertEqual(result["attempts_remaining"], 2 - index)
            self.assertEqual(result["blocked"], index == 2)
        self.assertEqual(result["blocked_until"], 400)
        self.assertEqual(result["retry_after"], 300)

    def test_short_keyboard_blocks_in_both_layouts(self):
        for text in ("фывафыва", "asdgasdg", "ыфвапфыав", "sdfgsdfg",
                     "asdfghjk", "фыавыфапв", "ASDG asdg!", "фыва ыфва"):
            with self.subTest(text=text):
                self.assertIsNotNone(inspect_message(text))
        for text in ("safeguard", "alfalfas", "flashdata", "SSH", "asdf-1234",
                     "Кызылорда", "Авраамов", "Правопорядок", "Пропадало",
                     "ыфвапфыав — код на экране", "Поле asdgasdg в файле"):
            with self.subTest(text=text):
                self.assertIsNone(inspect_message(text))

    def test_irregular_cyrillic_noise_is_rejected_without_exact_repeated_cycles(self):
        # The first five strings are production regressions. The rest exercise
        # different key combinations, spacing and punctuation rather than a
        # lookup table containing only the observed messages.
        for text in ("апвравпо", "ыаволрпрлоывап", "кщаогпрдлвапр",
                     "гшналопралпро", "ывраравыравы", "жцщшгзхфжцщ",
                     "щзхжцфыгшщз", "фжывапролдж", "лджэщшгнеку",
                     "апролджэждлорп", "ывапнгшлдщзхъ", "ждлорпавыфячс",
                     "кщаогпрдлвапр гшналопралпро",
                     "КЩАОГПРДЛВАПР!", "  ыаволрпрлоывап?  "):
            with self.subTest(text=text):
                self.assertIsNotNone(inspect_message(text))

    def test_unknown_names_typos_and_contextual_answers_are_not_noise(self):
        for text in (
            "Екатерина", "Константин", "Владислав", "Ибрагим", "Гульнара",
            "Нурсултан", "Айгерим", "Аружан", "Әбдірахман", "Қайрат",
            "Қызылорда", "Шымкент", "Петропавловск", "Жезказган",
            "Санкт-Петербург", "Сәтбаев", "Алматы", "Астана",
            "Кызылорда", "кызылорда", "Мухамеджанов", "мухамеджанов",
            "Абдрахманов", "абдрахманов",
            "интренет", "подклчение", "бюджетирование", "переподключение",
            "авторизация", "недоступно", "перезагрузил", "заработало",
            "нет сети", "не помогло", "вчера", "сегодня", "всё ещё нет",
            "да", "нет", "ага", "угу", "ок", "20:15", "Абай 164",
            "ERR_CONNECTION_RESET", "NullPointerException", "0x80070005",
            "БП-224524", "224524", "SIM", "VPN", "DHCP", "Wi-Fi",
            "Импорт услуг не запускаеться", "Пишет ошибку фвыпфывпфывп",
            "Нет интернета, гшналопралпро — текст ошибки",
        ):
            with self.subTest(text=text):
                self.assertIsNone(inspect_message(text))

    def test_conservative_detection(self):
        for text in ("фвыпфывпфывп", "аааааааа", "абабабабабаб", "!!!???", "   "):
            with self.subTest(text=text):
                self.assertIsNotNone(inspect_message(text))
        for text in ("да", "нет", "нет сети", "ERP", "СИМ", "интернет",
                     "Абай 164", "Ибрагим", "Караганда", "TIMEOUT", "E-404", "123456",
                     "00000000", "AB123456", "https://example.org", "PIN", "👍", "Здравствуйте"):
            with self.subTest(text=text):
                self.assertIsNone(inspect_message(text))

    def test_three_attempts_and_expiry(self):
        state = {}
        for index in range(3):
            result = evaluate_input(state, "фвыпфывпфывп", str(index), 100)
            self.assertFalse(result["allowed"])
            self.assertEqual(result["attempts_remaining"], 2 - index)
            self.assertEqual(result["blocked"], index == 2)
            self.assertEqual(result["code"], "input_blocked" if index == 2 else "invalid_message")
        self.assertEqual(result["blocked_until"], 400)
        self.assertEqual(result["retry_after"], 300)
        blocked = evaluate_input(state, "нет сети", "valid", 399.1)
        self.assertTrue(blocked["blocked"])
        self.assertEqual(blocked["retry_after"], 1)
        self.assertEqual(blocked["blocked_until"], 400)
        self.assertTrue(evaluate_input(state, "нет сети", "valid", 400)["allowed"])
        self.assertEqual(evaluate_input(state, "!!!", "fresh", 401)["attempts_remaining"], 2)

    def test_keyboard_noise_variants_preserve_real_words_and_titles(self):
        for text in ("фвыпфывпфывп", "фывафывафыва", "йцукенйцукен",
                     "asdfasdfasdf", "qwertyqwertyqwerty", "ФЫВА фыва, ФЫВА!",
                     "ЙЦУКЕН, йцукен!", "ASDF asdf asdf!", "фвып пфыв фывп",
                     "asdf fdsa asdf", "abcdefabcdefabcdef"):
            with self.subTest(text=text):
                self.assertIsNotNone(inspect_message(text))
        for text in ("правда", "пропало", "правопорядок", "выправка", "Поддержка",
                     "Управление договорами", "Правила оплаты", "ERP: Счета и акты",
                     "Мой пакет → Остатки", "Клиентский кабинет", "ASDF-1234",
                     "йцукен123", "фыва фыва 164", "https://asdfasdfasdf.example"):
            with self.subTest(text=text):
                self.assertIsNone(inspect_message(text))

    def test_retry_and_persistence(self):
        state = {}
        first = evaluate_input(state, "!!!", "same", 10)
        restored = json.loads(json.dumps(state))
        self.assertEqual(evaluate_input(restored, "!!!", "same", 11), first)
        conflict = evaluate_input(restored, "???", "same", 12)
        self.assertEqual(conflict["code"], "idempotency_conflict")
        self.assertFalse(conflict["allowed"])
        self.assertEqual(evaluate_input(restored, "!!!", "second", 13)["attempts_remaining"], 1)
        self.assertTrue(evaluate_input(restored, "!!!", "third", 14)["blocked"])
        self.assertTrue(evaluate_input(restored, "!!!", "same", 15)["blocked"])

    def test_valid_resets_but_replayed_valid_does_not(self):
        state = {}
        evaluate_input(state, "да", "yes", 0)
        evaluate_input(state, "!!!", "bad1", 1)
        evaluate_input(state, "да", "yes", 2)
        self.assertEqual(evaluate_input(state, "!!!", "bad2", 3)["attempts_remaining"], 1)
        self.assertTrue(evaluate_input(state, "нет сети", "new", 4)["allowed"])
        self.assertEqual(evaluate_input(state, "!!!", "bad3", 5)["attempts_remaining"], 2)

    def test_owner_isolation_and_decision_copy(self):
        first, second = {}, {}
        result = evaluate_input(first, "!!!", "key", 0)
        result["allowed"] = True
        self.assertFalse(evaluate_input(first, "!!!", "key", 1)["allowed"])
        self.assertEqual(evaluate_input(second, "!!!", "key", 1)["attempts_remaining"], 2)

    def test_status_preserves_strikes_and_expires_without_recording(self):
        state = {}
        self.assertTrue(check_input_status(state, 0)["allowed"])
        self.assertEqual(state, {})
        evaluate_input(state, "!!!", "one", 0)
        snapshot = json.loads(json.dumps(state))
        self.assertEqual(check_input_status(state, 1)["attempts_remaining"], 2)
        self.assertEqual(state, snapshot)
        evaluate_input(state, "!!!", "two", 0)
        evaluate_input(state, "!!!", "three", 0)
        self.assertTrue(check_input_status(state, 299.99)["blocked"])
        self.assertTrue(check_input_status(state, 300)["allowed"])
        self.assertEqual(state, {"attempts": 0, "blocked_until": None, "receipts": {}})

    def test_receipts_are_bounded_and_do_not_store_raw_messages(self):
        state = {}
        for index in range(70):
            evaluate_input(state, "Абай 164", str(index), index)
        self.assertEqual(len(state["receipts"]), 64)
        self.assertNotIn("0", state["receipts"])
        self.assertIn("69", state["receipts"])
        self.assertNotIn("Абай 164", json.dumps(state, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()

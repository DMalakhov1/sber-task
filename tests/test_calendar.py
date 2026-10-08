import argparse
import unittest
from datetime import date
from task3_excel.export_excel import previous_month, parse_month

class CalendarTests(unittest.TestCase):
    def test_year_boundary(self):
        self.assertEqual(previous_month(date(2026, 1, 15)), (date(2025, 12, 1), date(2025, 12, 31)))

    def test_leap_february(self):
        self.assertEqual(previous_month(date(2024, 3, 1)), (date(2024, 2, 1), date(2024, 2, 29)))

    def test_explicit_month(self):
        self.assertEqual(parse_month("2026-09"), (date(2026, 9, 1), date(2026, 9, 30)))

    def test_invalid_month(self):
        for value in ("2026-13", "2026-00", "2026-9", "abc", "2026-09-01"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                parse_month(value)

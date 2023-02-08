import unittest
from datetime import datetime, timedelta
from unittest import mock

from parse import parse


class TestParse(unittest.TestCase):
    mocked_datetime = datetime(2023, 2, 5, 11, 0, 0)

    @mock.patch("parse.datetime")
    def test_parse_with_offset_only(self, mock_utcnow):
        mock_utcnow.utcnow = mock.Mock(return_value=self.mocked_datetime)
        time_str = "now()+10d"
        expected_time = self.mocked_datetime + timedelta(days=10)
        result = parse(time_str)
        self.assertEqual(result, expected_time)

    @mock.patch("parse.datetime")
    def test_parse_with_snap_only(self, mock_utcnow):
        mock_utcnow.utcnow = mock.Mock(return_value=self.mocked_datetime)

        time_str = "now()@h"
        expected_time = self.mocked_datetime.replace(minute=0, second=0, microsecond=0)
        result = parse(time_str)
        self.assertEqual(result, expected_time)

    @mock.patch("parse.datetime")
    def test_parse_with_offset_and_snap(self, mock_utcnow):
        mock_utcnow.utcnow = mock.Mock(return_value=self.mocked_datetime)

        time_str = "now()+10d@h"
        expected_time = (self.mocked_datetime + timedelta(days=10)).replace(
            minute=0, second=0, microsecond=0
        )
        result = parse(time_str)
        self.assertEqual(result, expected_time)

    def test_parse_with_invalid_offset(self):
        time_str = "now()+10z"
        with self.assertRaises(Exception) as context:
            parse(time_str)
        self.assertEqual(str(context.exception), "Invalid time offset")

    def test_parse_with_no_offset_number(self):
        time_str = "now()+d"
        with self.assertRaises(Exception) as context:
            parse(time_str)
        self.assertEqual(str(context.exception), "Invalid time offset")

    def test_parse_with_invalid_snap(self):
        time_str = "now()@z"
        with self.assertRaises(Exception) as context:
            parse(time_str)
        self.assertEqual(str(context.exception), "Invalid snap time unit")

    def test_parse_with_invalid_time_function(self):
        time_str = "nowhere()"
        with self.assertRaises(Exception) as context:
            parse(time_str)
        self.assertEqual(
            str(context.exception),
            "Invalid time function specified, currently only 'now()' is supported",
        )

    @mock.patch("parse.datetime")
    def test_parse_with_spaces_in_string(self, mock_utcnow):
        mock_utcnow.utcnow = mock.Mock(return_value=self.mocked_datetime)

        time_str = "  now() + 10d @ h  "
        expected_time = (self.mocked_datetime + timedelta(days=10)).replace(
            minute=0, second=0, microsecond=0
        )
        result = parse(time_str)
        self.assertEqual(result, expected_time)

    @mock.patch("parse.datetime")
    def test_parse_with_chained_offsets(self, mock_utcnow):
        mock_utcnow.utcnow = mock.Mock(return_value=self.mocked_datetime)

        time_str = "now()+10d-2h+3m"
        expected_time = (
            self.mocked_datetime
            + timedelta(days=10)
            - timedelta(hours=2)
            + timedelta(minutes=3)
        )
        result = parse(time_str)
        self.assertEqual(result, expected_time)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib.util
import io
import sys
import unittest
import zipfile
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fetch_official_quant_data.py"
SPEC = importlib.util.spec_from_file_location("fetch_official_quant_data", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def zip_csv(name: str, text: str) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(name, text)
    return output.getvalue()


class OfficialQuantDataTests(unittest.TestCase):
    def test_fred_parser_normalizes_missing_values(self) -> None:
        raw = b"observation_date,VIXCLS\n2024-01-02,13.20\n2024-01-03,.\n"
        rows, url, returned = MODULE.fetch_fred_series(
            "VIXCLS",
            start=date(2024, 1, 1),
            end=date(2024, 1, 4),
            fetcher=lambda _: raw,
        )

        self.assertEqual(
            rows,
            [
                {"date": "2024-01-02", "series_id": "VIXCLS", "value": "13.20"},
                {"date": "2024-01-03", "series_id": "VIXCLS", "value": ""},
            ],
        )
        self.assertIn("id=VIXCLS", url)
        self.assertEqual(returned, raw)

    def test_french_daily_parser_converts_percent_to_decimal(self) -> None:
        raw = zip_csv(
            "factors.csv",
            "Created from test data\n"
            ",Mkt-RF,SMB,HML,RF\n"
            "20240102,0.50,-0.20,0.10,0.02\n"
            "Annual Factors: January-December\n",
        )
        rows = MODULE.parse_french_zip(raw, {"Mkt-RF", "SMB", "HML", "RF"})

        self.assertEqual(rows[0]["date"], "2024-01-02")
        self.assertEqual(rows[0]["Mkt-RF"], "0.005")
        self.assertEqual(rows[0]["SMB"], "-0.002")

    def test_bls_parser_keeps_selected_releases_and_time(self) -> None:
        html = b"""
        <table>
          <tr><th>Date</th><th>Time</th><th>Release</th></tr>
          <tr><td>Friday, January 05, 2024</td><td>08:30 AM</td>
              <td>Employment Situation for December 2023</td></tr>
          <tr><td>Thursday, January 18, 2024</td><td>10:00 AM</td>
              <td>Unselected Local Report</td></tr>
        </table>
        """
        rows = MODULE.parse_bls_calendar(html, 2024)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["release_time_et"], "08:30")
        self.assertEqual(rows[0]["before_09_cutoff"], "true")

    def test_bea_json_converts_utc_to_eastern(self) -> None:
        raw = (
            b'{"Gross Domestic Product":{"release_dates":'
            b'["2025-07-30T12:30:00+00:00"]},'
            b'"file_last_updated":"2025-01-01"}'
        )
        rows = MODULE.parse_bea_json(raw)

        self.assertEqual(rows[0]["release_date"], "2025-07-30")
        self.assertEqual(rows[0]["release_time_et"], "08:30")
        self.assertEqual(rows[0]["before_09_cutoff"], "true")

    def test_fomc_parser_uses_last_scheduled_meeting_day(self) -> None:
        html = b"""
        <h3>2016</h3>
        <h5>January 26-27 Meeting - 2016</h5>
        <h5>March 2 (unscheduled) Meeting - 2016</h5>
        <h5>March 17-18 (cancelled) Meeting - 2016</h5>
        """
        rows = MODULE.parse_fomc_headings(html, 2016)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["release_date"], "2016-01-27")
        self.assertEqual(rows[0]["release_time_et"], "14:00")

    def test_current_fomc_parser_tracks_year_sections(self) -> None:
        html = b"""
        <h4>2024 FOMC Meetings</h4>
        <div class="row fomc-meeting">
          <div class="fomc-meeting__month col-xs-5"><strong>Apr/May</strong></div>
          <div class="fomc-meeting__date col-xs-4">30-1</div>
        </div>
        <h4>2023 FOMC Meetings</h4>
        <div class="row fomc-meeting">
          <div class="fomc-meeting__month col-xs-5"><strong>December</strong></div>
          <div class="fomc-meeting__date col-xs-4">12-13</div>
        </div>
        """
        rows = MODULE.parse_fomc_headings(html, None)

        self.assertEqual(
            [row["release_date"] for row in rows],
            ["2024-05-01", "2023-12-13"],
        )


if __name__ == "__main__":
    unittest.main()

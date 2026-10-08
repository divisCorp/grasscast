import pandas as pd

from grasscast.cli import main
from grasscast.pipeline import make_plan
from grasscast.weather import Place


def test_make_plan_end_to_end(fake_open_meteo, sample_log, tmp_path):
    plan = make_plan(sample_log, Place(40.76, -111.89, "America/Denver"), cache_dir=tmp_path,
                     now=pd.Timestamp("2026-10-07 00:00"))
    assert plan.n_outings == 40
    assert 1 <= len(plan.best) <= 7
    assert "Best window each day" in plan.text
    assert plan.ics.startswith("BEGIN:VCALENDAR")


def test_cli_log_appends(tmp_path, capsys):
    p = tmp_path / "o.csv"
    main(["log", "hike", "5", "--hours", "2", "--start", "2026-10-07T07:30", "--log", str(p), "--notes", "aspens!"])
    main(["log", "run", "3", "--start", "2026-10-08T06:30", "--log", str(p)])
    lines = p.read_text().strip().splitlines()
    assert lines[0] == "date,start,duration_h,activity,rating,notes"
    assert lines[1] == "2026-10-07,07:30,2.0,hike,5,aspens!"
    assert len(lines) == 3

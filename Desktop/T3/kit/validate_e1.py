"""Run the supplied T3 E1 rule module, without claiming to replace the missing
official check_experiments.py / experiment_common.py / Markdown checker."""
import csv
import importlib.util
from pathlib import Path
import sys
import types
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ec = types.ModuleType("experiment_common")
def params(s):
    return dict(item.split("=", 1) for item in s.split(";") if "=" in item)
def integer(s):
    try: return int(s)
    except (ValueError, TypeError): return None
def number(s):
    try: return float(s)
    except (ValueError, TypeError): return None
def select(rows, experiment, **criteria):
    return [r for r in rows if r["experiment"] == experiment
            and all(r["_param"].get(k) == v for k, v in criteria.items())]
class Report:
    def __init__(self): self.ok = self.fail = 0
    def test(self, code, description, passed, detail=""):
        self.ok += bool(passed); self.fail += not passed
        print(f"[{'OK' if passed else 'FAIL'}] {code} {description}" +
              (f" -- {detail}" if detail else ""))
ec.params, ec.integer, ec.number, ec.select, ec.Report = params, integer, number, select, Report
sys.modules["experiment_common"] = ec
base = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("supplied_e1_rules", base / "experiment_rules.py")
rules = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rules)
directory = Path(sys.argv[1])
with (directory / "experiments.tsv").open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
for row in rows: row["_param"] = params(row["parameters"])
print("Supplemental adapter executing the supplied, unchanged T3 experiment_rules.py.")
print("The official common/Markdown checker was not supplied and is NOT run here.")
report = Report()
rules.check(report, rows, directory)
print(f"Summary: {report.ok}/{report.ok + report.fail} supplied E1 rule checks passed")
raise SystemExit(bool(report.fail))

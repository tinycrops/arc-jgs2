from __future__ import annotations

import argparse
from pathlib import Path

from .loaders import demo_tasks, load_tasks
from .orient import orient_task
from .report import write_reports
from .solvers import write_solutions


def main() -> None:
    parser = argparse.ArgumentParser(description="Orient ARC tasks into corpus-level logical world states.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    orient = sub.add_parser("orient", help="Build orientation records and reports.")
    orient.add_argument("--data", type=Path, help="ARC task JSON file or directory.")
    orient.add_argument("--out", type=Path, default=Path("runs/orientation"), help="Output directory.")
    orient.add_argument("--demo", action="store_true", help="Use built-in demo tasks.")
    orient.add_argument("--limit", type=int, help="Limit number of tasks for a smoke run.")

    solve = sub.add_parser("solve", help="Run conservative primitive plans across tasks.")
    solve.add_argument("--data", type=Path, help="ARC task JSON file or directory.")
    solve.add_argument("--out", type=Path, default=Path("runs/solutions"), help="Output directory.")
    solve.add_argument("--demo", action="store_true", help="Use built-in demo tasks.")
    solve.add_argument("--limit", type=int, help="Limit number of tasks for a smoke run.")

    args = parser.parse_args()
    if args.cmd == "orient":
        if args.demo:
            tasks = demo_tasks()
        elif args.data:
            tasks = load_tasks(args.data)
        else:
            raise SystemExit("Pass --data PATH or --demo.")
        if args.limit is not None:
            tasks = tasks[: args.limit]
        records = [orient_task(task) for task in tasks]
        write_reports(records, args.out)
        print(f"oriented {len(records)} tasks -> {args.out}")
    elif args.cmd == "solve":
        if args.demo:
            tasks = demo_tasks()
        elif args.data:
            tasks = load_tasks(args.data)
        else:
            raise SystemExit("Pass --data PATH or --demo.")
        if args.limit is not None:
            tasks = tasks[: args.limit]
        results = write_solutions(tasks, args.out)
        attempted = sum(1 for result in results if result.plan != "abstain")
        print(f"solved/attempted {attempted}/{len(results)} tasks -> {args.out}")


if __name__ == "__main__":
    main()

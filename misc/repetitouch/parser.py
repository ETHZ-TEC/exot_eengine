#!/usr/bin/env python

from argparse import ArgumentParser
from functools import lru_cache
from json import dumps as jdumps
from pathlib import Path
from pprint import pprint
from re import findall
from sys import stderr
from typing import List, Tuple, Union

from invoke import Config, Context
from pandas import DataFrame, set_option
from toml import dumps as tdumps

DESCRIPTION = "Deserialises Repetitouch schedules and prints durations."
REQUIRED_EXECUTABLES = ["file", "java"]
RUNNER = Context(Config({"run": {"warn": True, "hide": True}}))
JAVA_DESERIALISER = Path(__file__).parent / "dumper.jar"
FIELDS = ["code", "deviceIndex", "timeInMicroSeconds", "time_sec", "time_usec", "type", "value"]


def eprint(*args, **kwargs):
    print(*args, file=stderr, **kwargs)


def get_available_executables(*executables: str) -> bool:
    return [RUNNER.run(f"command -v {_!s}", shell="bash").ok for _ in executables]


def is_java_data(file: str):
    _ = RUNNER.run(f"file -b -L {file!s}").stdout
    return "java serialization data" in _.lower()


def determine_duration(file: Union[Path, str]) -> float:
    return float(parse(file)[FIELDS[2]].iloc[-1]) / 1e6


@lru_cache(maxsize=5)
def deserialise(file: Path) -> str:
    if not isinstance(file, Path):
        raise TypeError(f"'file' must be of type {Path}, got: {type(file)}")

    _ = RUNNER.run(f"java -jar {JAVA_DESERIALISER!s} -r {file!s}")
    if not _.ok:
        raise RuntimeError(f"Could not deserialise Java data file {file!s}!")
    return _.stdout.rstrip()


def parse_raw(dump: str) -> List[Tuple[str]]:
    regex = r"\n.*".join([r"{}\n\s+\(\S+\)(\d+).*".format(_) for _ in FIELDS])
    return findall(regex, dump)


@lru_cache(maxsize=5)
def parse(file: Path) -> DataFrame:
    if not isinstance(file, Path):
        raise TypeError(f"'file' must be of type {Path}, got: {type(file)}")

    raw = deserialise(file)
    df = DataFrame(parse_raw(raw)).astype(int)
    df.columns = FIELDS
    return df


if __name__ == "__main__":
    available = get_available_executables(*REQUIRED_EXECUTABLES)
    if not all(available):
        missing = dict(zip(REQUIRED_EXECUTABLES, available))
        raise RuntimeError(
            f"at least one of the required executables does not exist: {missing}"
        )

    if not JAVA_DESERIALISER.exists():
        raise RuntimeError(f"the Java deserialiser not available at {JAVA_DESERIALISER!s}")

    parser = ArgumentParser(description=DESCRIPTION)
    parser.add_argument("command", choices=["list", "dump"], help="action to perform")

    formats = parser.add_mutually_exclusive_group()
    formats.add_argument("-t", "--toml", action="store_true", help="print as toml")
    formats.add_argument("-j", "--json", action="store_true", help="print as json")
    formats.add_argument("-c", "--csv", action="store_true", help="print as csv")

    parser.add_argument("files", nargs="+", type=str, help="the Repetitouch files")
    args = parser.parse_args()

    if args.command == "dump" and len(args.files) > 1:
        raise ValueError("'dump' can only work on individual records")

    for file in args.files:
        if not Path(file).exists():
            raise FileNotFoundError(f"Repetitouch file {file} not found!")
        if not is_java_data(file):
            raise RuntimeError(f"File {file} is not Java serialization data!")

    set_option("display.max_rows", None)
    set_option("display.max_colwidth", 120)

    if args.command == "list":
        files = []
        durations = []

        for file in args.files:
            file = Path(file)
            duration = determine_duration(file)
            eprint(f"Deserialised {file!s} with duration of {duration:.3f}s")
            files.append(str(file))
            durations.append(duration)

        if args.toml:
            print(tdumps({"schedules": files, "durations": durations}))
        elif args.json:
            print(jdumps({"schedules": files, "durations": durations}, indent=2))
        elif args.csv:
            _ = DataFrame({"schedule": files, "duration": durations})
            print(_.to_csv(index=False))
        else:
            pprint([[file, duration] for file, duration in zip(files, durations)])
    else:
        parsed = parse(Path(args.files[0]))

        if args.json:
            print(parsed.to_json(index=None, orient="split"))
        elif args.csv:
            print(parsed.to_csv(index=False))
        else:
            print(parsed)

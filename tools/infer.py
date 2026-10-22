import _bootstrap  # noqa: F401
from selectvp.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["infer", *__import__("sys").argv[1:]]))

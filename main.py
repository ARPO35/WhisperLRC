from __future__ import annotations

import sys

from whisperlrc.cli import run_interactive_menu


def main() -> int:
    try:
        return run_interactive_menu()
    except KeyboardInterrupt:
        print("\n已中断，程序退出。")
        return 130
    except Exception as e:
        print(f"\n程序异常退出：{e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

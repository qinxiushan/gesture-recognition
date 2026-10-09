"""Install the bundled pure-Python wheels without pip or ensurepip."""

import os
import sys
import zipfile


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: bootstrap_wheels.py PACKAGE_DIR")
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wheel_dir = os.path.join(project_dir, "wheelhouse")
    destination = os.path.abspath(sys.argv[1])
    wheels = sorted(name for name in os.listdir(wheel_dir) if name.endswith(".whl"))
    if not wheels:
        raise SystemExit("wheelhouse is empty")
    if not os.path.isdir(destination):
        os.makedirs(destination)
    for name in wheels:
        if (name.startswith("dataclasses-") or name.startswith("contextvars-") or
                name.startswith("immutables-") or name.startswith("contextlib2-")) and sys.version_info >= (3, 7):
            continue
        wheel_path = os.path.join(wheel_dir, name)
        with zipfile.ZipFile(wheel_path) as archive:
            if archive.testzip() is not None:
                raise SystemExit("corrupt wheel: " + name)
            for member in archive.namelist():
                target = os.path.abspath(os.path.join(destination, member))
                if not target.startswith(destination + os.sep):
                    raise SystemExit("unsafe wheel member: " + member)
            archive.extractall(destination)
        print("Installed", name)


if __name__ == "__main__":
    main()

import importlib
import subprocess
import sys


def is_package_installed(package_name):
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


def install_package(package_name, alias, min_version=None, max_version=None):
    try:
        version_spec = ""
        if min_version:
            version_spec += f">={min_version}"
        if max_version:
            version_spec += f",<={max_version}"

        package_with_version = f"{package_name}{version_spec}"

        subprocess.run(
            [sys.executable, "-m", "pip", "install", package_with_version, "--quiet"],
            capture_output=True,
            check=True,
        )

    except Exception as e:
        print(f"Failed to install package '{package_name}'. Error: {e}")


def install():
    # requirements
    deps = [
        ("beautifulsoup4", "bs4", "4.11.1", None),
        ("cachetools", None, None, None),
    ]

    for pkg in deps:
        if not is_package_installed(pkg[1] if pkg[1] else pkg[0]):
            install_package(*pkg)


try:
    import launch

    skip_install = launch.args.skip_install
except Exception:
    skip_install = False

if not skip_install:
    install()

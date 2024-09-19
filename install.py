import importlib.util
import subprocess
import sys

from civitai_assistant.logger import logger


def is_package_installed(package_name):
    return importlib.util.find_spec(package_name) is not None


def install_package(package_name, min_version=None, max_version=None):
    try:
        version_spec = ""
        if min_version:
            version_spec += f">={min_version}"
        if max_version:
            version_spec += f",<={max_version}"

        package_with_version = f"{package_name}{version_spec}"
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_with_version])

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install package '{package_name}'. Error: {e}")


def install():

    # requirements
    deps = [
        ("beautifulsoup4", "4.11.1", None),
        ("cachetools", None, None),
    ]

    for pkg in deps:
        if not is_package_installed(pkg[0]):
            install_package(*pkg)


try:
    import launch

    skip_install = launch.args.skip_install
except Exception:
    skip_install = False

if not skip_install:
    install()

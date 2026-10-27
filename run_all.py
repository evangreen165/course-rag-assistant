import subprocess
import sys


def main():
    processes = [
        subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api.main:app", "--port", "8000"]
        ),
        subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "ui/app.py"]
        ),
    ]
    try:
        for process in processes:
            process.wait()
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()


if __name__ == "__main__":
    main()

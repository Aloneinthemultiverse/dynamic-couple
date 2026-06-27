"""Kaggle T4x2 driver. Mounts model weights + the couple package, runs the SWE-bench loop.

Push with kaggle/push.sh. Requires the `couple` package uploaded as a utility dataset
(or pip-installed from the repo) and both model weights as Kaggle Models.
"""
# import sys; sys.path.append("/kaggle/input/dynamic-couple-src/src")
# from couple.runtime.loader import load_4bit, QWYTHOS_PATH, GEMMA_PATH
# from couple.loop.controller import run_task
# ... load both models (cuda:0 / cuda:1), build GraphCouple, run SWE-bench instances ...

if __name__ == "__main__":
    print("dynamic-couple driver — TODO: wire loader + controller + SWE-bench loop")

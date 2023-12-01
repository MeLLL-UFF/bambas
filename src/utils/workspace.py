import os

def get_workdir():
    try:
        from google.colab import drive
        workdir = os.getenv("COLAB_WORKDIR")
        if workdir is None:
            raise Exception("To execute on Google Colab you must provide the 'COLAB_WORKDIR' environment variable")
        print("Running on Google Colab, workdir:", workdir)
        return workdir
    except BaseException:
        print("Running outside Google Colab")
        return "."
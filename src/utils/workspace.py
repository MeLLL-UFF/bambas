import os

def get_workdir():
    try:
        from google.colab import drive
        workdir = os.getenv("COLAB_WORKDIR")
        if workdir is None:
            raise Exception("To execute on Google Colab you must provide the 'COLAB_WORKDIR' environment variable")
        return workdir
    except BaseException:
        return "."
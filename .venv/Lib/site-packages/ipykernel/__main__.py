"""The cli entry point for ipykernel."""
if __name__ == "__main__":
    from ipykernel import kernelapp as app

    app.launch_new_instance()

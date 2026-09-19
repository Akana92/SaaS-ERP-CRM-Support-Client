"""Run the installed HF CLI with the native Windows TLS trust store."""
import sys


def main():
    # Application entrypoint only: inject before HF/requests import SSLContext.
    if sys.platform == "win32":
        import truststore

        truststore.inject_into_ssl()
    from huggingface_hub.cli.hf import main as hf_main

    hf_main()


if __name__ == "__main__":
    main()

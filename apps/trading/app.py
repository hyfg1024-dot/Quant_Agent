"""Trading app entrypoint."""


def main() -> None:
    # Trading UI keeps page assembly logic; importing it runs the app layout.
    import trading_ui  # noqa: F401


if __name__ == "__main__":
    main()

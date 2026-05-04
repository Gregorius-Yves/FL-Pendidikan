import argparse
from dataclasses import dataclass

from client import run_client


@dataclass
class ClientContext:
    name: str
    client_id: int


def main():
    parser = argparse.ArgumentParser(description="FL Client")
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--server_host", default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=9999)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--name", default="Kelompok 1")
    args = parser.parse_args()

    client_context = ClientContext(
        name=args.name,
        client_id=args.client_id,
    )

    run_client(
        client_context=client_context,
        server_host=args.server_host,
        server_port=args.server_port,
        rounds=args.rounds,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()

import os


def main() -> None:
	mode = (os.getenv("PRODUCER_MODE", "csv") or "csv").strip().lower()

	if mode in {"csv", "replay"}:
		from app.producer.replay_counts_producer import main as replay_main

		replay_main()
		return

	if mode in {"yolo", "yolo_folder", "folder"}:
		from app.producer.yolo_folder_producer import main as yolo_main

		yolo_main()
		return

	raise ValueError(
		"Unsupported PRODUCER_MODE. Use one of: csv|replay, yolo|yolo_folder|folder. "
		f"Got: {mode!r}"
	)


if __name__ == "__main__":
	main()


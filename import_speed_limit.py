from time import sleep, time

from config import DAILY_IMPORT_SPEED, MIN_SLEEP_AFTER_IMPORT


class ImportSpeedLimit:
    def __init__(self) -> None:
        self._start = time()

    def sleep(self, size: int) -> None:
        end = self._start + size / DAILY_IMPORT_SPEED * 24 * 3600
        sleep_duration = end - time()

        if sleep_duration < MIN_SLEEP_AFTER_IMPORT:
            print(f'[SLEEP-IMPORT] 💤 Sleeping for {MIN_SLEEP_AFTER_IMPORT:.0f} seconds...')
            sleep(MIN_SLEEP_AFTER_IMPORT)
            self._start = end + (MIN_SLEEP_AFTER_IMPORT - sleep_duration)
        else:
            print(f'[SLEEP-IMPORT] 💤 Sleeping for {sleep_duration:.0f} seconds...')
            sleep(sleep_duration)
            self._start = end

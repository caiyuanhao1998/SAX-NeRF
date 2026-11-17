"""Persistent successful uses of the hosted SAX-NeRF demo."""
import copy
import html
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

TIMEZONE = 'America/Los_Angeles'
CHANNEL = 'sax-nerf'
LOG = logging.getLogger(__name__)


class HubStore:
    def __init__(self, repo, token):
        from huggingface_hub import HfApi
        self.repo, self.token = repo, token
        self.api = HfApi(token=token)

    def read(self):
        info = self.api.dataset_info(self.repo, timeout=8)
        if not info.private:
            raise ValueError('Usage storage must be private')
        request = Request(
            f'https://huggingface.co/datasets/{self.repo}/raw/{info.sha}/counts.json',
            headers={'Authorization': f'Bearer {self.token}'},
        )
        with urlopen(request, timeout=8) as response:
            data = json.load(response)
        if data.get('schema_version') != 1 or data.get('timezone') != TIMEZONE:
            raise ValueError('Unexpected usage schema')
        return data, info.sha

    def commit(self, data, parent):
        from huggingface_hub import CommitOperationAdd
        self.api.create_commit(
            repo_id=self.repo, repo_type='dataset', parent_commit=parent,
            commit_message='Update SAX-NeRF successful use counts',
            operations=[CommitOperationAdd(
                path_in_repo='counts.json',
                path_or_fileobj=(json.dumps(data, indent=2) + '\n').encode(),
            )],
        )


class UsageCounter:
    def __init__(self, store):
        self.store, self.cached, self.cache_time = store, None, 0
        self.lock = threading.RLock()

    @classmethod
    def from_env(cls):
        token_name = os.environ.get('SPACE_STATS_TOKEN_NAME', 'SPACE_STATS_TOKEN')
        token = os.environ.get(token_name)
        repo = os.environ.get('SPACE_STATS_REPO', 'CaiYuanhao/space-success-stats')
        return cls(HubStore(repo, token) if token else None)

    def _update(self, mutate):
        if self.store is None:
            raise RuntimeError('SPACE_STATS_TOKEN is not configured')
        with self.lock:
            for attempt in range(5):
                try:
                    data, revision = self.store.read()
                    if mutate(data):
                        self.store.commit(data, revision)
                    self.cached = copy.deepcopy(data)
                    self.cache_time = time.monotonic()
                    return True
                except Exception as error:
                    status = getattr(getattr(error, 'response', None), 'status_code', None)
                    status = status or getattr(error, 'code', None)
                    if status in (400, 401, 403, 404) or attempt == 4:
                        raise
                    time.sleep(.1 * 2 ** attempt)

    def start(self):
        now = datetime.now(timezone.utc).isoformat()
        def initialize(data):
            changed = False
            if CHANNEL not in data['started_at']:
                data['started_at'][CHANNEL] = now;changed = True
            return changed
        try:
            return self._update(initialize)
        except Exception as error:
            LOG.warning('Usage initialization unavailable (%s)', type(error).__name__)
            return False

    def record(self, *, now=None, event_id=None):
        now = now or datetime.now(timezone.utc)
        day = now.astimezone(ZoneInfo(TIMEZONE)).date().isoformat()
        event_id = event_id or str(uuid.uuid4())
        def increment(data):
            if event_id in data['recent_event_ids']:
                return False
            data['started_at'].setdefault(CHANNEL, now.isoformat())
            counts = data['days'].setdefault(day, {})
            counts[CHANNEL] = counts.get(CHANNEL, 0) + 1
            data['recent_event_ids'] = (data['recent_event_ids'] + [event_id])[-1024:]
            return True
        try:
            return self._update(increment)
        except Exception as error:
            # Statistics must not interrupt volume viewing or GPU inference.
            LOG.warning('Usage update unavailable (%s)', type(error).__name__)
            return False

    def summary(self, *, now=None):
        with self.lock:
            if self.store is not None and time.monotonic() - self.cache_time > 30:
                try:
                    self.cached, _ = self.store.read()
                    self.cache_time = time.monotonic()
                except Exception:
                    return None
            if self.cached is None:
                return None
            day = (now or datetime.now(timezone.utc)).astimezone(ZoneInfo(TIMEZONE)).date()
            first = day - timedelta(days=29)
            days = self.cached['days']
            return dict(today=days.get(day.isoformat(), {}).get(CHANNEL, 0),
                        last_30_days=sum(counts.get(CHANNEL, 0) for date, counts in days.items()
                                         if first.isoformat() <= date <= day.isoformat()),
                        since=self.cached['started_at'].get(CHANNEL), timezone=TIMEZONE)

    def badges(self):
        data = self.summary()
        values = [('Uses · 30 days', data['last_30_days'] if data else None),
                  ('Today', data['today'] if data else None)]
        chips = ''.join(
            '<span class="usage-chip"><span>' + html.escape(label) + '</span><b>'
            + (f'{value:,}' if value is not None else '—') + '</b></span>'
            for label, value in values
        )
        return ('<div class="usage-counts" aria-label="Successful uses" '
                'title="Successful user-requested X-ray renders and mesh exports; America/Los_Angeles. '
                '30 days includes today and the preceding 29 dates.">' + chips + '</div>')

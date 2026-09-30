"""Paired slow HTTP lab traffic with bounded parameters and packet evidence."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import http.client
import math
import random
import socket
import time

from campaign_runtime import BudgetExceeded, MAX_OUTBOUND_BYTES, VICTIM_IP


def configured_profile(profile, *, duration=20, interval=.6, connections=4, padding=0, jitter=0):
    numbers = (duration, interval, jitter)
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in numbers):
        raise ValueError('slow timing values must be finite')
    if not 2 <= duration <= 25 or not .2 <= interval <= 2 or not 0 <= jitter <= .2:
        raise ValueError('duration 2..25s, interval .2..2s, jitter 0..0.2 required')
    if type(connections) is not int or not 1 <= connections <= 8:
        raise ValueError('connections must be 1..8')
    if type(padding) is not int or not 0 <= padding <= 128:
        raise ValueError('padding must be 0..128')
    fragment = b'X-Lab-Pad: ' + b'x' * (1 + padding) + b'\r\n'
    maximum_fragments = math.ceil(duration / (interval * (1 - jitter))) + 2
    reservation = 120 + maximum_fragments * len(fragment)
    if reservation * connections > MAX_OUTBOUND_BYTES:
        raise ValueError('slow profile exceeds the lab outbound byte budget')
    return replace(profile, max_seconds=math.ceil(duration + 6), max_actions=connections,
        max_outbound_bytes=reservation * connections, parameters={
            'duration_seconds': duration, 'fragment_interval_seconds': interval,
            'connections': connections, 'header_padding_bytes': padding, 'jitter_fraction': jitter})


def run_slow(campaign, _):
    config = campaign.profile.parameters
    complete = campaign.profile.class_label == 'Benign'
    configured = configured_profile(campaign.profile,
        duration=config['duration_seconds'], interval=config['fragment_interval_seconds'],
        connections=config['connections'], padding=config.get('header_padding_bytes', 0),
        jitter=config.get('jitter_fraction', 0))
    fragment = b'X-Lab-Pad: ' + b'x' * (1 + configured.parameters['header_padding_bytes']) + b'\r\n'
    reservation = configured.max_outbound_bytes // config['connections']

    def one_connection(index):
        rng = random.Random(campaign.seed + index)
        try:
            campaign.reserve(estimated_outbound_bytes=reservation)
        except BudgetExceeded as error:
            campaign.note_error(str(error))
            return
        started = time.time()
        evidence = {'connection_index': index, 'headers_completed': False,
                    'fragments_sent': 0, 'http_status': None, 'client_port': None}
        response_bytes = 0
        success = False
        detail = ''
        try:
            with socket.create_connection((VICTIM_IP, 80), timeout=2) as connection:
                connection.settimeout(2)
                evidence['client_port'] = connection.getsockname()[1]
                # Identical path, Host and fragment format across both classes.
                connection.sendall(b'GET /lab/load HTTP/1.1\r\nHost: 192.168.100.10\r\n')
                end = min(time.monotonic() + config['duration_seconds'], campaign.deadline - 3)
                while time.monotonic() < end:
                    connection.sendall(fragment)
                    evidence['fragments_sent'] += 1
                    delay = config['fragment_interval_seconds'] * rng.uniform(
                        1 - configured.parameters['jitter_fraction'], 1 + configured.parameters['jitter_fraction'])
                    campaign.pause(min(delay, max(0, end - time.monotonic())))
                if complete:
                    connection.sendall(b'\r\n')
                    evidence['headers_completed'] = True
                    response = http.client.HTTPResponse(connection)
                    response.begin()
                    response_bytes = len(response.read(4097))
                    evidence['http_status'] = response.status
                    success = response.status == 200 and response_bytes <= 4096
                    detail = f'HTTP {response.status}; {evidence["fragments_sent"]} fragments'
                else:
                    success = evidence['fragments_sent'] > 0
                    detail = f'{evidence["fragments_sent"]} unfinished header fragments'
        except (OSError, http.client.HTTPException) as error:
            detail = type(error).__name__
        campaign.record('slow_http_complete' if complete else 'slow_http_headers', started,
            success=success, response_bytes=response_bytes, detail=detail, evidence=evidence)

    with ThreadPoolExecutor(max_workers=config['connections']) as pool:
        list(pool.map(one_connection, range(config['connections'])))

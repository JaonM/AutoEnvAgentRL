"""Opt-in HTTPS transport for the standard OpenAI Codex provider.

No global Codex configuration or credentials are modified. Custom endpoints and
profiles keep their existing behavior in auto mode; HTTPS mode refuses to
silently replace them. Both modes retain the CLI's sandbox and authentication.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tomllib
from collections.abc import Mapping

TRANSPORT_ENV = 'ENVFACTORY_CODEX_TRANSPORT'


def transport_args(environment: Mapping[str, str] | None = None) -> list[str]:
    env = os.environ if environment is None else environment
    mode = env.get(TRANSPORT_ENV, 'auto')
    if mode == 'auto':
        return []
    if mode != 'https':
        raise ValueError(f'{TRANSPORT_ENV} must be auto or https')
    config_home = Path(env.get('CODEX_HOME', str(Path.home() / '.codex')))
    config_file = config_home / 'config.toml'
    config = tomllib.loads(config_file.read_text()) if config_file.is_file() else {}
    if (config.get('model_provider', 'openai') != 'openai' or config.get('profile')
            or config.get('openai_base_url') or config.get('chatgpt_base_url')
            or env.get('OPENAI_BASE_URL')):
        raise ValueError('HTTPS transport requires the default OpenAI provider and endpoints; '
                         f'use {TRANSPORT_ENV}=auto to preserve customized Codex routing')
    return ['-c', 'model_provider="envfactory_openai_https"',
            '-c', 'model_providers.envfactory_openai_https.name="OpenAI HTTPS"',
            '-c', 'model_providers.envfactory_openai_https.requires_openai_auth=true',
            '-c', 'model_providers.envfactory_openai_https.supports_websockets=false',
            '-c', 'model_providers.envfactory_openai_https.wire_api="responses"']


def main():
    # Shared shell entry point; exec keeps signals and exit status with Codex.
    args = transport_args()
    os.execvp('codex', ['codex', *args, *sys.argv[1:]])


if __name__ == '__main__':
    main()

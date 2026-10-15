"""Test-wide settings: manuals are written by the stand-in chat client, and nothing waits for DeepSeek's off-peak
hours."""
import os

os.environ.setdefault('ENVFACTORY_POLICY_WRITER', 'chat')
os.environ.setdefault('ENVFACTORY_OFFPEAK', '0')

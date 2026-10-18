"""A natural opening preserves the scene's quotation unit across languages."""
import pytest

from env_factory.production import generate, openings


@pytest.mark.parametrize(('language', 'text'), [
    ('en', 'The quotation is 125 cents. Can you explain the charge?'),
    ('en', 'The fare is 3200 euro cents; what would the refund be?'),
    ('en', 'My account has 4000 yen available.'),
    ('zh', '报价是125分，这笔费用能退吗？'),
    ('zh', '每项收25分，一共要付多少分？'),
])
def test_scene_units_are_not_language_style_errors(language, text):
    assert openings.red_flags(text, language) == []
    # Request/world self-checks use this wrapper, not just the opening writer.
    assert generate.opening_problems(text, language) == []


@pytest.mark.parametrize(('language', 'text', 'known'), [
    ('en', 'The quotation is 125 cents. Can you explain the charge?', {'quoted_cents': 125}),
    ('en', 'The fare is 3200 euro cents; what would the refund be?', {'quoted_euro_cents': 3200}),
    ('zh', '报价是125分，这笔费用能退吗？', {'报价分': 125}),
])
def test_writer_accepts_original_quotation_without_conversion_feedback(language, text, known):
    class Writer:
        def __init__(self):
            self.prompts = []

        def chat(self, messages, **kwargs):
            self.prompts.append(messages[-1]['content'])
            return {'content': text}

    writer = Writer()
    request = {'language': language, 'user_scene': {
        'situation': 'Checking a received quotation', 'goal': 'Understand the quoted charge',
        'known': known, 'wants_to_know': [],
    }}
    result, _, problems = generate.write_opening(writer, request, 'unit-preservation')
    assert problems == []
    assert result == text
    assert len(writer.prompts) == 1


@pytest.mark.parametrize(('language', 'text', 'flag'), [
    ('en', 'The quote is 125 cents. Wait for my approval before each step.', 'dictates the confirmation protocol'),
    ('en', 'The current time is 10 and the quote is 125 cents.', 'virtual clock'),
    ('zh', '报价125分，等我确认再操作。', '交代确认流程'),
])
def test_monetary_units_do_not_disable_other_opening_checks(language, text, flag):
    assert flag in openings.red_flags(text, language)

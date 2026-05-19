def test_test_harness_is_configured():
    assert True


def test_package_imports():
    import openai_prep
    import openai_prep.agents

    assert openai_prep is not None
    assert openai_prep.agents is not None

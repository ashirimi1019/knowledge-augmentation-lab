from streamlit.testing.v1 import AppTest


def test_live_rag_lab_warns_for_blank_question_without_exception() -> None:
    app = AppTest.from_file("app.py").run()
    app.radio[0].set_value("Live RAG lab").run()
    app.text_input[0].set_value("").run()

    app.button[0].click().run()

    assert not app.exception
    assert [warning.value for warning in app.warning] == ["Enter a question before running the pipeline."]

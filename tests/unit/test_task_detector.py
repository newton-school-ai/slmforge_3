from slmforge.task.detector import detect_task_type

def test_detect_classification():
    data = [
        {"text": "I love this movie", "label": "positive"},
        {"text": "I hate this movie", "label": "negative"}
    ]
    result = detect_task_type(data)
    assert result["task"] == "classification"
    assert result["confidence"] >= 0.9

def test_detect_qa():
    data = [
        {"question": "What is the capital of France?", "answer": "Paris", "context": "France is a country in Europe."},
        {"question": "Who is the president?", "answer": "Macron", "context": "France is a country in Europe."}
    ]
    result = detect_task_type(data)
    assert result["task"] == "qa"
    assert result["confidence"] >= 0.9

def test_detect_summarisation():
    data = [
        {"document": "This is a very long text about something important...", "summary": "Short summary"},
        {"document": "Another long document with many words...", "summary": "Another summary"}
    ]
    result = detect_task_type(data)
    assert result["task"] == "summarisation"
    assert result["confidence"] >= 0.9

def test_detect_instruction():
    data = [
        {"instruction": "Write a poem", "input": "About the sea", "response": "The sea is blue..."},
        {"instruction": "Translate to French", "input": "Hello", "response": "Bonjour"}
    ]
    result = detect_task_type(data)
    assert result["task"] == "instruction"
    assert result["confidence"] >= 0.9
    
def test_detect_chat():
    data = [
        {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
    ]
    result = detect_task_type(data)
    assert result["task"] == "chat"
    assert result["confidence"] >= 0.9

def test_fallback_prompt_unknown():
    data = [
        {"col1": "val1", "col2": "val2"}
    ]
    result = detect_task_type(data)
    assert result["task"] == "unknown"
    assert result["confidence"] == 0.0
    assert result["fallback_prompt"] is not None
    assert "Could not confidently determine" in result["fallback_prompt"]

def test_fallback_prompt_low_confidence():
    # Only label present, no other strong signals, but many unique labels (though we don't penalize yet)
    # Just checking the structure of output
    data = [
        {"feature_a": "value", "feature_b": "value", "label": "1"}
    ]
    # We added 0.6 for label, 0.3 for unique labels, so it's 0.9.
    # To get low confidence, let's create a partial match
    data = [
        {"summary": "test"} # no text/document
    ]
    # No heuristic triggers fully because summary requires text
    result = detect_task_type(data)
    assert result["confidence"] < 0.6
    assert result["fallback_prompt"] is not None

def test_alternatives_returned():
    # Mix of QA and Instruction to trigger both
    data = [
        {"question": "What?", "answer": "That", "instruction": "Answer this", "response": "That"}
    ]
    result = detect_task_type(data)
    # Both QA and Instruction will get points.
    assert len(result["alternatives"]) >= 1
    tasks_found = [result["task"]] + [alt["task"] for alt in result["alternatives"]]
    assert "qa" in tasks_found
    assert "instruction" in tasks_found

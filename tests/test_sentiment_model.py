from app.helper.sentiment_model import Model


def test_translate_default_language():
    instance = Model()
    result = instance._translate("Olá")
    assert result == "Hello"


def test_translate_custom_language():
    instance = Model()
    result = instance._translate("olá", target_lang="fr")
    assert result == "Bonjour"


def test_analyze_feeling_positive():
    instance = Model()
    result = instance._analyze_feeling("I love the moon")
    assert result == {"neg": 0.0, "neu": 0.323, "pos": 0.677, "compound": 0.6369}


def test_analyze_feeling_negative():
    instance = Model()
    result = instance._analyze_feeling("This product is horrible")
    assert result == {"neg": 0.538, "neu": 0.462, "pos": 0.0, "compound": -0.5423}


def test_analyze_feeling_neutral():
    instance = Model()
    result = instance._analyze_feeling("Today is Tuesday")
    assert result == {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}


def test_transform_feeling_positive():
    instance = Model()
    result = instance._transform_feeling(
        {"neg": 0.0, "neu": 0.323, "pos": 0.677, "compound": 0.6369}
    )
    assert result == "Positivo"


def test_transform_feeling_negative():
    instance = Model()
    result = instance._transform_feeling(
        {"neg": 0.538, "neu": 0.462, "pos": 0.0, "compound": -0.5423}
    )
    assert result == "Negativo"


def test_transform_feeling_neutral():
    instance = Model()
    result = instance._transform_feeling(
        {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
    )
    assert result == "Neutro"


def test_predict_positive():
    instance = Model()
    sentiment, compound, probabilities = instance.predict("Eu amo a lua")
    assert sentiment == "Positivo"
    assert compound == 0.6369
    assert probabilities == {"neg": 0.0, "neu": 0.323, "pos": 0.677}


def test_predict_negative():
    instance = Model()
    sentiment, compound, probabilities = instance.predict("Esse produto é horrivel")
    assert sentiment == "Negativo"
    assert compound == -0.5423
    assert probabilities == {"neg": 0.538, "neu": 0.462, "pos": 0.0}


def test_predict_neutral():
    instance = Model()
    sentiment, compound, probabilities = instance.predict("Hoje é terça")
    assert sentiment == "Neutro"
    assert compound == 0.0
    assert probabilities == {"neg": 0.0, "neu": 1.0, "pos": 0.0}

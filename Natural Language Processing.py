def analyze_text(text):
    entities = []
    key_phrases = []
    sentiment = None

    # Call the Text Analytics API for entity extraction
    entities_result = text_analytics_client.recognize_entities(text)[0]
    for entity in entities_result.entities:
        entities.append((entity.text, entity.category))

    # Call the Text Analytics API for key phrase extraction
    key_phrases_result = text_analytics_client.extract_key_phrases(text)[0]
    key_phrases = key_phrases_result.key_phrases

    # Call the Text Analytics API for sentiment analysis
    sentiment_result = text_analytics_client.analyze_sentiment(text)[0]
    sentiment = sentiment_result.sentiment

    return entities, key_phrases, sentiment

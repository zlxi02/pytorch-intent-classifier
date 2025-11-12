"""
Training data for intent classification.
Each entry is a tuple of (text, intent_label).
"""

TRAINING_DATA = [
    # Greeting (more variations)
    ("Hi there", "Greeting"),
    ("Hello, how are you", "Greeting"),
    ("hey", "Greeting"),
    ("greetings", "Greeting"),
    ("good morning", "Greeting"),
    ("hi", "Greeting"),
    
    # GetWeather (more variations)
    ("what is the weather like in London", "GetWeather"),
    ("check the forecast for Paris", "GetWeather"),
    ("how is the weather today", "GetWeather"),
    ("what is the temperature", "GetWeather"),
    ("weather forecast please", "GetWeather"),
    ("is it going to rain", "GetWeather"),
    
    # BookFlight (more variations)
    ("I need to book a flight", "BookFlight"),
    ("can you reserve me a ticket to Rome", "BookFlight"),
    ("book me a ticket", "BookFlight"),
    ("I want to fly to Paris", "BookFlight"),
    ("reserve a flight", "BookFlight"),
    ("need a plane ticket", "BookFlight"),
    
    # Thanking (more variations)
    ("Thank you very much", "Thanking"),
    ("thanks for the help", "Thanking"),
    ("thanks", "Thanking"),
    ("appreciate it", "Thanking"),
    ("thank you", "Thanking"),
    ("much appreciated", "Thanking"),
    
    # GetTime (more variations)
    ("what time is it", "GetTime"),
    ("current time please", "GetTime"),
    ("tell me the time", "GetTime"),
    ("what is the time", "GetTime"),
    ("time please", "GetTime"),
    ("current time", "GetTime"),
    
    # Farewell (more variations)
    ("bye bye", "Farewell"),
    ("see you later", "Farewell"),
    ("goodbye", "Farewell"),
    ("see ya", "Farewell"),
    ("catch you later", "Farewell"),
    ("bye", "Farewell")
]


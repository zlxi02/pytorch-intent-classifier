"""
Training data for intent classification.
Each entry is a tuple of (text, intent_label).
"""

TRAINING_DATA = [
    # Greeting - 10 examples
    ("Hi there", "Greeting"),
    ("Hello, how are you", "Greeting"),
    ("hey", "Greeting"),
    ("greetings", "Greeting"),
    ("good morning", "Greeting"),
    ("hi", "Greeting"),
    ("hello", "Greeting"),
    ("good afternoon", "Greeting"),
    ("howdy", "Greeting"),
    ("what's up", "Greeting"),
    
    # GetWeather - 10 examples
    ("what is the weather like in London", "GetWeather"),
    ("check the forecast for Paris", "GetWeather"),
    ("how is the weather today", "GetWeather"),
    ("what is the temperature", "GetWeather"),
    ("weather forecast please", "GetWeather"),
    ("is it going to rain", "GetWeather"),
    ("will it be sunny tomorrow", "GetWeather"),
    ("show me the weather", "GetWeather"),
    ("what is the climate like", "GetWeather"),
    ("any chance of snow", "GetWeather"),
    
    # BookFlight - 10 examples
    ("I need to book a flight", "BookFlight"),
    ("can you reserve me a ticket to Rome", "BookFlight"),
    ("book me a ticket", "BookFlight"),
    ("I want to fly to Paris", "BookFlight"),
    ("reserve a flight", "BookFlight"),
    ("need a plane ticket", "BookFlight"),
    ("get me a flight to Tokyo", "BookFlight"),
    ("I want to book a plane ticket", "BookFlight"),
    ("find me a flight", "BookFlight"),
    ("schedule a flight for me", "BookFlight"),
    
    # Thanking - 10 examples
    ("Thank you very much", "Thanking"),
    ("thanks for the help", "Thanking"),
    ("thanks", "Thanking"),
    ("appreciate it", "Thanking"),
    ("thank you", "Thanking"),
    ("much appreciated", "Thanking"),
    ("thanks a lot", "Thanking"),
    ("grateful for your help", "Thanking"),
    ("many thanks", "Thanking"),
    ("cheers", "Thanking"),
    
    # GetTime - 10 examples
    ("what time is it", "GetTime"),
    ("current time please", "GetTime"),
    ("tell me the time", "GetTime"),
    ("what is the time", "GetTime"),
    ("time please", "GetTime"),
    ("current time", "GetTime"),
    ("what time do you have", "GetTime"),
    ("can you tell me the time", "GetTime"),
    ("do you have the time", "GetTime"),
    ("what is the current time", "GetTime"),
    
    # Farewell - 10 examples
    ("bye bye", "Farewell"),
    ("see you later", "Farewell"),
    ("goodbye", "Farewell"),
    ("see ya", "Farewell"),
    ("catch you later", "Farewell"),
    ("bye", "Farewell"),
    ("talk to you later", "Farewell"),
    ("see you soon", "Farewell"),
    ("take care", "Farewell"),
    ("have a good day", "Farewell")
]

